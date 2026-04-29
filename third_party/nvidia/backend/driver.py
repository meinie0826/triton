import functools
import hashlib
import os
import re
import subprocess
import triton
import ctypes
import sys
from triton import knobs
from triton.runtime.build import compile_module_from_file, compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver, decompose_descriptor, expand_signature, wrap_handle_tensordesc_impl

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['libcuda.so.1']
PyCUtensorMap = None
PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None
GSAN_PER_DEVICE_STATE_STRIDE = 1 << 30


def _is_tvmffi_launcher_enabled():
    return os.environ.get("TRITON_ENABLE_TVM_FFI_LAUNCHER", "0") == "1"


def _import_tvm_ffi():
    try:
        import tvm_ffi
        return tvm_ffi
    except ImportError:
        repo_tvmffi = os.path.abspath(os.path.join(dirname, "..", "..", "..", "tvm-ffi", "python"))
        if os.path.isdir(os.path.join(repo_tvmffi, "tvm_ffi")) and repo_tvmffi not in sys.path:
            sys.path.insert(0, repo_tvmffi)
        import tvm_ffi
        return tvm_ffi


@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


def _cuda_driver_is_active():
    candidates = ["libcuda.so.1"]
    try:
        candidates.extend([os.path.join(path, "libcuda.so.1") for path in libcuda_dirs()])
    except Exception:
        pass

    libcuda = None
    for candidate in candidates:
        try:
            libcuda = ctypes.CDLL(candidate)
            break
        except OSError:
            continue

    if libcuda is None:
        return False

    cu_init = libcuda.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    if cu_init(0) != 0:
        return False

    cu_device_get_count = libcuda.cuDeviceGetCount
    cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_device_get_count.restype = ctypes.c_int
    count = ctypes.c_int()
    if cu_device_get_count(ctypes.byref(count)) != 0:
        return False

    return count.value > 0


# ------------------------
# Utils
# ------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_file(
            src_path=os.path.join(dirname, "driver.c"),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        global PyCUtensorMap
        global PyKernelArg
        global ARG_CONSTEXPR
        global ARG_KERNEL
        global ARG_TUPLE
        PyCUtensorMap = mod.PyCUtensorMap
        PyKernelArg = mod.PyKernelArg
        ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        ARG_KERNEL = mod.ARG_KERNEL
        ARG_TUPLE = mod.ARG_TUPLE
        self.load_binary = mod.load_binary
        self.unload_module = mod.unload_module
        self.get_current_device = mod.get_current_device
        self.set_current_device = mod.set_current_device
        self.get_default_stream = mod.get_default_stream
        self.get_device_capability = mod.get_device_capability
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.fill_tma_descriptor_tiled = mod.fill_tma_descriptor_tiled
        self.fill_tma_descriptor_im2col = mod.fill_tma_descriptor_im2col
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


def make_kernel_signature(signature):
    """
    Creates a kernel signature in C to be able to efficiently extract
    arguments in the launcher.
    """

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    flat_signature = []
    for sig in signature:
        _flatten_signature(sig, flat_signature)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]

    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_signature)


def _is_tensordesc_type(sig):
    return isinstance(sig, str) and sig.startswith("tensordesc")


def _parse_tensordesc_type(descriptor):
    match = re.match(r"tensordesc(?:_im2col)?<([^[>]*)\[([^\]]*)\]", descriptor)
    assert match, f"Malformed tensor descriptor type: {descriptor}"
    block_shape = match.group(2)
    block_ndim = block_shape.count(",") + 1
    rank_match = re.search(r",input_rank=(\d+)", descriptor)
    ndim = int(rank_match.group(1)) if rank_match else block_ndim
    return ndim


def _normalize_tvmffi_meta(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _normalize_tvmffi_meta(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_tvmffi_meta(v) for v in value)
    return value


def _denormalize_tvmffi_meta(value):
    if isinstance(value, tuple) and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in value):
        return {k: _denormalize_tvmffi_meta(v) for k, v in value}
    if isinstance(value, tuple):
        return [_denormalize_tvmffi_meta(v) for v in value]
    return value


def _make_tvmffi_host_arg_plan(signature, tensordesc_meta):
    has_tensordesc_meta = bool(tensordesc_meta)
    desc_idx = 0
    plan = []

    def visit(sig, path):
        nonlocal desc_idx
        if isinstance(sig, tuple):
            for i, nested in enumerate(sig):
                visit(nested, (*path, i))
            return
        if sig == "constexpr":
            return
        if _is_tensordesc_type(sig):
            if not has_tensordesc_meta:
                raise ValueError("TVM FFI TensorDescriptor launch requires tensordesc metadata")
            meta = tensordesc_meta[desc_idx]
            desc_idx += 1
            if meta.get("is_im2col", False):
                raise ValueError("TVM FFI TensorDescriptor im2col launch is not supported yet")
            plan.append(("tma", path, _parse_tensordesc_type(sig), _normalize_tvmffi_meta(meta)))
            return
        plan.append(("arg", path, sig))

    for i, sig in enumerate(signature):
        visit(sig, (i, ))
    if has_tensordesc_meta and desc_idx != len(tensordesc_meta):
        raise ValueError("TVM FFI TensorDescriptor metadata count mismatch")
    return tuple(plan)


def _tvmffi_type_info(sig):
    if sig[0] == "*":
        return "CUdeviceptr", "extractPointer"
    return {
        "i1": ("int32_t", "extractI32"),
        "i8": ("int8_t", "extractI8"),
        "i16": ("int16_t", "extractI16"),
        "i32": ("int32_t", "extractI32"),
        "i64": ("int64_t", "extractI64"),
        "u1": ("uint32_t", "extractU32"),
        "u8": ("uint8_t", "extractU8"),
        "u16": ("uint16_t", "extractU16"),
        "u32": ("uint32_t", "extractU32"),
        "u64": ("uint64_t", "extractU64"),
        "fp32": ("uint32_t", "extractFP32"),
        "f32": ("uint32_t", "extractFP32"),
        "fp64": ("uint64_t", "extractFP64"),
    }[sig]


def _supports_tvmffi_host_stub(signature):
    try:
        for entry in signature:
            if entry[0] == "arg":
                _tvmffi_type_info(entry[2])
            elif entry[0] != "tma":
                return False
    except (KeyError, IndexError, TypeError):
        return False
    return True


def _tvmffi_arg_plan_has_tma(arg_plan):
    return any(entry[0] == "tma" for entry in arg_plan)


def _get_path_value(args, path):
    cur = args[path[0]]
    for step in path[1:]:
        cur = cur[step]
    return cur


def _set_path_value(args, path, value):
    if len(path) == 1:
        args[path[0]] = value
        return
    parent = args[path[0]]
    parent_list = list(parent)
    _set_path_value(parent_list, path[1:], value)
    args[path[0]] = tuple(parent_list) if isinstance(parent, tuple) else parent_list


def _make_tvmffi_arg_converters(arg_plan):
    converters = []
    for entry in arg_plan:
        if entry[0] == "arg":
            sig = entry[2]
            if sig.startswith("u"):
                converters.append((entry[1], False))
    return tuple(converters)


def _prepare_tvmffi_kernel_args(args, converters):
    if not converters:
        return args
    prepared = None
    for path, is_ptr in converters:
        arg = _get_path_value(prepared if prepared is not None else args, path)
        if isinstance(arg, int) and (is_ptr or arg > (1 << 63) - 1):
            if prepared is None:
                prepared = list(args)
            _set_path_value(prepared, path, ctypes.c_void_p(arg))
    return args if prepared is None else tuple(prepared)


def _make_tvmffi_host_stub_src(module_name, arg_plan):
    lines = []
    param_idx = 0
    for i, entry in enumerate(arg_plan):
        kind = entry[0]
        path = entry[1]
        nested_path = path[1:]
        path_literal = ", ".join(map(str, nested_path)) if nested_path else "0"
        lines.append(f"  const TVMFFIAny *arg_{i} = getArgAtPath(args, num_args, {path[0] + 7}, "
                     f"(const int[]){{{path_literal}}}, {len(nested_path)});")
        if kind == "arg":
            sig = entry[2]
            ctype, extractor = _tvmffi_type_info(sig)
            lines.append(f"  {ctype} arg_storage_{i};")
            lines.append(f"  if (arg_{i} == NULL || !{extractor}(&arg_storage_{i}, arg_{i})) {{")
            lines.append("    return -2;")
            lines.append("  }")
            lines.append(f"  params[{param_idx}] = &arg_storage_{i};")
            param_idx += 1
        elif kind == "tma":
            ndim = entry[2]
            meta = _denormalize_tvmffi_meta(entry[3])
            block_size = ", ".join(str(int(x)) for x in meta["block_size"])
            elem_type = TMA_TF32 if meta.get("round_f32_to_tf32", False) else TMA_DTYPE_DEVICE_TO_HOST[meta["elem_type"]]
            lines.append(f"  CUtensorMap arg_storage_{i};")
            lines.append(f"  int32_t tma_shape_{i}[{ndim}];")
            lines.append(f"  int64_t tma_strides_{i}[{ndim}];")
            lines.append(
                f"  if (arg_{i} == NULL || !extractTmaDescriptor(&arg_storage_{i}, tma_shape_{i}, tma_strides_{i}, "
                f"{ndim}, arg_{i}, {int(meta['swizzle'])}, {int(meta['elem_size'])}, {int(elem_type)}, "
                f"(const uint32_t[]){{{block_size}}}, {len(meta['block_size'])}, {int(meta['fp4_padded'])})) {{")
            lines.append("    return -2;")
            lines.append("  }")
            lines.append(f"  params[{param_idx}] = &arg_storage_{i};")
            param_idx += 1
            for d in range(ndim):
                lines.append(f"  params[{param_idx}] = &tma_shape_{i}[{d}];")
                param_idx += 1
            for d in range(ndim):
                lines.append(f"  params[{param_idx}] = &tma_strides_{i}[{d}];")
                param_idx += 1

    num_kernel_params = param_idx
    param_count = num_kernel_params + 2
    params_decl = f"  void *params[{param_count}];" if param_count else "  void **params = NULL;"
    arg_extract = "\n".join(lines)

    return f"""
#include "cuda.h"
#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
}} DLDataType;

typedef struct {{
  int32_t device_type;
  int32_t device_id;
}} DLDevice;

typedef struct {{
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
}} DLTensor;

typedef struct {{
  uint64_t combined_ref_count;
  int32_t type_index;
  uint32_t __padding;
  union {{
    void (*deleter)(void *self, int flags);
    int64_t __ensure_align;
  }};
}} TVMFFIObject;

typedef struct {{
  int32_t type_index;
  union {{
    uint32_t zero_padding;
    uint32_t small_str_len;
  }};
  union {{
    int64_t v_int64;
    double v_float64;
    void *v_ptr;
    const char *v_c_str;
    TVMFFIObject *v_obj;
    DLDataType v_dtype;
    DLDevice v_device;
    char v_bytes[8];
    uint64_t v_uint64;
  }};
}} TVMFFIAny;

typedef struct {{
  void *data;
  int64_t size;
  int64_t capacity;
  void (*data_deleter)(void *data);
}} TVMFFISeqCell;

typedef struct {{
  void *handle;
}} TVMFFIOpaqueObjectCell;

enum {{
  kTVMFFINone = 0,
  kTVMFFIInt = 1,
  kTVMFFIBool = 2,
  kTVMFFIFloat = 3,
  kTVMFFIOpaquePtr = 4,
  kTVMFFIDLTensorPtr = 7,
  kTVMFFITensor = 70,
  kTVMFFIArray = 71,
  kTVMFFIOpaquePyObject = 74,
}};

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig *config,
                                       CUfunction f, void **kernelParams,
                                       void **extra);

typedef CUresult (*cuTensorMapEncodeTiled_t)(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);

static bool gpuAssert(CUresult code) {{
  if (code == CUDA_SUCCESS)
    return true;
  const char *str = NULL;
  cuGetErrorString(code, &str);
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyErr_Format(PyExc_RuntimeError, "Triton Error [CUDA]: %s",
               str ? str : "Unknown error");
  PyGILState_Release(gil_state);
  return false;
}}

static cuLaunchKernelEx_t getLaunchKernelExHandle(void) {{
  void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!libHandle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");
    return NULL;
  }}
  dlerror();
  cuLaunchKernelEx_t funcHandle =
      (cuLaunchKernelEx_t)dlsym(libHandle, "cuLaunchKernelEx");
  const char *err = dlerror();
  if (err) {{
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to retrieve cuLaunchKernelEx from libcuda.so.1");
    dlclose(libHandle);
    return NULL;
  }}
  return funcHandle;
}}

static cuTensorMapEncodeTiled_t getCuTensorMapEncodeTiledHandle(void) {{
  void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!libHandle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");
    return NULL;
  }}
  dlerror();
  cuTensorMapEncodeTiled_t funcHandle =
      (cuTensorMapEncodeTiled_t)dlsym(libHandle, "cuTensorMapEncodeTiled");
  const char *err = dlerror();
  if (err) {{
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to retrieve cuTensorMapEncodeTiled from libcuda.so.1");
    dlclose(libHandle);
    return NULL;
  }}
  return funcHandle;
}}

static void launchKernel(int gridX, int gridY, int gridZ, int num_warps,
                         int num_ctas, int launch_cooperative_grid,
                         int launch_pdl, int shared_memory, CUstream stream,
                         CUfunction function, void **params) {{
  if (gridX * gridY * gridZ == 0)
    return;
  CUlaunchAttribute launchAttr[4];
  static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
  if (cuLaunchKernelExHandle == NULL) {{
    cuLaunchKernelExHandle = getLaunchKernelExHandle();
    if (cuLaunchKernelExHandle == NULL)
      return;
  }}
  CUlaunchConfig config;
  config.gridDimX = gridX * num_ctas;
  config.gridDimY = gridY;
  config.gridDimZ = gridZ;
  config.blockDimX = 32 * num_warps;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = shared_memory;
  config.hStream = stream;
  config.attrs = launchAttr;
  int num_attrs = 0;
  if (launch_pdl != 0) {{
    CUlaunchAttribute pdlAttr = {{
        .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        .value = 1}};
    launchAttr[num_attrs++] = pdlAttr;
  }}
  if (launch_cooperative_grid != 0) {{
    CUlaunchAttribute coopAttr = {{.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                                  .value = 1}};
    launchAttr[num_attrs++] = coopAttr;
  }}
  if (num_ctas != 1) {{
    CUlaunchAttribute clusterAttr = {{}};
    clusterAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    clusterAttr.value.clusterDim.x = num_ctas;
    clusterAttr.value.clusterDim.y = 1;
    clusterAttr.value.clusterDim.z = 1;
    launchAttr[num_attrs++] = clusterAttr;
    CUlaunchAttribute clusterSchedulingAttr = {{}};
    clusterSchedulingAttr.id =
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    clusterSchedulingAttr.value.clusterSchedulingPolicyPreference =
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    launchAttr[num_attrs++] = clusterSchedulingAttr;
  }}
  config.numAttrs = num_attrs;
  if (num_ctas == 16 &&
      !gpuAssert(cuFuncSetAttribute(
          function, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1))) {{
    return;
  }}
  gpuAssert(cuLaunchKernelExHandle(&config, function, params, 0));
}}

static DLTensor *getDLTensor(const TVMFFIAny *arg) {{
  if (arg->type_index == kTVMFFIDLTensorPtr)
    return (DLTensor *)arg->v_ptr;
  if (arg->type_index == kTVMFFITensor)
    return (DLTensor *)((char *)arg->v_obj + sizeof(TVMFFIObject));
  return NULL;
}}

static const TVMFFIAny *getArrayItem(const TVMFFIAny *arg, int index) {{
  if (arg->type_index != kTVMFFIArray || arg->v_obj == NULL)
    return NULL;
  TVMFFISeqCell *cell = (TVMFFISeqCell *)((char *)arg->v_obj + sizeof(TVMFFIObject));
  if (index < 0 || index >= cell->size)
    return NULL;
  return &((TVMFFIAny *)cell->data)[index];
}}

static const TVMFFIAny *getArgAtPath(const TVMFFIAny *args, int32_t num_args,
                                     int root, const int *path,
                                     int path_size) {{
  if (root < 0 || root >= num_args)
    return NULL;
  const TVMFFIAny *cur = &args[root];
  for (int i = 0; i < path_size; ++i) {{
    cur = getArrayItem(cur, path[i]);
    if (cur == NULL)
      return NULL;
  }}
  return cur;
}}

static PyObject *getOpaquePyObject(const TVMFFIAny *arg) {{
  if (arg->type_index != kTVMFFIOpaquePyObject || arg->v_obj == NULL) {{
    PyErr_SetString(PyExc_TypeError, "expected opaque Python TensorDescriptor");
    return NULL;
  }}
  TVMFFIOpaqueObjectCell *cell =
      (TVMFFIOpaqueObjectCell *)((char *)arg->v_obj + sizeof(TVMFFIObject));
  return (PyObject *)cell->handle;
}}

static bool getPySequenceI64(PyObject *obj, int64_t *out, int expected_size,
                             const char *name) {{
  PyObject *fast = PySequence_Fast(obj, name);
  if (fast == NULL)
    return false;
  Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
  if (size != expected_size) {{
    PyErr_Format(PyExc_RuntimeError, "%s rank mismatch", name);
    Py_DECREF(fast);
    return false;
  }}
  for (int i = 0; i < expected_size; ++i) {{
    PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
    if (!PyLong_Check(item)) {{
      PyErr_Format(PyExc_TypeError, "%s element must be an int", name);
      Py_DECREF(fast);
      return false;
    }}
    out[i] = PyLong_AsLongLong(item);
    if (PyErr_Occurred()) {{
      Py_DECREF(fast);
      return false;
    }}
  }}
  Py_DECREF(fast);
  return true;
}}

static bool extractTmaDescriptor(CUtensorMap *desc, int32_t *shape_out,
                                 int64_t *strides_out, int rank,
                                 const TVMFFIAny *arg, int swizzle,
                                 int elem_size, int elem_type,
                                 const uint32_t *block_size,
                                 int block_rank, int fp4_padded) {{
  PyObject *obj = getOpaquePyObject(arg);
  if (obj == NULL)
    return false;

  PyObject *base = PyObject_GetAttrString(obj, "base");
  PyObject *shape = PyObject_GetAttrString(obj, "shape");
  PyObject *strides = PyObject_GetAttrString(obj, "strides");
  PyObject *padding = PyObject_GetAttrString(obj, "padding");
  PyObject *round_tf32 = PyObject_GetAttrString(obj, "round_f32_to_tf32");
  if (base == NULL || shape == NULL || strides == NULL || padding == NULL ||
      round_tf32 == NULL) {{
    Py_XDECREF(base);
    Py_XDECREF(shape);
    Py_XDECREF(strides);
    Py_XDECREF(padding);
    Py_XDECREF(round_tf32);
    return false;
  }}

  int64_t shape_i64[5] = {{0, 0, 0, 0, 0}};
  int64_t strides_i64[5] = {{0, 0, 0, 0, 0}};
  bool ok = getPySequenceI64(shape, shape_i64, rank, "shape") &&
            getPySequenceI64(strides, strides_i64, rank, "strides");
  if (!ok) {{
    Py_DECREF(base);
    Py_DECREF(shape);
    Py_DECREF(strides);
    Py_DECREF(padding);
    Py_DECREF(round_tf32);
    return false;
  }}

  PyObject *data_ptr = PyObject_CallMethod(base, "data_ptr", NULL);
  if (data_ptr == NULL) {{
    Py_DECREF(base);
    Py_DECREF(shape);
    Py_DECREF(strides);
    Py_DECREF(padding);
    Py_DECREF(round_tf32);
    return false;
  }}
  unsigned long long global_address = PyLong_AsUnsignedLongLong(data_ptr);
  Py_DECREF(data_ptr);
  if (PyErr_Occurred()) {{
    Py_DECREF(base);
    Py_DECREF(shape);
    Py_DECREF(strides);
    Py_DECREF(padding);
    Py_DECREF(round_tf32);
    return false;
  }}

  int padding_cmp = PyUnicode_Check(padding) ? PyUnicode_CompareWithASCIIString(padding, "nan") : 1;
  if (padding_cmp == -1 && PyErr_Occurred()) {{
    Py_DECREF(base);
    Py_DECREF(shape);
    Py_DECREF(strides);
    Py_DECREF(padding);
    Py_DECREF(round_tf32);
    return false;
  }}
  int use_nan_padding = padding_cmp == 0;
  int round_tf32_bool = PyObject_IsTrue(round_tf32);
  if (round_tf32_bool < 0) {{
    Py_DECREF(base);
    Py_DECREF(shape);
    Py_DECREF(strides);
    Py_DECREF(padding);
    Py_DECREF(round_tf32);
    return false;
  }}
  if (round_tf32_bool)
    elem_type = 11;

  uint64_t tma_shape[5] = {{0, 0, 0, 0, 0}};
  uint64_t tma_strides[5] = {{0, 0, 0, 0, 0}};
  uint32_t tma_block[5] = {{1, 1, 1, 1, 1}};
  uint32_t element_strides[5] = {{1, 1, 1, 1, 1}};
  for (int i = 0; i < rank; ++i) {{
    int rev = rank - i - 1;
    int64_t logical_shape = shape_i64[i];
    if (fp4_padded && i == rank - 1)
      logical_shape *= 2;
    tma_shape[rev] = (uint64_t)logical_shape;
    shape_out[i] = (int32_t)shape_i64[i];
    strides_out[i] = strides_i64[i];
  }}
  for (int i = 0; i + 1 < rank; ++i)
    tma_strides[rank - i - 2] = (uint64_t)(elem_size * strides_i64[i]);
  tma_strides[rank - 1] =
      tma_shape[rank - 1] *
      (rank == 1 ? (uint64_t)elem_size : tma_strides[rank - 2]);
  for (int i = 0; i < block_rank; ++i)
    tma_block[block_rank - i - 1] = block_size[i];

  Py_DECREF(base);
  Py_DECREF(shape);
  Py_DECREF(strides);
  Py_DECREF(padding);
  Py_DECREF(round_tf32);

  CUtensorMapFloatOOBfill fill =
      use_nan_padding ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                      : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
  static cuTensorMapEncodeTiled_t cuTensorMapEncodeTiled = NULL;
  if (cuTensorMapEncodeTiled == NULL) {{
    cuTensorMapEncodeTiled = getCuTensorMapEncodeTiledHandle();
    if (cuTensorMapEncodeTiled == NULL)
      return false;
  }}
  return gpuAssert(cuTensorMapEncodeTiled(
      desc, elem_type, rank, (void *)global_address, tma_shape, tma_strides,
      tma_block, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill));
}}

static bool extractPointer(void *ptr, const TVMFFIAny *arg) {{
  CUdeviceptr *dev_ptr = (CUdeviceptr *)ptr;
  if (arg->type_index == kTVMFFINone) {{
    *dev_ptr = (CUdeviceptr)0;
    return true;
  }}
  if (arg->type_index == kTVMFFIOpaquePtr) {{
    *dev_ptr = (CUdeviceptr)arg->v_ptr;
  }} else if (arg->type_index == kTVMFFIInt) {{
    *dev_ptr = (CUdeviceptr)arg->v_uint64;
  }} else {{
    DLTensor *tensor = getDLTensor(arg);
    if (tensor == NULL) {{
      PyGILState_STATE gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_TypeError,
                      "pointer argument must be None, int, opaque pointer, or tensor");
      PyGILState_Release(gil_state);
      return false;
    }}
    *dev_ptr = (CUdeviceptr)((char *)tensor->data + tensor->byte_offset);
  }}
  if (*dev_ptr == 0)
    return true;
  CUresult status =
      cuPointerGetAttribute(dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {{
    PyGILState_STATE gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_ValueError,
                    "Pointer argument cannot be accessed from Triton (cpu tensor?)");
    PyGILState_Release(gil_state);
    return false;
  }}
  return gpuAssert(status);
}}

static bool getInt64(const TVMFFIAny *arg, int64_t *out) {{
  if (arg->type_index == kTVMFFIInt || arg->type_index == kTVMFFIBool) {{
    *out = arg->v_int64;
    return true;
  }}
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_TypeError, "expected integer argument");
  PyGILState_Release(gil_state);
  return false;
}}

static bool getUInt64(const TVMFFIAny *arg, uint64_t *out) {{
  if (arg->type_index == kTVMFFIOpaquePtr) {{
    *out = (uint64_t)arg->v_ptr;
    return true;
  }}
  int64_t val;
  if (!getInt64(arg, &val))
    return false;
  *out = (uint64_t)val;
  return true;
}}

static bool getDouble(const TVMFFIAny *arg, double *out) {{
  if (arg->type_index == kTVMFFIFloat) {{
    *out = arg->v_float64;
    return true;
  }}
  if (arg->type_index == kTVMFFIInt || arg->type_index == kTVMFFIBool) {{
    *out = (double)arg->v_int64;
    return true;
  }}
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_TypeError, "expected floating point argument");
  PyGILState_Release(gil_state);
  return false;
}}

static bool extractI8(void *ptr, const TVMFFIAny *arg) {{
  int64_t val;
  if (!getInt64(arg, &val))
    return false;
  *((int8_t *)ptr) = (int8_t)val;
  return true;
}}
static bool extractI16(void *ptr, const TVMFFIAny *arg) {{
  int64_t val;
  if (!getInt64(arg, &val))
    return false;
  *((int16_t *)ptr) = (int16_t)val;
  return true;
}}
static bool extractI32(void *ptr, const TVMFFIAny *arg) {{
  int64_t val;
  if (!getInt64(arg, &val))
    return false;
  *((int32_t *)ptr) = (int32_t)val;
  return true;
}}
static bool extractI64(void *ptr, const TVMFFIAny *arg) {{
  return getInt64(arg, (int64_t *)ptr);
}}
static bool extractU8(void *ptr, const TVMFFIAny *arg) {{
  uint64_t val;
  if (!getUInt64(arg, &val))
    return false;
  *((uint8_t *)ptr) = (uint8_t)val;
  return true;
}}
static bool extractU16(void *ptr, const TVMFFIAny *arg) {{
  uint64_t val;
  if (!getUInt64(arg, &val))
    return false;
  *((uint16_t *)ptr) = (uint16_t)val;
  return true;
}}
static bool extractU32(void *ptr, const TVMFFIAny *arg) {{
  uint64_t val;
  if (!getUInt64(arg, &val))
    return false;
  *((uint32_t *)ptr) = (uint32_t)val;
  return true;
}}
static bool extractU64(void *ptr, const TVMFFIAny *arg) {{
  return getUInt64(arg, (uint64_t *)ptr);
}}
static bool extractFP32(void *ptr, const TVMFFIAny *arg) {{
  double val;
  if (!getDouble(arg, &val))
    return false;
  float f32 = (float)val;
  *((uint32_t *)ptr) = *(uint32_t *)&f32;
  return true;
}}
static bool extractFP64(void *ptr, const TVMFFIAny *arg) {{
  double val;
  if (!getDouble(arg, &val))
    return false;
  *((uint64_t *)ptr) = *(uint64_t *)&val;
  return true;
}}

static int tvmffiLaunchKernel(void *self, const TVMFFIAny *args,
                              int32_t num_args, TVMFFIAny *result) {{
  (void)self;
  if (num_args < 7) {{
    PyGILState_STATE gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_TypeError,
                    "Triton TVM FFI launcher expects at least 7 arguments");
    PyGILState_Release(gil_state);
    return -2;
  }}
  uint64_t function, stream;
  int64_t gridX, gridY, gridZ;
  if (!getUInt64(&args[0], &function) || !getUInt64(&args[1], &stream) ||
      !getInt64(&args[2], &gridX) || !getInt64(&args[3], &gridY) ||
      !getInt64(&args[4], &gridZ)) {{
    return -2;
  }}
{params_decl}
{arg_extract}
  CUdeviceptr global_scratch;
  if (!extractPointer(&global_scratch, &args[5]))
    return -2;
  params[{num_kernel_params}] = &global_scratch;
  CUdeviceptr profile_scratch;
  if (!extractPointer(&profile_scratch, &args[6]))
    return -2;
  params[{num_kernel_params + 1}] = &profile_scratch;

  launchKernel((int)gridX, (int)gridY, (int)gridZ, {module_name}_NUM_WARPS,
               {module_name}_NUM_CTAS, {module_name}_LAUNCH_COOPERATIVE_GRID,
               {module_name}_LAUNCH_PDL, {module_name}_SHARED_MEMORY,
               (CUstream)stream, (CUfunction)function, params);
  if (PyErr_Occurred())
    return -2;
  result->type_index = kTVMFFINone;
  result->zero_padding = 0;
  return 0;
}}

static PyObject *getTVMFFILaunchSymbol(PyObject *self, PyObject *args) {{
  return PyLong_FromUnsignedLongLong(
      (unsigned long long)(uintptr_t)&tvmffiLaunchKernel);
}}

static PyMethodDef ModuleMethods[] = {{
    {{"get_tvmffi_launch_symbol", getTVMFFILaunchSymbol, METH_VARARGS,
     "Return the generated TVM FFI CUDA launch safe-call symbol"}},
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef ModuleDef = {{PyModuleDef_HEAD_INIT, "{module_name}",
                                       NULL, -1, ModuleMethods}};

PyMODINIT_FUNC PyInit_{module_name}(void) {{
  return PyModule_Create(&ModuleDef);
}}
"""


@functools.lru_cache(maxsize=None)
def _compile_tvmffi_host_stub(arg_plan, num_warps, num_ctas, shared, launch_cooperative_grid, launch_pdl):
    key = repr(("v1", arg_plan, num_warps, num_ctas, shared, launch_cooperative_grid, launch_pdl)).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:16]
    module_name = f"cuda_tvmffi_host_stub_{digest}"
    src = _make_tvmffi_host_stub_src(module_name, arg_plan)
    defines = [
        f"-D{module_name}_NUM_WARPS={int(num_warps)}",
        f"-D{module_name}_NUM_CTAS={int(num_ctas)}",
        f"-D{module_name}_SHARED_MEMORY={int(shared)}",
        f"-D{module_name}_LAUNCH_COOPERATIVE_GRID={int(launch_cooperative_grid)}",
        f"-D{module_name}_LAUNCH_PDL={int(launch_pdl)}",
    ]
    return compile_module_from_src(
        src,
        module_name,
        library_dirs=library_dirs(),
        include_dirs=include_dirs,
        libraries=libraries,
        ccflags=defines,
    )


def annotate_arguments(signature):
    """
    This recreates the signature with annotations as C objects which can then
    be used to efficiently flatten tuples, and remove constexpr in the launcher.
    """
    annotated_arguments = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated_arguments.append((PyKernelArg(nested_tuple=annotate_arguments(sig), type=ARG_TUPLE)))
        elif sig != "constexpr":
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated_arguments


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9
TMA_TF32 = 11


def make_tensordesc_arg(arg, metadata, _):
    if metadata is None:
        return decompose_descriptor(arg)

    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]
    is_im2col = metadata.get("is_im2col", False)

    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1
    padding = 1 if arg.padding == "nan" else 0

    if fp4_padded:
        expanded_shape = list(shape)
        expanded_shape[-1] *= 2
    else:
        expanded_shape = shape

    if arg.round_f32_to_tf32:
        elem_type = TMA_TF32

    if is_im2col:
        # Im2col mode - use im2col descriptor fill function
        # block_size from metadata is [pixelsPerColumn, channelsPerPixel] (possibly clamped)
        element_strides = arg.element_strides if arg.element_strides is not None else [1] * len(shape)
        cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_im2col(
            arg.base.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            expanded_shape,
            strides,
            padding,
            arg.pixel_box_lower_corner,
            arg.pixel_box_upper_corner,
            element_strides,
        )
    else:
        # Tiled mode - use existing tiled descriptor fill function
        cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_tiled(
            arg.base.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            expanded_shape,
            strides,
            padding,
        )

    return [cu_tensor_map, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    return wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta, make_tensordesc_arg)


class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        self.gsan_enabled = "gsan" in getattr(metadata, "instrumentation_mode", "")
        if self.gsan_enabled:
            signature["_gsan_globals_ptr"] = "*i8"

        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta, "nvTmaDesc")
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl
        self._tvmffi_host_launch = None
        self._tvmffi_host_stub_mod = None
        self._tvmffi_arg_plan = None
        self._tvmffi_arg_converters = ()
        self._tvmffi_function = None
        self._tvmffi_function_arg = None
        try:
            arg_plan = _make_tvmffi_host_arg_plan(tuple(signature.values()), tensordesc_meta)
        except ValueError:
            arg_plan = None
        if (not self.gsan_enabled and _is_tvmffi_launcher_enabled() and arg_plan is not None
                and _supports_tvmffi_host_stub(arg_plan)):
            tvm_ffi = _import_tvm_ffi()
            self._tvmffi_host_stub_mod = _compile_tvmffi_host_stub(
                arg_plan, metadata.num_warps, self.num_ctas, metadata.shared, self.launch_cooperative_grid,
                self.launch_pdl)
            self._tvmffi_host_launch = tvm_ffi.Function.__from_extern_c__(
                self._tvmffi_host_stub_mod.get_tvmffi_launch_symbol(),
                keep_alive_object=self._tvmffi_host_stub_mod,
            )
            # The stub still uses Python exception APIs for error reporting.
            self._tvmffi_host_launch.release_gil = False
            self._tvmffi_arg_plan = arg_plan
            self._tvmffi_arg_converters = _make_tvmffi_arg_converters(arg_plan)

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        active_driver = triton.runtime.driver.active

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        def allocate_default_profile_scratch(size, align):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                return active_driver.allocate_default_profile_scratch(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        if _allocation.has_profile_allocator():
            profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                               _allocation._profile_allocator)
        else:
            profile_scratch = allocate_default_profile_scratch(self.profile_scratch_size, self.profile_scratch_align)

        kernel_args = args
        if self.gsan_enabled:
            import triton.experimental.gsan._allocator as gsan_allocator
            device = triton.runtime.driver.active.get_current_device()
            gsan_state_ptr = gsan_allocator.get_global_state_pointer() + device * GSAN_PER_DEVICE_STATE_STRIDE
            kernel_args = (*args, gsan_state_ptr)

        if self._tvmffi_host_launch is not None:
            if launch_enter_hook is not None:
                launch_enter_hook(launch_metadata)

            def ptr_arg(obj):
                return None if obj is None else ctypes.c_void_p(obj.data_ptr())

            kernel_args = _prepare_tvmffi_kernel_args(kernel_args, self._tvmffi_arg_converters)
            if function != self._tvmffi_function:
                self._tvmffi_function = function
                self._tvmffi_function_arg = ctypes.c_void_p(function)
            self._tvmffi_host_launch(self._tvmffi_function_arg, ctypes.c_void_p(stream), gridX, gridY, gridZ,
                                     ptr_arg(global_scratch), ptr_arg(profile_scratch), *kernel_args)
            if launch_exit_hook is not None:
                launch_exit_hook(launch_metadata)
            return

        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, global_scratch,
                    profile_scratch, self.arg_annotations, self.kernel_signature, kernel_args)
        if self.gsan_enabled:
            import triton.experimental.gsan._stream_sync as gsan_stream_sync
            gsan_stream_sync.synchronize_launch_stream(device)


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        if sys.modules.get("torch") is not None:
            super().__init__()
        else:
            self.get_device_capability = self._get_device_capability
            self.get_current_stream = self._get_current_stream
            self.get_current_device = self._get_current_device
            self.set_current_device = self._set_current_device

    def _get_device_capability(self, device):
        return self.utils.get_device_capability(device)

    def _get_current_stream(self, device):
        # The CUDA driver API does not expose PyTorch's notion of the current
        # stream. In torch-free launches we fall back to the device's default
        # stream after making that device's primary context current.
        return self.utils.get_default_stream(device)

    def _get_current_device(self):
        return self.utils.get_current_device()

    def _set_current_device(self, device):
        self.utils.set_current_device(device)

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        return _cuda_driver_is_active()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
