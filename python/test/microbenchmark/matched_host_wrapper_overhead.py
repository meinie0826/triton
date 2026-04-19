"""
Benchmark matched Python-side kernel launch overhead for Triton and CuTeDSL.

The goal is to compare the steady-state host cost of invoking a compiled kernel
from Python with a matched high-level contract:

- 5 tensor-like inputs
- 9 runtime scalar arguments
- 5 constexpr arguments

This keeps the wrapper signature aligned across frameworks and separates:

- cache-hit invocation cost
- per-call host object construction cost
"""

import argparse
from collections import OrderedDict
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


RUNTIME_ARGS = (
    0,
    1,
    1024,
    4096,
    8192,
    False,
    True,
    7,
    16,
)
CONSTEXPR_ARGS = (
    32,
    False,
    True,
    0,
    64,
)


@triton.jit
def triton_nop_tensor_args(
    t1,
    t2,
    t3,
    t4,
    t5,
    nc1,
    nc2,
    nc3,
    nc4,
    nc5,
    nc6,
    nc7,
    nc8,
    nc9,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
):
    pass


@triton.jit
def triton_nop_tensordesc_args(
    td1,
    td2,
    td3,
    td4,
    td5,
    nc1,
    nc2,
    nc3,
    nc4,
    nc5,
    nc6,
    nc7,
    nc8,
    nc9,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
):
    pass


def _median_p10_p90(values):
    arr = np.asarray(values, dtype=np.float64)
    return np.median(arr), np.percentile(arr, 10), np.percentile(arr, 90)


def summarize_result(values):
    median, p10, p90 = _median_p10_p90(values)
    return {
        "median_us": float(median),
        "p10_us": float(p10),
        "p90_us": float(p90),
        "samples_us": [float(v) for v in values],
    }


def do_bench_walltime(fn, *, n_warmup=1000, n_repeat=10000, n_samples=25):
    fn()
    torch.cuda.synchronize()

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    mses = []
    for _ in range(n_samples):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(n_repeat):
            fn()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        mses.append((end_time - start_time) * 1e3 / n_repeat)
    return np.asarray(mses)


def setup_cutlass_python(cutlass_root=None):
    if importlib.util.find_spec("cutlass") is not None:
        return "installed_package"
    if cutlass_root is None:
        raise ModuleNotFoundError(
            "CuTeDSL Python package 'cutlass' is not importable. "
            "Install nvidia-cutlass-dsl or pass --cutlass-root to a source checkout."
        )
    cute_python_root = Path(cutlass_root) / "python" / "CuTeDSL"
    if not cute_python_root.exists():
        raise FileNotFoundError(f"CuTeDSL python root not found: {cute_python_root}")
    sys.path.insert(0, str(cute_python_root))
    if importlib.util.find_spec("cutlass") is None:
        raise ModuleNotFoundError(f"Unable to import 'cutlass' from source root: {cute_python_root}")
    return str(cute_python_root)


def make_torch_tensors(numel, *, dtype=torch.float32):
    shape = (1, numel)
    return [torch.zeros(shape, device="cuda", dtype=dtype) for _ in range(5)]


def make_triton_tensordescs(tensors, numel):
    return [TensorDescriptor.from_tensor(t, block_shape=[1, numel]) for t in tensors]


def build_triton_sections(numel):
    grid = (1,)
    tensor_tensors = make_torch_tensors(numel)
    tensordesc_tensors = make_torch_tensors(numel)

    tensor_launcher = triton_nop_tensor_args[grid]
    tensordesc_launcher = triton_nop_tensordesc_args[grid]
    prebuilt_tensordescs = make_triton_tensordescs(tensordesc_tensors, numel)

    sections = OrderedDict([
        (
            "triton_tensor_end_to_end_matched",
            lambda: tensor_launcher(*tensor_tensors, *RUNTIME_ARGS, *CONSTEXPR_ARGS),
        ),
        (
            "triton_tensordesc_end_to_end_prebuilt_matched",
            lambda: tensordesc_launcher(*prebuilt_tensordescs, *RUNTIME_ARGS, *CONSTEXPR_ARGS),
        ),
        (
            "triton_tensordesc_end_to_end_build_each_time_matched",
            lambda: tensordesc_launcher(
                *make_triton_tensordescs(tensordesc_tensors, numel),
                *RUNTIME_ARGS,
                *CONSTEXPR_ARGS,
            ),
        ),
    ])
    return sections


def build_cutedsl_sections(numel):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_ptr

    @cute.kernel
    def nop_tensor_kernel(
        t1: cute.Tensor,
        t2: cute.Tensor,
        t3: cute.Tensor,
        t4: cute.Tensor,
        t5: cute.Tensor,
        nc1: int,
        nc2: int,
        nc3: int,
        nc4: int,
        nc5: int,
        nc6: bool,
        nc7: bool,
        nc8: int,
        nc9: int,
        c1: cutlass.Constexpr[int],
        c2: cutlass.Constexpr[bool],
        c3: cutlass.Constexpr[bool],
        c4: cutlass.Constexpr[int],
        c5: cutlass.Constexpr[int],
    ):
        pass

    @cute.jit
    def tensor_wrapper(
        t1: cute.Tensor,
        t2: cute.Tensor,
        t3: cute.Tensor,
        t4: cute.Tensor,
        t5: cute.Tensor,
        nc1: int,
        nc2: int,
        nc3: int,
        nc4: int,
        nc5: int,
        nc6: bool,
        nc7: bool,
        nc8: int,
        nc9: int,
        c1: cutlass.Constexpr[int],
        c2: cutlass.Constexpr[bool],
        c3: cutlass.Constexpr[bool],
        c4: cutlass.Constexpr[int],
        c5: cutlass.Constexpr[int],
    ):
        nop_tensor_kernel(
            t1,
            t2,
            t3,
            t4,
            t5,
            nc1,
            nc2,
            nc3,
            nc4,
            nc5,
            nc6,
            nc7,
            nc8,
            nc9,
            c1,
            c2,
            c3,
            c4,
            c5,
        ).launch(grid=[1, 1, 1], block=[1, 1, 1])

    @cute.jit
    def ptr_wrapper(
        p1: cute.Pointer,
        p2: cute.Pointer,
        p3: cute.Pointer,
        p4: cute.Pointer,
        p5: cute.Pointer,
        nc1: int,
        nc2: int,
        nc3: int,
        nc4: int,
        nc5: int,
        nc6: bool,
        nc7: bool,
        nc8: int,
        nc9: int,
        c1: cutlass.Constexpr[int],
        c2: cutlass.Constexpr[bool],
        c3: cutlass.Constexpr[bool],
        c4: cutlass.Constexpr[int],
        c5: cutlass.Constexpr[int],
        numel_arg: cutlass.Constexpr[int],
    ):
        layout = cute.make_ordered_layout((1, numel_arg), order=(1, 0))
        nop_tensor_kernel(
            cute.make_tensor(p1, layout),
            cute.make_tensor(p2, layout),
            cute.make_tensor(p3, layout),
            cute.make_tensor(p4, layout),
            cute.make_tensor(p5, layout),
            nc1,
            nc2,
            nc3,
            nc4,
            nc5,
            nc6,
            nc7,
            nc8,
            nc9,
            c1,
            c2,
            c3,
            c4,
            c5,
        ).launch(grid=[1, 1, 1], block=[1, 1, 1])

    torch_tensors = make_torch_tensors(numel)
    fake_tensors = [make_fake_compact_tensor(cutlass.Float32, (1, numel), stride_order=(1, 0)) for _ in range(5)]
    fake_ptrs = [make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16) for _ in range(5)]

    compiled_tensor = cute.compile(
        tensor_wrapper,
        *fake_tensors,
        *RUNTIME_ARGS,
        *CONSTEXPR_ARGS,
        options="--enable-tvm-ffi",
    )
    compiled_ptr = cute.compile(
        ptr_wrapper,
        *fake_ptrs,
        *RUNTIME_ARGS,
        *CONSTEXPR_ARGS,
        numel_arg=numel,
        options="--enable-tvm-ffi",
    )

    def build_cute_tensors():
        return [from_dlpack(t, enable_tvm_ffi=True) for t in torch_tensors]

    prebuilt_cute_tensors = build_cute_tensors()
    prebuilt_ptrs = [
        make_ptr(cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        for t in torch_tensors
    ]

    sections = OrderedDict([
        (
            "cutedsl_tensor_end_to_end_torch_matched",
            lambda: compiled_tensor(*torch_tensors, *RUNTIME_ARGS),
        ),
        (
            "cutedsl_tensor_end_to_end_prebuilt_matched",
            lambda: compiled_tensor(*prebuilt_cute_tensors, *RUNTIME_ARGS),
        ),
        (
            "cutedsl_tensor_end_to_end_build_each_time_matched",
            lambda: compiled_tensor(*build_cute_tensors(), *RUNTIME_ARGS),
        ),
        (
            "cutedsl_raw_ptr_end_to_end_prebuilt_matched",
            lambda: compiled_ptr(*prebuilt_ptrs, *RUNTIME_ARGS),
        ),
        (
            "cutedsl_raw_ptr_end_to_end_build_each_time_matched",
            lambda: compiled_ptr(
                *[
                    make_ptr(cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                    for t in torch_tensors
                ],
                *RUNTIME_ARGS,
            ),
        ),
    ])
    return sections


def collect_results(sections, *, n_repeat, n_samples, n_warmup):
    results = OrderedDict()
    skipped = OrderedDict()
    for name, fn in sections.items():
        print(f"\n== {name} ==")
        try:
            values_ms = do_bench_walltime(
                fn,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_samples=n_samples,
            )
        except Exception as exc:
            skipped[name] = str(exc)
            print(f"Skipping {name}: {exc}")
            continue
        results[name] = values_ms * 1000.0
    return results, skipped


def print_summary(results, skipped):
    print("\n### Matched Host Wrapper")
    summaries = {name: summarize_result(values) for name, values in results.items()}
    for name, summary in summaries.items():
        print(
            f"{name:>44}: median={summary['median_us']:8.3f} us"
            f"  p10={summary['p10_us']:8.3f} us"
            f"  p90={summary['p90_us']:8.3f} us"
        )

    derived = derive_metrics(results)
    for name, value in derived.items():
        print(f"{name:>44}: median={value:8.3f} us")

    for name, reason in skipped.items():
        print(f"{name:>44}: skipped ({reason})")


def derive_metrics(results):
    medians = {name: summarize_result(values)["median_us"] for name, values in results.items()}
    derived = OrderedDict()
    if (
        "triton_tensor_end_to_end_matched" in medians
        and "cutedsl_tensor_end_to_end_torch_matched" in medians
    ):
        derived["gap_tensor_torch_entry_us"] = (
            medians["triton_tensor_end_to_end_matched"]
            - medians["cutedsl_tensor_end_to_end_torch_matched"]
        )
    if (
        "triton_tensor_end_to_end_matched" in medians
        and "cutedsl_tensor_end_to_end_prebuilt_matched" in medians
    ):
        derived["gap_tensor_prebuilt_entry_us"] = (
            medians["triton_tensor_end_to_end_matched"]
            - medians["cutedsl_tensor_end_to_end_prebuilt_matched"]
        )
    if (
        "triton_tensordesc_end_to_end_build_each_time_matched" in medians
        and "triton_tensordesc_end_to_end_prebuilt_matched" in medians
    ):
        derived["triton_tensordesc_build_delta_us"] = (
            medians["triton_tensordesc_end_to_end_build_each_time_matched"]
            - medians["triton_tensordesc_end_to_end_prebuilt_matched"]
        )
    if (
        "cutedsl_tensor_end_to_end_build_each_time_matched" in medians
        and "cutedsl_tensor_end_to_end_prebuilt_matched" in medians
    ):
        derived["cutedsl_tensor_build_delta_us"] = (
            medians["cutedsl_tensor_end_to_end_build_each_time_matched"]
            - medians["cutedsl_tensor_end_to_end_prebuilt_matched"]
        )
    if (
        "cutedsl_raw_ptr_end_to_end_build_each_time_matched" in medians
        and "cutedsl_raw_ptr_end_to_end_prebuilt_matched" in medians
    ):
        derived["cutedsl_raw_ptr_build_delta_us"] = (
            medians["cutedsl_raw_ptr_end_to_end_build_each_time_matched"]
            - medians["cutedsl_raw_ptr_end_to_end_prebuilt_matched"]
        )
    return derived


def dump_results(path, results, skipped, args, cutlass_source):
    payload = {
        "benchmark": "matched_host_wrapper_overhead",
        "cutlass_root": args.cutlass_root,
        "cutlass_source": cutlass_source,
        "numel": args.numel,
        "tensor_shape": [1, args.numel],
        "tensor_count": 5,
        "runtime_args": list(RUNTIME_ARGS),
        "constexpr_args": list(CONSTEXPR_ARGS),
        "n_repeat": args.n_repeat,
        "n_samples": args.n_samples,
        "n_warmup": args.n_warmup,
        "sections": {},
        "skipped_sections": skipped,
        "derived": derive_metrics(results),
    }
    for name, values in results.items():
        payload["sections"][name] = summarize_result(values)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nWrote results to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark matched Triton vs CuTeDSL host launch overhead.")
    parser.add_argument(
        "--cutlass-root",
        type=str,
        default=None,
        help="Optional CUTLASS source root. Only needed if the installed 'cutlass' package is unavailable.",
    )
    parser.add_argument("--numel", type=int, default=1024)
    parser.add_argument("--n-repeat", type=int, default=10000)
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--n-warmup", type=int, default=1000)
    parser.add_argument("--output", type=str, help="Optional JSON file path for dumping raw samples and summaries.")
    return parser.parse_args()


def main():
    args = parse_args()
    cutlass_source = setup_cutlass_python(args.cutlass_root)

    sections = OrderedDict()
    sections.update(build_triton_sections(args.numel))
    sections.update(build_cutedsl_sections(args.numel))

    results, skipped = collect_results(
        sections,
        n_repeat=args.n_repeat,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
    )
    print_summary(results, skipped)
    if args.output:
        dump_results(args.output, results, skipped, args, cutlass_source)


if __name__ == "__main__":
    main()
