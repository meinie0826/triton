import argparse
from collections import OrderedDict
import importlib
import json
import subprocess
from pathlib import Path
import sys
import time

import numpy as np
import torch


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


def do_bench_via_subprocess(section_name, args):
    values_us = []
    base_cmd = [
        sys.executable,
        __file__,
        "--numel",
        str(args.numel),
        "--internal-section",
        section_name,
    ]
    if args.cutlass_root is not None:
        base_cmd.extend(["--cutlass-root", args.cutlass_root])

    for _ in range(args.n_samples):
        try:
            proc = subprocess.run(base_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"subprocess exited with return code {exc.returncode}"
            raise RuntimeError(f"{section_name} failed in isolated subprocess: {detail}") from exc
        payload = json.loads(proc.stdout)
        values_us.append(payload["median_us"])
    return np.asarray(values_us, dtype=np.float64) / 1000.0


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


def build_benchmarks(numel):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_ptr

    @cute.kernel
    def nop_kernel(x: cute.Tensor):
        pass

    @cute.jit
    def tensor_wrapper(x: cute.Tensor):
        nop_kernel(x).launch(grid=[1, 1, 1], block=[1, 1, 1])

    @cute.jit
    def ptr_wrapper(x_ptr: cute.Pointer, numel: cutlass.Constexpr[int]):
        x = cute.make_tensor(x_ptr, cute.make_ordered_layout((numel,), order=(0,)))
        nop_kernel(x).launch(grid=[1, 1, 1], block=[1, 1, 1])

    fake_tensor = make_fake_compact_tensor(cutlass.Float32, (numel,), stride_order=(0,))
    fake_ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)

    compiled_tensor = cute.compile(tensor_wrapper, fake_tensor, options="--enable-tvm-ffi")
    compiled_ptr = cute.compile(ptr_wrapper, fake_ptr, numel=numel, options="--enable-tvm-ffi")

    x = torch.zeros(numel, device="cuda", dtype=torch.float32)
    cute_tensor = from_dlpack(x, enable_tvm_ffi=True)
    cute_ptr = make_ptr(cutlass.Float32, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    sections = OrderedDict([
        ("from_dlpack_only", lambda: from_dlpack(x, enable_tvm_ffi=True)),
        ("tensor_end_to_end_torch", lambda: compiled_tensor(x)),
        ("tensor_end_to_end_prebuilt", lambda: compiled_tensor(cute_tensor)),
        ("make_ptr_only", lambda: make_ptr(cutlass.Float32, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)),
        ("raw_ptr_end_to_end_build_each_time", lambda: compiled_ptr(
            make_ptr(cutlass.Float32, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16))),
        ("raw_ptr_end_to_end_prebuilt", lambda: compiled_ptr(cute_ptr)),
    ])

    try:
        import cutlass.utils.blackwell_helpers as sm100_utils
        from cutlass._mlir import ir
        from cutlass.cute.nvgpu import tcgen05

        mma_inst_shape_mnk = (128, 256, 16)
        mma_tiler_mnk = (128, 256, 64)
        ab_stages = 4
        a_torch = torch.zeros((mma_tiler_mnk[0], mma_tiler_mnk[2]), device="cuda", dtype=torch.float16)
        b_torch = torch.zeros((mma_tiler_mnk[1], mma_tiler_mnk[2]), device="cuda", dtype=torch.float16)
        a_cute = from_dlpack(a_torch, assumed_align=32, enable_tvm_ffi=True)
        b_cute = from_dlpack(b_torch, assumed_align=32, enable_tvm_ffi=True)

        def prepare_tma_tensors(a_tensor, b_tensor):
            with ir.Context(), ir.Location.unknown():
                mma_op = tcgen05.MmaF16BF16Op(
                    cutlass.Float16,
                    cutlass.Float32,
                    mma_inst_shape_mnk,
                    tcgen05.CtaGroup.ONE,
                    tcgen05.OperandSource.SMEM,
                    tcgen05.OperandMajorMode.K,
                    tcgen05.OperandMajorMode.K,
                )
                tiled_mma = cute.make_tiled_mma(mma_op)
                a_smem_layout = sm100_utils.make_smem_layout_a(
                    tiled_mma,
                    mma_tiler_mnk,
                    a_tensor.element_type,
                    ab_stages,
                )
                b_smem_layout = sm100_utils.make_smem_layout_b(
                    tiled_mma,
                    mma_tiler_mnk,
                    b_tensor.element_type,
                    ab_stages,
                )
                a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
                b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])
                tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
                a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
                    tma_load_op,
                    a_tensor,
                    a_smem_layout_one_stage,
                    mma_tiler_mnk,
                    tiled_mma,
                )
                b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
                    tma_load_op,
                    b_tensor,
                    b_smem_layout_one_stage,
                    mma_tiler_mnk,
                    tiled_mma,
                )
                # Keep the benchmark focused on host-side TMA preparation cost
                # without leaking MLIR objects outside the context lifetime.
                return (
                    cute.rank(a_tma_tensor.layout),
                    cute.rank(b_tma_tensor.layout),
                    cute.size(a_tma_tensor),
                    cute.size(b_tma_tensor),
                    cute.rank(tiled_mma),
                    type(a_tma_atom).__name__,
                    type(b_tma_atom).__name__,
                )

        sections["tma_prepare_only_prebuilt"] = lambda: prepare_tma_tensors(a_cute, b_cute)
        sections["tma_prepare_from_dlpack_each_time"] = lambda: prepare_tma_tensors(
            from_dlpack(a_torch, assumed_align=32, enable_tvm_ffi=True),
            from_dlpack(b_torch, assumed_align=32, enable_tvm_ffi=True),
        )
    except Exception as exc:
        print(f"Skipping CuTeDSL TMA prepare benchmarks: {exc}")

    return sections


def print_summary(results, skipped_sections):
    print("\n### CuTeDSL Wrapper")
    baselines = {name: summarize_result(values)["median_us"] for name, values in results.items()}
    total = baselines["tensor_end_to_end_torch"]
    raw_ptr = baselines["raw_ptr_end_to_end_prebuilt"]
    for name, values in results.items():
        summary = summarize_result(values)
        print(
            f"{name:>32}: median={summary['median_us']:8.3f} us"
            f"  p10={summary['p10_us']:8.3f} us"
            f"  p90={summary['p90_us']:8.3f} us"
            f"  share_of_torch_tensor={100.0 * summary['median_us'] / total:6.2f}%"
        )

    print(f"{'dlpack_wrapper_delta':>32}: median={total - baselines['tensor_end_to_end_prebuilt']:8.3f} us")
    print(f"{'raw_ptr_wrapper_delta':>32}: median={baselines['raw_ptr_end_to_end_build_each_time'] - raw_ptr:8.3f} us")
    print(f"{'torch_tensor_vs_raw_ptr':>32}: median={total - raw_ptr:8.3f} us")
    if "tma_prepare_only_prebuilt" in baselines:
        print(f"{'tma_prepare_only_prebuilt':>32}: median={baselines['tma_prepare_only_prebuilt']:8.3f} us")
    if "tma_prepare_from_dlpack_each_time" in baselines:
        print(
            f"{'tma_prepare_from_dlpack_delta':>32}: "
            f"median={baselines['tma_prepare_from_dlpack_each_time'] - baselines['tma_prepare_only_prebuilt']:8.3f} us"
        )
    for name, reason in skipped_sections.items():
        print(f"{name:>32}: skipped ({reason})")


def dump_results(path, results, skipped_sections, args, cutlass_source):
    baselines = {name: summarize_result(values)["median_us"] for name, values in results.items()}
    payload = {
        "framework": "cutedsl",
        "cutlass_root": args.cutlass_root,
        "cutlass_source": cutlass_source,
        "numel": args.numel,
        "n_repeat": args.n_repeat,
        "n_samples": args.n_samples,
        "n_warmup": args.n_warmup,
        "sections": {},
        "skipped_sections": skipped_sections,
        "derived": {
            "dlpack_wrapper_delta_us": baselines["tensor_end_to_end_torch"] - baselines["tensor_end_to_end_prebuilt"],
            "raw_ptr_wrapper_delta_us": baselines["raw_ptr_end_to_end_build_each_time"] - baselines["raw_ptr_end_to_end_prebuilt"],
            "torch_tensor_vs_raw_ptr_us": baselines["tensor_end_to_end_torch"] - baselines["raw_ptr_end_to_end_prebuilt"],
        },
    }
    total = baselines["tensor_end_to_end_torch"]
    for name, values in results.items():
        summary = summarize_result(values)
        summary["share_of_torch_tensor_pct"] = 100.0 * summary["median_us"] / total
        payload["sections"][name] = summary
    if "tma_prepare_only_prebuilt" in baselines:
        payload["derived"]["tma_prepare_only_prebuilt_us"] = baselines["tma_prepare_only_prebuilt"]
    if "tma_prepare_from_dlpack_each_time" in baselines:
        payload["derived"]["tma_prepare_from_dlpack_delta_us"] = (
            baselines["tma_prepare_from_dlpack_each_time"] - baselines["tma_prepare_only_prebuilt"]
        )

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nWrote results to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CuTeDSL Python wrapper overhead.")
    parser.add_argument("--cutlass-root", type=str, default=None,
                        help="Optional CUTLASS source root. Only needed if the installed 'cutlass' package is unavailable.")
    parser.add_argument("--numel", type=int, default=1024)
    parser.add_argument("--n-repeat", type=int, default=10000)
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--n-warmup", type=int, default=1000)
    parser.add_argument("--output", type=str, help="Optional JSON file path for dumping raw samples and summaries.")
    parser.add_argument("--internal-section", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cutlass_source = setup_cutlass_python(args.cutlass_root)
    sections = build_benchmarks(args.numel)

    if args.internal_section:
        start_time = time.perf_counter()
        sections[args.internal_section]()
        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1e6
        print(json.dumps({
            "median_us": elapsed_us,
            "p10_us": elapsed_us,
            "p90_us": elapsed_us,
            "samples_us": [elapsed_us],
        }))
        sys.exit(0)

    results = OrderedDict()
    skipped_sections = OrderedDict()
    for name, fn in sections.items():
        print(f"\n== {name} ==")
        try:
            if name.startswith("tma_prepare_"):
                values_ms = do_bench_via_subprocess(name, args)
            else:
                values_ms = do_bench_walltime(
                    fn,
                    n_warmup=args.n_warmup,
                    n_repeat=args.n_repeat,
                    n_samples=args.n_samples,
                )
        except Exception as exc:
            skipped_sections[name] = str(exc)
            print(f"Skipping {name}: {exc}")
            continue
        results[name] = values_ms * 1000.0

    print_summary(results, skipped_sections)
    if args.output:
        dump_results(args.output, results, skipped_sections, args, cutlass_source)
