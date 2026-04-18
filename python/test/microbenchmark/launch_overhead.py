"""
Original code by @bertmaher; profiling added by @apgoucher

Extended to attribute steady-state Python launch overhead across the Triton
runtime hot path.
"""

import argparse
import cProfile
from collections import OrderedDict
import json
import pstats
import time

import numpy as np
import torch

import triton
import triton.language as tl
from triton.runtime.jit import compute_cache_key
from triton.tools.tensor_descriptor import TensorDescriptor
from third_party.nvidia.backend.driver import make_tensordesc_arg


@triton.jit
def nop_args(
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


def do_bench_walltime(fn, *, n_warmup=1000, n_repeat=10000, n_samples=25, profile=False, profile_lines=20):
    print("Compiling...")
    fn()
    torch.cuda.synchronize()

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    mses = []

    for _ in range(n_samples):
        print("Running %d benchmarking iterations..." % n_repeat)
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(n_repeat):
            fn()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        wall_time_ms = (end_time - start_time) * 1e3 / n_repeat
        mses.append(wall_time_ms)

    mses = np.array(mses)

    if profile:
        print("Running profiler...")
        prof = cProfile.Profile()
        prof.enable()
        for _ in range(n_repeat):
            fn()
        torch.cuda.synchronize()
        prof.disable()
        stats = pstats.Stats(prof)
        stats.sort_stats("time")
        stats.print_stats(profile_lines)
    return mses


def make_args(use_tensor_desc: bool):
    if use_tensor_desc:
        targs = [TensorDescriptor.from_tensor(torch.zeros(1, 16, device="cuda"), block_shape=[1, 16]) for _ in range(5)]
    else:
        targs = [torch.zeros(1, device="cuda") for _ in range(5)]
    ncargs = [0, 1, 1024, 2**31 - 1, 2**64 - 1, False, True, None, (16, 16)]
    cargs = [32, False, True, 0, 64]
    return targs, ncargs, cargs


def make_runtime_kwargs(kernel_fn):
    return {
        "debug": kernel_fn.debug or triton.knobs.runtime.debug,
        "instrumentation_mode": triton.knobs.compilation.instrumentation_mode,
    }


def resolve_cached_kernel(kernel_fn, args, runtime_kwargs):
    device = triton.runtime.driver.active.get_current_device()
    stream = triton.runtime.driver.active.get_current_stream(device)
    kernel_cache, kernel_key_cache, _target, _backend, binder = kernel_fn.device_caches[device]
    bound_args, specialization, options = binder(*args, **runtime_kwargs)
    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache[key]
    return device, stream, kernel_cache, kernel_key_cache, binder, bound_args, specialization, options, kernel


def _is_tensordesc_signature(sig):
    return isinstance(sig, str) and sig.startswith("tensordesc")


def _flatten_runtime_signature(signature_dict):
    return [signature_dict[idx] for idx in sorted(signature_dict)]


def transform_kernel_args_for_launch(signature, kernel_args, tensordesc_meta):
    transformed = []
    desc_idx = 0
    for sig, arg in zip(signature, kernel_args):
        if _is_tensordesc_signature(sig):
            metadata = tensordesc_meta[desc_idx] if tensordesc_meta else None
            desc_idx += 1
            transformed.extend(make_tensordesc_arg(arg, metadata, ()))
        else:
            transformed.append(arg)
    return tuple(transformed)


def convert_pointer_args_to_raw(signature, kernel_args):
    raw_args = []
    for sig, arg in zip(signature, kernel_args):
        if isinstance(sig, str) and sig.startswith("*") and hasattr(arg, "data_ptr"):
            raw_args.append(arg.data_ptr())
        else:
            raw_args.append(arg)
    return tuple(raw_args)


def launch_via_raw_driver(launcher_obj, raw_launch, grid, stream, function, kernel_metadata, launch_metadata, kernel_args,
                          launch_enter_hook, launch_exit_hook, *, skip_scratch=False):
    grid_x, grid_y, grid_z = grid
    active_driver = triton.runtime.driver.active

    def allocate_scratch(size, align, allocator):
        if skip_scratch or size <= 0:
            return None
        grid_size = grid_x * grid_y * grid_z
        alloc_size = grid_size * launcher_obj.num_ctas * size
        alloc_fn = allocator.get()
        return alloc_fn(alloc_size, align, stream)

    def allocate_default_profile_scratch(size, align):
        if skip_scratch or size <= 0:
            return None
        grid_size = grid_x * grid_y * grid_z
        alloc_size = grid_size * launcher_obj.num_ctas * size
        return active_driver.allocate_default_profile_scratch(alloc_size, align, stream)

    global_scratch = allocate_scratch(launcher_obj.global_scratch_size, launcher_obj.global_scratch_align,
                                      triton.runtime._allocation._allocator)
    if triton.runtime._allocation.has_profile_allocator():
        profile_scratch = allocate_scratch(launcher_obj.profile_scratch_size, launcher_obj.profile_scratch_align,
                                           triton.runtime._allocation._profile_allocator)
    else:
        profile_scratch = allocate_default_profile_scratch(launcher_obj.profile_scratch_size,
                                                           launcher_obj.profile_scratch_align)

    launch_kernel_args = kernel_args
    if launcher_obj.gsan_enabled:
        import triton.experimental.gsan._allocator as gsan_allocator
        device = triton.runtime.driver.active.get_current_device()
        gsan_state_ptr = gsan_allocator.get_global_state_pointer() + device * (1 << 30)
        launch_kernel_args = (*kernel_args, gsan_state_ptr)

    raw_launch(
        grid_x,
        grid_y,
        grid_z,
        stream,
        function,
        launcher_obj.launch_cooperative_grid,
        launcher_obj.launch_pdl,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        global_scratch,
        profile_scratch,
        launcher_obj.arg_annotations,
        launcher_obj.kernel_signature,
        launch_kernel_args,
    )


def build_launch_state(kernel, bound_args, grid, stream):
    launcher_obj = kernel.run
    grid_size = len(grid)
    grid_xyz = (
        grid[0],
        grid[1] if grid_size > 1 else 1,
        grid[2] if grid_size > 2 else 1,
    )
    launch_metadata = kernel.launch_metadata(grid_xyz, stream, *bound_args.values())
    signature = _flatten_runtime_signature(kernel.src.signature)
    tensordesc_meta = getattr(kernel.metadata, "tensordesc_meta", None)
    bound_arg_values = tuple(bound_args.values())
    transformed_kernel_args = transform_kernel_args_for_launch(signature, bound_arg_values, tensordesc_meta)
    raw_pointer_kernel_args = convert_pointer_args_to_raw(signature, bound_arg_values)
    return {
        "launcher_obj": launcher_obj,
        "grid_xyz": grid_xyz,
        "launch_metadata": launch_metadata,
        "signature": signature,
        "bound_arg_values": bound_arg_values,
        "transformed_kernel_args": transformed_kernel_args,
        "raw_pointer_kernel_args": raw_pointer_kernel_args,
    }


def benchmark_sections(use_tensor_desc: bool, *, n_repeat: int, n_samples: int, n_warmup: int, profile: bool,
                       profile_lines: int):
    grid = (1, )
    launcher = nop_args[grid]
    targs, ncargs, cargs = make_args(use_tensor_desc)
    args = (*targs, *ncargs, *cargs)
    runtime_kwargs = make_runtime_kwargs(nop_args)

    print("Compiling steady-state kernel...")
    launcher(*args)
    torch.cuda.synchronize()

    device, stream, kernel_cache, kernel_key_cache, binder, bound_args, specialization, options, kernel = (
        resolve_cached_kernel(nop_args, args, runtime_kwargs))

    # Force module/launcher materialization outside the timed region.
    kernel._init_handles()
    launch_state = build_launch_state(kernel, bound_args, grid, stream)
    launcher_obj = launch_state["launcher_obj"]
    grid_xyz = launch_state["grid_xyz"]
    launch_metadata = launch_state["launch_metadata"]
    raw_launch = triton.runtime.driver.active.utils.launch

    sections = OrderedDict([
        ("end_to_end_getitem", lambda: launcher(*args)),
        ("end_to_end_run", lambda: nop_args.run(*args, grid=grid, warmup=False)),
        ("device_plus_stream", lambda: (
            triton.runtime.driver.active.get_current_device(),
            triton.runtime.driver.active.get_current_stream(device),
        )),
        ("binder_only", lambda: binder(*args, **runtime_kwargs)),
        ("cache_key_only", lambda: compute_cache_key(kernel_key_cache, specialization, options)),
        ("host_setup_no_launch", lambda: _host_setup_no_launch(args, grid, device, stream, kernel_cache, kernel_key_cache,
                                                               binder, runtime_kwargs)),
        ("tensordesc_transform_only", lambda: transform_kernel_args_for_launch(
            launch_state["signature"], launch_state["bound_arg_values"], getattr(kernel.metadata, "tensordesc_meta", None))),
        ("cuda_launcher_no_tensordesc_transform", lambda: launch_via_raw_driver(
            launcher_obj,
            raw_launch,
            grid_xyz,
            stream,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            launch_state["transformed_kernel_args"],
            triton.knobs.runtime.launch_enter_hook,
            triton.knobs.runtime.launch_exit_hook,
        )),
        ("raw_driver_launch_pretransformed", lambda: raw_launch(
            grid_xyz[0], grid_xyz[1], grid_xyz[2], stream, kernel.function,
            launcher_obj.launch_cooperative_grid, launcher_obj.launch_pdl, kernel.packed_metadata,
            launch_metadata, None, None, None, None, launcher_obj.arg_annotations,
            launcher_obj.kernel_signature, launch_state["transformed_kernel_args"])),
        ("raw_driver_launch_rawptr", lambda: raw_launch(
            grid_xyz[0], grid_xyz[1], grid_xyz[2], stream, kernel.function,
            launcher_obj.launch_cooperative_grid, launcher_obj.launch_pdl, kernel.packed_metadata,
            launch_metadata, None, None, None, None, launcher_obj.arg_annotations,
            launcher_obj.kernel_signature,
            launch_state["transformed_kernel_args"]
            if any(_is_tensordesc_signature(sig) for sig in launch_state["signature"])
            else launch_state["raw_pointer_kernel_args"])),
        ("compiled_kernel_run", lambda: kernel.run(grid_xyz[0], grid_xyz[1], grid_xyz[2], stream, kernel.function,
                                                   kernel.packed_metadata, launch_metadata,
                                                   triton.knobs.runtime.launch_enter_hook,
                                                   triton.knobs.runtime.launch_exit_hook, *bound_args.values())),
    ])

    results = OrderedDict()
    for idx, (name, fn) in enumerate(sections.items()):
        print(f"\n== {name} ==")
        values_ms = do_bench_walltime(
            fn,
            n_warmup=n_warmup,
            n_repeat=n_repeat,
            n_samples=n_samples,
            profile=profile and idx == 0,
            profile_lines=profile_lines,
        )
        results[name] = values_ms * 1000.0
    return results


def _host_setup_no_launch(args, grid, device, stream, kernel_cache, kernel_key_cache, binder, runtime_kwargs):
    bound_args, specialization, options = binder(*args, **runtime_kwargs)
    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache[key]
    if callable(grid):
        grid = grid(bound_args)
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    launch_metadata = kernel.launch_metadata((grid_0, grid_1, grid_2), stream, *bound_args.values())
    return device, stream, grid_0, grid_1, grid_2, launch_metadata


def print_summary(title, results):
    print(f"\n### {title}")
    baselines = {name: summarize_result(values)["median_us"] for name, values in results.items()}
    total = baselines["end_to_end_getitem"]
    run_total = baselines["end_to_end_run"]
    for name, values in results.items():
        summary = summarize_result(values)
        median = summary["median_us"]
        p10 = summary["p10_us"]
        p90 = summary["p90_us"]
        pct_total = 100.0 * median / total
        pct_run = 100.0 * median / run_total
        print(
            f"{name:>20}: median={median:8.3f} us  p10={p10:8.3f} us  p90={p90:8.3f} us"
            f"  share_of_getitem={pct_total:6.2f}%  share_of_run={pct_run:6.2f}%"
        )

    wrapper_only = total - baselines["compiled_kernel_run"]
    host_only = baselines["host_setup_no_launch"]
    lambda_only = total - run_total
    print(f"{'wrapper_minus_launcher':>20}: median={wrapper_only:8.3f} us")
    print(f"{'host_setup_minus_launch':>20}: median={host_only:8.3f} us")
    print(f"{'getitem_lambda_only':>20}: median={lambda_only:8.3f} us")


def dump_results(path, all_results, args):
    payload = {
        "n_repeat": args.n_repeat,
        "n_samples": args.n_samples,
        "n_warmup": args.n_warmup,
        "cases": [],
    }
    for title, results in all_results:
        case = {
            "title": title,
            "sections": {},
        }
        baselines = {name: summarize_result(values)["median_us"] for name, values in results.items()}
        total = baselines["end_to_end_getitem"]
        run_total = baselines["end_to_end_run"]
        for name, values in results.items():
            summary = summarize_result(values)
            summary["share_of_getitem_pct"] = 100.0 * summary["median_us"] / total
            summary["share_of_run_pct"] = 100.0 * summary["median_us"] / run_total
            case["sections"][name] = summary
        case["derived"] = {
            "wrapper_minus_launcher_us": total - baselines["compiled_kernel_run"],
            "host_setup_minus_launch_us": baselines["host_setup_no_launch"],
            "getitem_lambda_only_us": total - run_total,
        }
        payload["cases"].append(case)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nWrote results to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Attribute Triton steady-state Python launch overhead.")
    parser.add_argument("--n-repeat", type=int, default=10000, help="Number of timed iterations inside one sample.")
    parser.add_argument("--n-samples", type=int, default=25, help="Number of timing samples to collect.")
    parser.add_argument("--n-warmup", type=int, default=1000, help="Warmup iterations per section.")
    parser.add_argument("--output", type=str, help="Optional JSON file path for dumping raw samples and summaries.")
    parser.add_argument("--tensor-desc-only", action="store_true", help="Run only the TensorDescriptor case.")
    parser.add_argument("--tensor-only", action="store_true", help="Run only the plain Tensor case.")
    parser.add_argument("--profile", action="store_true", help="cProfile the first end-to-end section.")
    parser.add_argument("--profile-lines", type=int, default=20, help="How many cProfile lines to print.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tensor_desc_only and args.tensor_only:
        raise ValueError("Choose at most one of --tensor-desc-only and --tensor-only")

    cases = []
    if not args.tensor_desc_only:
        cases.append(("Tensor inputs", False))
    if not args.tensor_only:
        cases.append(("TensorDescriptor inputs", True))

    all_results = []
    for title, use_tensor_desc in cases:
        results = benchmark_sections(
            use_tensor_desc,
            n_repeat=args.n_repeat,
            n_samples=args.n_samples,
            n_warmup=args.n_warmup,
            profile=args.profile,
            profile_lines=args.profile_lines,
        )
        all_results.append((title, results))
        print_summary(title, results)
    if args.output:
        dump_results(args.output, all_results, args)
