"""
Original code by @bertmaher; profiling added by @apgoucher

Extended to attribute steady-state Python launch overhead across the Triton
runtime hot path.
"""

import argparse
import cProfile
from collections import OrderedDict
import pstats
import time

import numpy as np
import torch

import triton
import triton.language as tl
from triton.runtime.jit import compute_cache_key
from triton.tools.tensor_descriptor import TensorDescriptor


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


def benchmark_sections(use_tensor_desc: bool, *, n_repeat: int, n_samples: int, n_warmup: int, profile: bool,
                       profile_lines: int):
    grid = (1, )
    launcher = nop_args[grid]
    targs, ncargs, cargs = make_args(use_tensor_desc)
    args = (*targs, *ncargs, *cargs)

    print("Compiling steady-state kernel...")
    launcher(*args)
    torch.cuda.synchronize()

    device = triton.runtime.driver.active.get_current_device()
    stream = triton.runtime.driver.active.get_current_stream(device)
    kernel_cache, kernel_key_cache, _target, _backend, binder = nop_args.device_caches[device]
    bound_args, specialization, options = binder(*args)
    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache[key]

    # Force module/launcher materialization outside the timed region.
    kernel._init_handles()
    _ = kernel.run

    grid_0 = grid[0]
    grid_1 = 1
    grid_2 = 1
    launch_metadata = kernel.launch_metadata((grid_0, grid_1, grid_2), stream, *bound_args.values())

    sections = OrderedDict([
        ("end_to_end_getitem", lambda: launcher(*args)),
        ("end_to_end_run", lambda: nop_args.run(*args, grid=grid, warmup=False)),
        ("device_plus_stream", lambda: (
            triton.runtime.driver.active.get_current_device(),
            triton.runtime.driver.active.get_current_stream(device),
        )),
        ("binder_only", lambda: binder(*args)),
        ("cache_key_only", lambda: compute_cache_key(kernel_key_cache, specialization, options)),
        ("host_setup_no_launch", lambda: _host_setup_no_launch(args, grid, device, stream, kernel_cache, kernel_key_cache,
                                                               binder)),
        ("compiled_kernel_run", lambda: kernel.run(grid_0, grid_1, grid_2, stream, kernel.function,
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


def _host_setup_no_launch(args, grid, device, stream, kernel_cache, kernel_key_cache, binder):
    bound_args, specialization, options = binder(*args)
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
    baselines = {name: _median_p10_p90(values)[0] for name, values in results.items()}
    total = baselines["end_to_end_getitem"]
    run_total = baselines["end_to_end_run"]
    for name, values in results.items():
        median, p10, p90 = _median_p10_p90(values)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Attribute Triton steady-state Python launch overhead.")
    parser.add_argument("--n-repeat", type=int, default=10000, help="Number of timed iterations inside one sample.")
    parser.add_argument("--n-samples", type=int, default=25, help="Number of timing samples to collect.")
    parser.add_argument("--n-warmup", type=int, default=1000, help="Warmup iterations per section.")
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

    for title, use_tensor_desc in cases:
        results = benchmark_sections(
            use_tensor_desc,
            n_repeat=args.n_repeat,
            n_samples=args.n_samples,
            n_warmup=args.n_warmup,
            profile=args.profile,
            profile_lines=args.profile_lines,
        )
        print_summary(title, results)
