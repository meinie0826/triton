from __future__ import annotations

import argparse
import os
import statistics

import torch

from bench_mlp import parse_dtype, quantize_weight, run_mlp as run_mlp_materialized, was_launched_with_torchrun
from triton_kernels.distributed import (
    convert_dp_to_ep,
    convert_ep_to_dp,
    make_expt_assignment,
    make_expt_dict_uniform,
    remote_gather_dp_to_ep,
    SymmetricMemoryPool,
)
from triton_kernels.matmul import FlexCtx, FusedActivation, FnSpecs, PrecisionConfig, matmul
from triton_kernels.matmul_details.opt_flags import scoped_opt_flags_constraints
from triton_kernels.reduce import reduce
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import FP4, make_ragged_tensor_metadata, remap_ragged_tensor_metadata
from triton_kernels.tensor_details import layout
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE
from triton_kernels.target_info import get_cdna_version
from triton_kernels.topk import topk
from triton_kernels.distributed_details.mesh import Mesh


def _time_ms(fn, iters: int):
    warmup = max(2, min(5, iters // 4))
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "p90": times[int(0.9 * (len(times) - 1))],
    }


def _assert_close_with_stats(name: str, ref: torch.Tensor, val: torch.Tensor, rtol=1e-3, atol=1e-2):
    try:
        torch.testing.assert_close(ref, val, rtol=rtol, atol=atol)
    except AssertionError:
        diff = (ref.float() - val.float()).abs()
        max_diff = diff.max().item()
        max_idx = diff.flatten().argmax().item()
        print(
            f"[{name}] mismatch: max_diff={max_diff} "
            f"ref={ref.flatten()[max_idx].item()} val={val.flatten()[max_idx].item()} "
            f"shape={tuple(ref.shape)}",
            flush=True,
        )
        raise


def run_mlp_fused(
    x_dp_local_bf16,
    x_dp_local_fp8,
    wg_global,
    bg_global,
    pcg,
    w1_ep_local,
    b1_ep_local,
    pc1,
    act1,
    w2_ep_local,
    b2_ep_local,
    pc2,
    n_expts_act,
    expt_assignment,
    rank,
    symm_mem_pool,
    fc1_constraints=None,
    fc2_constraints=None,
):
    l_dp_local = matmul(x_dp_local_bf16, wg_global, bg_global, precision_config=pcg)
    l_global_active = topk(l_dp_local, n_expts_act, apply_softmax=True, all_gather=True, symm_mem_pool=symm_mem_pool)

    active_indx = l_global_active.indx
    expt_sizes = l_global_active.mask_metadata.col_sum
    dispatch_indx = l_global_active.mask_metadata.row_sorted_indx
    combine_indx = l_global_active.mask_metadata.col_sorted_indx
    x_global_metadata = make_ragged_tensor_metadata(expt_sizes, dispatch_indx.shape[0])
    y_ep_local_metadata = remap_ragged_tensor_metadata(x_global_metadata, expt_assignment.expt_map[rank, :])
    debug_checks = os.environ.get("TRITON_MOE_GATHER_DEBUG", "0") not in ("", "0", "false", "False")

    if symm_mem_pool.mesh.world_size > 1:
        # MegaMoE-style first cut: avoid source-rank push dispatch by exposing
        # local X through symmetric memory and pulling the needed remote rows on
        # each expert rank. FC1/FC2 stay on the existing ragged matmul path.
        y_ep_local_remote = remote_gather_dp_to_ep(
            x_dp_local_fp8, expt_assignment, active_indx, dispatch_indx, symm_mem_pool
        )
        if debug_checks:
            y_ep_local_remote_clone = y_ep_local_remote.clone()
            y_ep_local_ref = convert_dp_to_ep(
                x_dp_local_fp8, expt_assignment, active_indx, dispatch_indx, symm_mem_pool
            ).clone()
            # Check valid rows (those actually accessed by fc1 matmul)
            active_flat = active_indx.reshape(-1)
            dispatch_flat = dispatch_indx
            n_tot = active_flat.shape[0]
            expt_map = expt_assignment.expt_map[rank, :]
            local_expt_set = set(expt_map[expt_map >= 0].tolist())
            valid_row_mask = torch.tensor([int(a.item()) in local_expt_set for a in active_flat],
                                          device=active_flat.device, dtype=torch.bool)
            valid_row_indices = dispatch_flat[:n_tot][valid_row_mask]
            if valid_row_indices.numel() > 0:
                _assert_close_with_stats(
                    "dp_to_ep_remote[valid_rows]",
                    y_ep_local_ref[valid_row_indices],
                    y_ep_local_remote_clone[valid_row_indices],
                    rtol=0.0, atol=0.0,
                )
            # Zero out invalid rows in both so matmul reads same data regardless
            invalid_row_mask = torch.ones(y_ep_local_remote_clone.shape[0], dtype=torch.bool, device=y_ep_local_remote_clone.device)
            if valid_row_indices.numel() > 0:
                invalid_row_mask[valid_row_indices] = False
            y_ep_local_remote_clone.view(torch.uint8)[invalid_row_mask] = 0
            y_ep_local_ref.view(torch.uint8)[invalid_row_mask] = 0
            y_ep_local_for_fc1 = y_ep_local_remote_clone
        else:
            y_ep_local_for_fc1 = y_ep_local_remote
        with scoped_opt_flags_constraints(fc1_constraints or {}):
            y_fc1_local = matmul(
                y_ep_local_for_fc1,
                w1_ep_local,
                b1_ep_local,
                a_ragged_metadata=y_ep_local_metadata,
                precision_config=pc1,
                fused_activation=act1,
            )
        if debug_checks:
            y_fc1_ref = matmul(
                y_ep_local_ref,
                w1_ep_local,
                b1_ep_local,
                a_ragged_metadata=y_ep_local_metadata,
                precision_config=pc1,
                fused_activation=act1,
            )
            _assert_close_with_stats("fc1", y_fc1_ref.float(), y_fc1_local.float())
        with scoped_opt_flags_constraints(fc2_constraints or {}):
            y_ep_local = matmul(
                y_fc1_local,
                w2_ep_local,
                b2_ep_local,
                a_ragged_metadata=y_ep_local_metadata,
                precision_config=pc2,
            )
        if debug_checks:
            y_fc2_ref = matmul(
                y_fc1_ref,
                w2_ep_local,
                b2_ep_local,
                a_ragged_metadata=y_ep_local_metadata,
                precision_config=pc2,
            )
            _assert_close_with_stats("fc2", y_fc2_ref.float(), y_ep_local.float())
        y_dp_local = convert_ep_to_dp(y_ep_local, expt_assignment, active_indx, combine_indx, symm_mem_pool)
        y_dp_local = y_dp_local.view(-1, n_expts_act, y_dp_local.shape[-1])
        z_dp_local, _ = reduce(y_dp_local, dim=1)
        return z_dp_local

    # SonicMoE-style path:
    # keep X dense, gather it on the fly inside FC1, then scatter FC2 back to
    # token/top-k order instead of materializing a separate dispatch tensor.
    x_gather_idx = combine_indx // n_expts_act
    with scoped_opt_flags_constraints(fc1_constraints or {}):
        y_ep_local = matmul(
            x_dp_local_fp8,
            w1_ep_local,
            b1_ep_local,
            a_ragged_metadata=y_ep_local_metadata,
            gather_indx=x_gather_idx,
            precision_config=pc1,
            fused_activation=act1,
        )

    with scoped_opt_flags_constraints(fc2_constraints or {}):
        y_dp_local = matmul(
            y_ep_local,
            w2_ep_local,
            b2_ep_local,
            a_ragged_metadata=y_ep_local_metadata,
            scatter_indx=combine_indx,
            precision_config=pc2,
        )

    y_dp_local = y_dp_local.view(-1, n_expts_act, y_dp_local.shape[-1])
    z_dp_local, _ = reduce(y_dp_local, dim=1)
    return z_dp_local


def _build_case(
    batch_per_expt,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    ep,
    shuffle_mx4=False,
):
    rank = torch.distributed.get_rank()
    dev = torch.cuda.current_device()
    batch = batch_per_expt * n_expts_tot // n_expts_act

    symm_mem_pool = SymmetricMemoryPool(Mesh(torch.distributed.group.WORLD))
    symm_mem_pool.initialize_matmul(
        n_tokens_global=batch_per_expt * n_expts_tot // n_expts_act,
        d_input=dim1,
        d_model=dim2,
        n_expts_act=n_expts_act,
        n_expts_tot=n_expts_tot,
        dtype=x_dtype,
        device=torch.device(dev),
    )

    wg_global = torch.randn((dim1, n_expts_tot), device=dev)
    torch.distributed.broadcast(wg_global, src=0)
    w1_ep_local = torch.randn((n_expts_tot // ep, dim1, dim2), device=dev)
    w2_ep_local = torch.randn((n_expts_tot // ep, dim2 // 2, dim1), device=dev)
    bg_global = torch.randn((n_expts_tot,), device=dev)
    torch.distributed.broadcast(bg_global, src=0)
    b1_ep_local = torch.randn((n_expts_tot // ep, dim2), device=dev)
    b2_ep_local = torch.randn((n_expts_tot // ep, dim1), device=dev)
    torch.distributed.barrier()

    opt1 = {}
    opt2 = {}
    if w_dtype == FP4:
        num_warps = 4 if batch <= 512 else 8
        value_layout = layout.make_default_matmul_mxfp4_w_layout(
            mx_axis=-2,
            allow_blackwell_value_shuffle=shuffle_mx4,
        )
        scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=-2, num_warps=num_warps)
        opt1 = {"value_layout": value_layout, "scale_layout": scale_layout}
        opt2 = dict(opt1)

    wg_global, wg_flex, wg_scale = quantize_weight(wg_global, torch.bfloat16)
    w1_ep_local, w1_flex, w1_scale = quantize_weight(w1_ep_local, w_dtype, **opt1)
    w2_ep_local, w2_flex, w2_scale = quantize_weight(w2_ep_local, w_dtype, **opt2)

    pcg = PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=wg_flex),
        b_mx_scale=wg_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
    )
    pc1 = PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=w1_flex),
        b_mx_scale=w1_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
    )
    pc2 = PrecisionConfig(
        flex_ctx=FlexCtx(rhs_data=w2_flex),
        b_mx_scale=w2_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
    )

    x_dp_local_fp8 = torch.randn((batch // ep, dim1), device=dev).to(x_dtype)
    x_dp_local_bf16 = x_dp_local_fp8.to(torch.bfloat16)

    act1 = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2), (1.0, 1.0))

    expt_dict = make_expt_dict_uniform(ep, n_expts_tot)
    expt_assignment = make_expt_assignment(ep, n_expts_tot, expt_dict, torch.device(dev))

    return dict(
        rank=rank,
        symm_mem_pool=symm_mem_pool,
        wg_global=wg_global,
        bg_global=bg_global,
        pcg=pcg,
        w1_ep_local=w1_ep_local,
        b1_ep_local=b1_ep_local,
        pc1=pc1,
        act1=act1,
        w2_ep_local=w2_ep_local,
        b2_ep_local=b2_ep_local,
        pc2=pc2,
        n_expts_act=n_expts_act,
        expt_assignment=expt_assignment,
        x_dp_local_fp8=x_dp_local_fp8,
        x_dp_local_bf16=x_dp_local_bf16,
    )


def compare_case(
    batch_per_expt,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    ep,
    shuffle_mx4=False,
    iters=8,
):
    case = _build_case(batch_per_expt, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, ep, shuffle_mx4)

    def run_standard():
        return run_mlp_materialized(
            case["x_dp_local_bf16"],
            case["x_dp_local_fp8"],
            case["wg_global"],
            case["bg_global"],
            case["pcg"],
            case["w1_ep_local"],
            case["b1_ep_local"],
            case["pc1"],
            case["act1"],
            case["w2_ep_local"],
            case["b2_ep_local"],
            case["pc2"],
            case["n_expts_act"],
            case["expt_assignment"],
            case["rank"],
            case["symm_mem_pool"],
        )

    def run_fused():
        return run_mlp_fused(
            case["x_dp_local_bf16"],
            case["x_dp_local_fp8"],
            case["wg_global"],
            case["bg_global"],
            case["pcg"],
            case["w1_ep_local"],
            case["b1_ep_local"],
            case["pc1"],
            case["act1"],
            case["w2_ep_local"],
            case["b2_ep_local"],
            case["pc2"],
            case["n_expts_act"],
            case["expt_assignment"],
            case["rank"],
            case["symm_mem_pool"],
        )

    with torch.no_grad():
        ref = run_standard()
        fused = run_fused()
        _assert_close_with_stats("final", ref.float(), fused.float(), rtol=1e-3, atol=1e-2)

        standard_stats = _time_ms(run_standard, iters)
        fused_stats = _time_ms(run_fused, iters)

    speedup = standard_stats["median"] / fused_stats["median"]
    print(
        f"batch_per_expt={batch_per_expt} "
        f"standard_ms={standard_stats['median']:.3f} "
        f"fused_ms={fused_stats['median']:.3f} "
        f"speedup={speedup:.3f} "
        f"mode={'local_gather_scatter' if ep == 1 else 'remote_pull_dispatch'}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare materialized MoE dispatch vs SonicMoE-style fused gather.")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--dim1", type=int, default=5760)
    parser.add_argument("--dim2", type=int, default=5760)
    parser.add_argument("--n-expts-tot", type=int, default=128)
    parser.add_argument("--n-expts-act", type=int, default=4)
    parser.add_argument("--shuffle-mx4", action="store_true", default=False)
    args = parser.parse_args()

    if not was_launched_with_torchrun():
        print("usage: torchrun --nproc-per-node=1 ./bench_mlp_fused.py")
        return

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, device_id=torch.device(local_rank))
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    ep = torch.distributed.get_world_size()

    print("### fp8x-fp8w", flush=True)
    for batch_per_expt in batch_sizes:
        compare_case(
            batch_per_expt,
            args.dim1,
            args.dim2,
            args.n_expts_tot,
            args.n_expts_act,
            parse_dtype(dense_dtypes[0]),
            parse_dtype(dense_dtypes[1]),
            ep,
            iters=args.iters,
        )

    print("### fp8x-mx4w", flush=True)
    for batch_per_expt in batch_sizes:
        compare_case(
            batch_per_expt,
            args.dim1,
            args.dim2,
            args.n_expts_tot,
            args.n_expts_act,
            parse_dtype(quantized_dtypes[0]),
            parse_dtype(quantized_dtypes[1]),
            ep,
            shuffle_mx4=args.shuffle_mx4,
            iters=args.iters,
        )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
