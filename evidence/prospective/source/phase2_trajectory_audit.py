#!/usr/bin/env python
"""Trajectory audit for official PyTorch ELF checkpoints.

The goal is diagnostic, not a new sampler:
1. generate samples with the official ELF-B checkpoint;
2. record internal trajectory signals (drift, self-conditioning agreement,
   decoder entropy, layer collapse proxy, effective-rank proxy, CKA);
3. correlate those signals with per-sample GPT-2-Large perplexity.

This script leaves the official source files untouched. Hidden states are
captured with forward hooks on Transformer blocks at selected timesteps.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def add_src_to_path(root: Path) -> None:
    src = root / "official-ELF-pytorch_elf" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def get_pad_token_id(tokenizer, pad_token="pad") -> int:
    if pad_token == "eos" and tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    eos_mask = predicted_ids == eos_token_id
    keep_mask = eos_mask.to(torch.int32).cumsum(dim=1) == 0
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def cosine_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.float().flatten(1)
    b = b.float().flatten(1)
    return F.cosine_similarity(a, b, dim=-1, eps=eps)


def rel_norm(a: torch.Tensor, b: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    a = a.float().flatten(1)
    if b is not None:
        b = b.float().flatten(1)
        a = a - b
    denom = a.new_tensor(eps)
    if b is not None:
        denom = b.norm(dim=-1).clamp_min(eps)
    return a.norm(dim=-1) / denom


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.float().softmax(dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(-1).mean(-1)


def unigram_entropy(ids: torch.Tensor, pad_token_id: int) -> List[float]:
    out = []
    for row in ids.detach().cpu().numpy():
        row = row[row != pad_token_id]
        if len(row) == 0:
            out.append(float("nan"))
            continue
        vals, counts = np.unique(row, return_counts=True)
        p = counts.astype(np.float64) / counts.sum()
        out.append(float(-(p * np.log(p + 1e-12)).sum()))
    return out


def linear_cka(x: torch.Tensor, y: torch.Tensor, max_rows: int = 2048, seed: int = 0) -> float:
    x = x.detach().float().reshape(-1, x.shape[-1])
    y = y.detach().float().reshape(-1, y.shape[-1])
    n = min(x.shape[0], y.shape[0])
    x, y = x[:n], y[:n]
    if n > max_rows:
        g = torch.Generator(device=x.device).manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=x.device)[:max_rows]
        x, y = x[idx], y[idx]
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xty = torch.linalg.matrix_norm(x.T @ y, ord="fro") ** 2
    xtx = torch.linalg.matrix_norm(x.T @ x, ord="fro")
    yty = torch.linalg.matrix_norm(y.T @ y, ord="fro")
    return float((xty / (xtx * yty).clamp_min(1e-12)).detach().cpu())


def effective_rank_proxy(hidden: torch.Tensor) -> torch.Tensor:
    """Cheap per-sample participation-ratio rank proxy from token variance.

    Full eigenspectra are too expensive for every layer/timestep/sample. This
    proxy is enough for correlation screening; exact CKA/effective-rank heatmaps
    can be computed offline for selected states after we know where to look.
    """
    x = hidden.float()
    x = x - x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False).clamp_min(0.0)
    return (var.sum(-1) ** 2) / (var.pow(2).sum(-1).clamp_min(1e-12))


def layer_collapse_proxy(layer_hiddens: Dict[int, torch.Tensor]) -> torch.Tensor:
    means = []
    for idx in sorted(layer_hiddens):
        h = layer_hiddens[idx].float().mean(1)
        means.append(F.normalize(h, dim=-1))
    if len(means) < 2:
        return torch.full((next(iter(layer_hiddens.values())).shape[0],), float("nan"))
    sims = [(means[i] * means[i + 1]).sum(-1) for i in range(len(means) - 1)]
    return torch.stack(sims, dim=0).mean(0)


@contextmanager
def capture_block_outputs(model, trim_prefix: int):
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            captured[layer_idx] = out[:, trim_prefix:].detach()
        return hook

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_hook(i)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def forward_x(
    model,
    z: torch.Tensor,
    t_value: float,
    x_prev: torch.Tensor,
    sc_scale: float,
    config,
    capture: bool = False,
):
    batch = z.shape[0]
    t_batch = torch.full((batch,), float(t_value), dtype=z.dtype, device=z.device)
    sc_batch = (
        torch.full((batch,), float(sc_scale), dtype=z.dtype, device=z.device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = torch.cat([z, x_prev], dim=-1) if config.self_cond_prob > 0 else z
    trim_prefix = config.num_time_tokens + config.num_self_cond_cfg_tokens + config.num_model_mode_tokens
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    if capture:
        with capture_block_outputs(model, trim_prefix=trim_prefix) as hiddens:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                x_pred, _ = model(
                    z_input, t_batch, deterministic=True,
                    self_cond_cfg_scale=sc_batch,
                    decoder_step_active=None,
                )
            return x_pred, {k: v for k, v in hiddens.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        x_pred, _ = model(
            z_input, t_batch, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=None,
        )
    return x_pred, {}


def decode_logits(model, z: torch.Tensor, t_value: float, sc_scale: float, config) -> torch.Tensor:
    batch = z.shape[0]
    t_batch = torch.full((batch,), float(t_value), dtype=z.dtype, device=z.device)
    sc_batch = (
        torch.full((batch,), float(sc_scale), dtype=z.dtype, device=z.device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, logits = model(
            z_input, t_batch, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
        )
    return logits


def get_sampling_steps(n_steps, schedule, p_mean, p_std, device, dtype, generator):
    if schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    if schedule == "logit_normal":
        # Keep this local so each job is deterministic from its own generator.
        z = torch.randn((n_steps - 1,), generator=generator, dtype=torch.float32) * p_std + p_mean
        steps = torch.sort(torch.sigmoid(z).to(device=device, dtype=dtype)).values
        return torch.cat([
            torch.zeros((1,), dtype=dtype, device=device),
            steps,
            torch.ones((1,), dtype=dtype, device=device),
        ])
    raise ValueError(schedule)


def load_elf(args, device):
    add_src_to_path(args.root)
    from configs.config import apply_config_overrides, load_config_from_yaml, load_sampling_configs
    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder

    config = load_config_from_yaml(str(args.config))
    overrides = [
        f"encoder_model_name={args.encoder_model}",
        f"tokenizer_name={args.tokenizer}",
        f"output_dir={args.out}",
        "use_wandb=false",
        "online_eval=false",
        f"num_samples={args.num_samples}",
        f"global_batch_size={args.batch_size}",
        f"eval_ppl_model={args.ppl_model}",
        "use_compile=false",
    ]
    config = apply_config_overrides(config, overrides)
    if config.sampling_configs_path:
        # Resolve official relative path if needed.
        sp = Path(config.sampling_configs_path)
        if not sp.is_absolute():
            official_root = args.root / "official-ELF-pytorch_elf"
            sp = official_root / sp
        config.sampling_configs = load_sampling_configs(str(sp))

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    enc_cfg, _encoder = get_encoder(config.encoder_model_name, torch.float32)

    model = ELF_models[config.model](
        text_encoder_dim=enc_cfg.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=tokenizer.vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)
    checkpoint_path = args.checkpoint
    if checkpoint_path.is_dir():
        candidates = sorted(
            checkpoint_path.glob("checkpoint_*"),
            key=lambda p: int(p.name.rsplit("_", 1)[-1]),
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoint_* under {checkpoint_path}")
        checkpoint_path = candidates[-1]
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    parameter_key = args.weights
    if parameter_key not in checkpoint:
        raise KeyError(f"{parameter_key!r} missing from {checkpoint_path}")
    model.load_state_dict(checkpoint[parameter_key])
    del checkpoint
    eval_model = model.to(device).eval()
    for p in eval_model.parameters():
        p.requires_grad_(False)
    torch.cuda.empty_cache()
    return config, tokenizer, eval_model


def run_generation_with_audit(args, config, tokenizer, model, device):
    dtype = next(model.parameters()).dtype
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    record_steps = set()
    if args.record_steps:
        record_steps.update(int(x) for x in args.record_steps.split(",") if x.strip())
    if args.record_stride > 0:
        record_steps.update(range(0, args.steps, args.record_stride))
    record_steps.update([0, args.steps // 2, args.steps - 1])

    samples = []
    per_sample_rows = []
    step_rows = []
    layer_rows = []
    cka_rows = []
    final_token_id_rows = []

    n_batches = math.ceil(args.num_samples / args.batch_size)
    local_id = 0
    start_all = time.time()
    for batch_idx in range(n_batches):
        current_batch = min(args.batch_size, args.num_samples - local_id)
        if current_batch <= 0:
            break
        if args.paired_rng:
            # Keep the schedule, initial latent, and SDE innovations independent
            # of how many random numbers another sampler consumes.  Reusing the
            # same seed scheme across jobs makes ODE / official-SDE /
            # coefficient-preserving-SDE comparisons exactly paired by sample.
            schedule_generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 1_000_003 * batch_idx + 11
            )
            initial_generator = torch.Generator(device=device).manual_seed(
                args.seed + 1_000_003 * batch_idx + 23
            )
            sde_generator = torch.Generator(device=device).manual_seed(
                args.seed + 1_000_003 * batch_idx + 37
            )
        else:
            schedule_generator = generator
            initial_generator = None
            sde_generator = None

        t_steps = get_sampling_steps(
            n_steps=args.steps,
            schedule=args.time_schedule,
            p_mean=config.denoiser_p_mean,
            p_std=config.denoiser_p_std,
            device=device,
            dtype=dtype,
            generator=schedule_generator,
        )
        z = (
            torch.randn(
                (current_batch, config.max_length, model.text_encoder_dim),
                dtype=dtype, device=device, generator=initial_generator,
            )
            * config.denoiser_noise_scale
        )
        x_prev = torch.zeros_like(z)
        prev_step_x = None
        prev_gate_delta = None
        prev_gate_agreement = None
        sample_ids = list(range(local_id, local_id + current_batch))
        batch_step_row_start = len(step_rows)
        captured_top1_ids: Dict[int, np.ndarray] = {}

        with torch.no_grad():
            for step_idx in range(args.steps):
                t = float(t_steps[step_idx].item())
                t_next = float(t_steps[step_idx + 1].item())
                alpha = float("nan")
                noise_coeff = float("nan")
                marginal_std_ratio = float("nan")
                if args.sampling_method == "sde" and step_idx < args.steps - 1:
                    h = t_next - t
                    alpha = max(0.0, min(1.0, 1.0 - args.sde_gamma * h))
                    t_model = alpha * t
                    eps = torch.randn(
                        z.shape, dtype=z.dtype, device=z.device,
                        generator=sde_generator,
                    ) * config.denoiser_noise_scale
                    if args.sde_noise_mode == "official":
                        noise_coeff = 1.0 - alpha
                    elif args.sde_noise_mode == "coeff_preserving":
                        target = 1.0 - t_model
                        inherited = alpha * (1.0 - t)
                        noise_coeff = math.sqrt(max(target * target - inherited * inherited, 0.0))
                    else:
                        raise ValueError(args.sde_noise_mode)
                    target_std = max(1.0 - t_model, 1e-12)
                    inherited_std = alpha * (1.0 - t)
                    achieved_std = math.sqrt(inherited_std * inherited_std + noise_coeff * noise_coeff)
                    marginal_std_ratio = achieved_std / target_std
                    z_model = alpha * z + noise_coeff * eps
                else:
                    t_model = t
                    z_model = z

                capture = step_idx in record_steps
                base_sc = float(args.self_cond_cfg_scale)
                current_sc = base_sc
                gate_delta = float("nan")
                gate_agreement = float("nan")

                if args.dynamic_rule == "frontloaded":
                    if t_model < args.frontload_t1:
                        current_sc = args.sc_high
                    elif t_model < args.frontload_t2:
                        current_sc = base_sc
                    else:
                        current_sc = args.sc_low

                signal_rules = {
                    "delta_gate", "agreement_gate",
                    "delta_gate_delayed", "agreement_gate_delayed",
                    "delta_gate_clipped", "agreement_gate_clipped",
                    "delta_gate_random", "agreement_gate_random",
                }
                if args.dynamic_rule in signal_rules:
                    x_zero_base, _ = forward_x(
                        model, z_model, t_model, torch.zeros_like(x_prev),
                        base_sc, config, capture=False,
                    )
                    x_base, _ = forward_x(
                        model, z_model, t_model, x_prev,
                        base_sc, config, capture=False,
                    )
                    gate_delta = float(rel_norm(x_base, x_zero_base).mean().detach().cpu())
                    gate_agreement = float(cosine_mean(x_base, x_zero_base).mean().detach().cpu())
                    delta_signal = gate_delta
                    agreement_signal = gate_agreement
                    if args.dynamic_rule == "delta_gate_delayed":
                        delta_signal = prev_gate_delta if prev_gate_delta is not None else (args.delta_low + args.delta_high) / 2.0
                    elif args.dynamic_rule == "agreement_gate_delayed":
                        agreement_signal = prev_gate_agreement if prev_gate_agreement is not None else (args.agreement_low + args.agreement_high) / 2.0
                    elif args.dynamic_rule == "delta_gate_clipped":
                        delta_signal = min(max(gate_delta, args.delta_low), args.delta_high)
                    elif args.dynamic_rule == "agreement_gate_clipped":
                        agreement_signal = min(max(gate_agreement, args.agreement_low), args.agreement_high)
                    elif args.dynamic_rule == "delta_gate_random":
                        r = random.random()
                        if r < 1 / 3:
                            delta_signal = args.delta_low - 1e-3
                        elif r < 2 / 3:
                            delta_signal = args.delta_high + 1e-3
                        else:
                            delta_signal = (args.delta_low + args.delta_high) / 2.0
                    elif args.dynamic_rule == "agreement_gate_random":
                        r = random.random()
                        if r < 1 / 3:
                            agreement_signal = args.agreement_high + 1e-3
                        elif r < 2 / 3:
                            agreement_signal = args.agreement_low - 1e-3
                        else:
                            agreement_signal = (args.agreement_low + args.agreement_high) / 2.0
                    if args.dynamic_rule.startswith("delta_gate"):
                        if delta_signal < args.delta_low:
                            current_sc = args.sc_high
                        elif delta_signal > args.delta_high:
                            current_sc = args.sc_low
                    elif args.dynamic_rule.startswith("agreement_gate"):
                        if agreement_signal > args.agreement_high:
                            current_sc = args.sc_high
                        elif agreement_signal < args.agreement_low:
                            current_sc = args.sc_low
                    prev_gate_delta = gate_delta
                    prev_gate_agreement = gate_agreement

                x_zero, _ = forward_x(
                    model, z_model, t_model, torch.zeros_like(x_prev),
                    current_sc, config, capture=False,
                )
                x_pred, hiddens = forward_x(
                    model, z_model, t_model, x_prev,
                    current_sc, config, capture=capture,
                )
                denom = max(1.0 - t_model, float(config.t_eps))
                v_pred = (x_pred - z_model) / denom
                z = z_model + (t_next - t_model) * v_pred

                drift = (
                    torch.full((current_batch,), float("nan"), device=device)
                    if prev_step_x is None else rel_norm(x_pred, prev_step_x)
                )
                drift_cos = (
                    torch.full((current_batch,), float("nan"), device=device)
                    if prev_step_x is None else cosine_mean(x_pred, prev_step_x)
                )
                agreement_zero = cosine_mean(x_pred, x_zero)
                sc_delta = rel_norm(x_pred, x_zero)
                x_norm = rel_norm(x_pred)

                decoder_ent = torch.full((current_batch,), float("nan"), device=device)
                decoder_margin = torch.full((current_batch,), float("nan"), device=device)
                decoder_margin_p10 = torch.full((current_batch,), float("nan"), device=device)
                if capture:
                    logits = decode_logits(model, z_model, t_model, args.self_cond_cfg_scale, config)
                    decoder_ent = entropy_from_logits(logits)
                    top2 = logits.float().topk(k=2, dim=-1).values
                    margin_tokens = top2[..., 0] - top2[..., 1]
                    decoder_margin = margin_tokens.mean(-1)
                    decoder_margin_p10 = torch.quantile(margin_tokens, 0.10, dim=-1)
                    captured_top1_ids[step_idx] = logits.argmax(dim=-1).detach().cpu().to(torch.int32).numpy()

                collapse = torch.full((current_batch,), float("nan"), device=device)
                rank_mid = torch.full((current_batch,), float("nan"), device=device)
                rank_last = torch.full((current_batch,), float("nan"), device=device)
                if hiddens:
                    collapse = layer_collapse_proxy(hiddens).to(device)
                    mid_layer = model.depth // 2 - 1
                    last_layer = model.depth - 1
                    if mid_layer in hiddens:
                        rank_mid = effective_rank_proxy(hiddens[mid_layer]).to(device)
                    if last_layer in hiddens:
                        rank_last = effective_rank_proxy(hiddens[last_layer]).to(device)
                    # Aggregate layer diagnostics and CKA at selected steps.
                    for li, h_li in hiddens.items():
                        er = effective_rank_proxy(h_li).mean().item()
                        layer_rows.append({
                            "batch": batch_idx,
                            "step": step_idx,
                            "t": t_model,
                            "layer": li,
                            "effective_rank_proxy_mean": er,
                            "hidden_norm": float(h_li.float().norm(dim=-1).mean().cpu()),
                        })
                    if args.compute_cka:
                        layers = sorted(hiddens)
                        for i, li in enumerate(layers):
                            for lj in layers[i:]:
                                if li == lj:
                                    val = 1.0
                                else:
                                    val = linear_cka(hiddens[li], hiddens[lj], max_rows=args.cka_max_rows, seed=args.seed + step_idx)
                                cka_rows.append({
                                    "batch": batch_idx,
                                    "step": step_idx,
                                    "t": t_model,
                                    "layer_i": li,
                                    "layer_j": lj,
                                    "cka": val,
                                })

                vals = {
                    "drift_rel": drift.detach().cpu().numpy(),
                    "drift_cos": drift_cos.detach().cpu().numpy(),
                    "agreement_zero_cos": agreement_zero.detach().cpu().numpy(),
                    "selfcond_delta_rel": sc_delta.detach().cpu().numpy(),
                    "x_pred_norm": x_norm.detach().cpu().numpy(),
                    "decoder_entropy": decoder_ent.detach().cpu().numpy(),
                    "decoder_margin_mean": decoder_margin.detach().cpu().numpy(),
                    "decoder_margin_p10": decoder_margin_p10.detach().cpu().numpy(),
                    "layer_collapse_proxy": collapse.detach().cpu().numpy(),
                    "rank_mid_proxy": rank_mid.detach().cpu().numpy(),
                    "rank_last_proxy": rank_last.detach().cpu().numpy(),
                }
                for bi, sid in enumerate(sample_ids):
                    row = {
                        "sample_id": sid,
                        "batch": batch_idx,
                        "step": step_idx,
                        "t": t_model,
                        "t_next": t_next,
                        "sde_alpha": alpha,
                        "sde_noise_coeff": noise_coeff,
                        "marginal_std_ratio": marginal_std_ratio,
                        "sc_used": current_sc,
                        "gate_delta_batch": gate_delta,
                        "gate_agreement_batch": gate_agreement,
                    }
                    row.update({k: float(v[bi]) for k, v in vals.items()})
                    step_rows.append(row)

                x_prev = x_pred.detach()
                prev_step_x = x_pred.detach()

            final_logits = decode_logits(model, z, 1.0, args.self_cond_cfg_scale, config)
            final_raw_ids = final_logits.argmax(dim=-1)
            predicted_ids = final_raw_ids
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            final_entropy = entropy_from_logits(final_logits).detach().cpu().numpy()
            final_top2 = final_logits.float().topk(k=2, dim=-1).values
            final_margin_tokens = final_top2[..., 0] - final_top2[..., 1]
            final_margin = final_margin_tokens.mean(-1).detach().cpu().numpy()
            final_margin_p10 = torch.quantile(final_margin_tokens, 0.10, dim=-1).detach().cpu().numpy()
            sample_unigram_entropy = unigram_entropy(predicted_ids, pad_token_id)
            final_token_id_rows.append(predicted_ids.detach().cpu().to(torch.int32).numpy())

            # A token has entered the empirical decoder basin at the earliest
            # captured phase after which its top-1 choice never changes again.
            # The final decode is appended explicitly, so every token has an
            # entry point even when sparse recording is requested.
            captured_steps = sorted(captured_top1_ids)
            final_raw_np = final_raw_ids.detach().cpu().to(torch.int32).numpy()
            observed_steps = captured_steps + [args.steps]
            observed_ids = [captured_top1_ids[s] for s in captured_steps] + [final_raw_np]
            id_history = np.stack(observed_ids, axis=0)
            agrees = id_history == final_raw_np[None, ...]
            persistent = np.logical_and.accumulate(agrees[::-1], axis=0)[::-1]
            entry_obs_idx = np.argmax(persistent, axis=0)
            step_values = np.asarray(observed_steps, dtype=np.float32)
            entry_phase = step_values[entry_obs_idx] / max(float(args.steps), 1.0)

            captured_pos = {step: pos for pos, step in enumerate(captured_steps)}
            for row in step_rows[batch_step_row_start:]:
                pos = captured_pos.get(int(row["step"]))
                if pos is None:
                    row["decoder_final_token_agreement"] = float("nan")
                    row["decoder_persistent_token_fraction"] = float("nan")
                    continue
                bi = int(row["sample_id"]) - local_id
                row["decoder_final_token_agreement"] = float(agrees[pos, bi].mean())
                row["decoder_persistent_token_fraction"] = float(persistent[pos, bi].mean())

        for bi in range(current_batch):
            sid = sample_ids[bi]
            text = tokenizer.decode(predicted_ids[bi].detach().cpu().numpy(), skip_special_tokens=True)
            samples.append({"id": sid, "generated": text})
            per_sample_rows.append({
                "sample_id": sid,
                "generated": text,
                "final_decoder_entropy": float(final_entropy[bi]),
                "final_decoder_margin_mean": float(final_margin[bi]),
                "final_decoder_margin_p10": float(final_margin_p10[bi]),
                "sample_unigram_entropy": float(sample_unigram_entropy[bi]),
                "basin_entry_phase_mean": float(entry_phase[bi].mean()),
                "basin_entry_phase_p50": float(np.quantile(entry_phase[bi], 0.50)),
                "basin_entry_phase_p90": float(np.quantile(entry_phase[bi], 0.90)),
            })
        local_id += current_batch
        print(f"batch {batch_idx + 1}/{n_batches} done; samples={local_id}", flush=True)

    print(f"generation+audit time: {time.time() - start_all:.2f}s", flush=True)
    final_token_ids = np.concatenate(final_token_id_rows, axis=0) if final_token_id_rows else np.empty((0, 0), dtype=np.int32)
    return samples, per_sample_rows, step_rows, layer_rows, cka_rows, final_token_ids


def evaluate_ppl(args, texts: List[str]) -> Tuple[List[float], float, float]:
    """Compute per-sample GPT-style PPL.

    The local GPT-2-Large mirror may contain a corrupt safetensors file while the
    PyTorch bin is intact, so explicitly set use_safetensors=False.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    valid_indices = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    nonempty = [texts[i] for i in valid_indices]
    if not nonempty:
        return [float("nan")] * len(texts), float("nan"), float("nan")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(args.ppl_model), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_safe = (Path(args.ppl_model) / "model.safetensors").is_file()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.ppl_model),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        use_safetensors=use_safe,
    ).to(device).eval()
    valid_ppl = []
    per_entropy = []
    total_nll = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        for start in range(0, len(nonempty), args.ppl_batch_size):
            batch_texts = nonempty[start:start + args.ppl_batch_size]
            enc = tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.ppl_max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            logits = out.logits[:, :-1].float()
            targets = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nll_sum = (nll * mask).sum(-1)
            tok_count = mask.sum(-1)
            ppl = torch.exp(nll_sum / tok_count.clamp_min(1.0))
            valid_ppl.extend(ppl.detach().cpu().float().tolist())
            total_nll += float(nll_sum.sum().detach().cpu())
            total_tokens += float(tok_count.sum().detach().cpu())
            for ids, attn in zip(enc["input_ids"].detach().cpu(), enc["attention_mask"].detach().cpu()):
                vals = ids[attn.bool()].numpy()
                if len(vals) == 0:
                    per_entropy.append(float("nan"))
                else:
                    _, counts = np.unique(vals, return_counts=True)
                    p = counts.astype(np.float64) / counts.sum()
                    per_entropy.append(float(-(p * np.log(p + 1e-12)).sum()))
    mean_ppl = float(math.exp(total_nll / max(total_tokens, 1.0)))
    mean_entropy = float(np.nanmean(per_entropy))
    del model
    torch.cuda.empty_cache()
    per_ppl = [float("nan")] * len(texts)
    for idx, value in zip(valid_indices, valid_ppl):
        per_ppl[idx] = value
    return per_ppl, mean_ppl, mean_entropy


def add_correlations(out: Path, per_sample_rows: List[dict], step_rows: List[dict]) -> List[dict]:
    ps = pd.DataFrame(per_sample_rows)
    st = pd.DataFrame(step_rows)
    features = []
    for name, group in st.groupby("sample_id"):
        row = {"sample_id": name}
        for col in [
            "drift_rel", "drift_cos", "agreement_zero_cos", "selfcond_delta_rel",
            "x_pred_norm", "decoder_entropy", "decoder_margin_mean", "decoder_margin_p10",
            "decoder_final_token_agreement", "decoder_persistent_token_fraction", "layer_collapse_proxy",
            "rank_mid_proxy", "rank_last_proxy",
        ]:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_max"] = float(vals.max())
            row[f"{col}_min"] = float(vals.min())
            if len(vals) > 0:
                early = vals[group["t"] < 0.35]
                mid = vals[(group["t"] >= 0.35) & (group["t"] < 0.75)]
                late = vals[group["t"] >= 0.75]
                row[f"{col}_early_mean"] = float(early.mean())
                row[f"{col}_mid_mean"] = float(mid.mean())
                row[f"{col}_late_mean"] = float(late.mean())
        features.append(row)
    feat = pd.DataFrame(features)
    merged = ps.merge(feat, on="sample_id", how="left")
    merged.to_csv(out / "per_sample_with_features.csv", index=False)

    corr_rows = []
    target_cols = ["per_sample_log_ppl", "per_sample_ppl", "sample_unigram_entropy"]
    feature_cols = [
        c for c in merged.columns
        if c not in {"sample_id", "generated"} and c not in target_cols and pd.api.types.is_numeric_dtype(merged[c])
    ]
    for target in target_cols:
        if target not in merged:
            continue
        for feat_col in feature_cols:
            sub = merged[[target, feat_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 8 or sub[feat_col].nunique() <= 1:
                continue
            rho = sub[target].corr(sub[feat_col], method="spearman")
            corr_rows.append({
                "target": target,
                "feature": feat_col,
                "spearman_rho": float(rho),
                "n": int(len(sub)),
            })
    corr_rows = sorted(corr_rows, key=lambda r: abs(r["spearman_rho"]), reverse=True)
    write_csv(out / "correlations.csv", corr_rows)
    return corr_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/duzc/ELF-main"))
    ap.add_argument("--src", type=Path, default=Path("/data/duzc/ELF-main/official-ELF-pytorch_elf/src"))
    ap.add_argument("--config", type=Path, default=Path("/data/duzc/ELF-main/models/ELF-B-owt-torch/config.yml"))
    ap.add_argument("--checkpoint", type=Path, default=Path("/data/duzc/ELF-main/models/ELF-B-owt-torch"))
    ap.add_argument("--weights", choices=["params", "ema_params1"], default="ema_params1")
    ap.add_argument("--tokenizer", type=Path, default=Path("/data/duzc/ELF-main/models/t5-small"))
    ap.add_argument("--encoder-model", type=Path, default=Path("/data/duzc/ELF-main/models/t5-small"))
    ap.add_argument("--ppl-model", type=Path, default=Path("/data/duzc/ELF-main/models/gpt2-large"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--sampling-method", choices=["sde", "ode"], default="sde")
    ap.add_argument("--time-schedule", choices=["logit_normal", "uniform"], default="logit_normal")
    ap.add_argument("--sde-gamma", type=float, default=1.5)
    ap.add_argument("--sde-noise-mode", choices=["official", "coeff_preserving"], default="official")
    ap.add_argument("--self-cond-cfg-scale", type=float, default=3.0)
    ap.add_argument(
        "--dynamic-rule",
        choices=[
            "fixed", "frontloaded", "delta_gate", "agreement_gate",
            "delta_gate_delayed", "agreement_gate_delayed",
            "delta_gate_clipped", "agreement_gate_clipped",
            "delta_gate_random", "agreement_gate_random",
        ],
        default="fixed",
    )
    ap.add_argument("--sc-low", type=float, default=2.0)
    ap.add_argument("--sc-high", type=float, default=5.0)
    ap.add_argument("--frontload-t1", type=float, default=0.35)
    ap.add_argument("--frontload-t2", type=float, default=0.75)
    ap.add_argument("--delta-low", type=float, default=0.35)
    ap.add_argument("--delta-high", type=float, default=0.70)
    ap.add_argument("--agreement-low", type=float, default=0.85)
    ap.add_argument("--agreement-high", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--paired-rng", action="store_true",
        help="Use batch-local RNG streams so separate sampler jobs are exactly paired.",
    )
    ap.add_argument("--record-stride", type=int, default=8)
    ap.add_argument("--record-steps", default="")
    ap.add_argument("--compute-cka", action="store_true")
    ap.add_argument("--cka-max-rows", type=int, default=1024)
    ap.add_argument("--ppl-batch-size", type=int, default=4)
    ap.add_argument("--ppl-max-length", type=int, default=1024)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    add_src_to_path(args.root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config, tokenizer, model = load_elf(args, device)
    config.output_dir = str(args.out)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples, per_sample, step_rows, layer_rows, cka_rows, final_token_ids = run_generation_with_audit(
        args, config, tokenizer, model, device
    )
    sampling_peak_gib = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )

    with (args.out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.out / "step_metrics.csv", step_rows)
    write_csv(args.out / "layer_diagnostics.csv", layer_rows)
    write_csv(args.out / "cka_raw.csv", cka_rows)
    np.save(args.out / "final_token_ids.npy", final_token_ids)
    if cka_rows:
        cka = pd.DataFrame(cka_rows)
        cka.groupby(["step", "t", "layer_i", "layer_j"], as_index=False)["cka"].mean().to_csv(
            args.out / "cka_summary.csv", index=False
        )

    del model
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generated_texts = [r["generated"] for r in samples]
    ppl, mean_ppl, mean_entropy = evaluate_ppl(args, generated_texts)
    ppl_peak_gib = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    for row, p in zip(per_sample, ppl):
        row["per_sample_ppl"] = float(p)
        row["per_sample_log_ppl"] = float(math.log(max(float(p), 1e-12))) if math.isfinite(float(p)) else float("nan")
    write_csv(args.out / "per_sample_metrics.csv", per_sample)
    corr_rows = add_correlations(args.out, per_sample, step_rows)

    summary = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "sampling_method": args.sampling_method,
        "time_schedule": args.time_schedule,
        "sde_gamma": args.sde_gamma,
        "sde_noise_mode": args.sde_noise_mode,
        "self_cond_cfg_scale": args.self_cond_cfg_scale,
        "dynamic_rule": args.dynamic_rule,
        "seed": args.seed,
        "paired_rng": args.paired_rng,
        "sampling_peak_gpu_memory_gib": sampling_peak_gib,
        "ppl_peak_gpu_memory_gib": ppl_peak_gib,
        "mean_ppl": mean_ppl,
        "mean_entropy": mean_entropy,
        "num_nonempty": sum(bool(t.strip()) for t in generated_texts),
        "empty_fraction": 1.0 - sum(bool(t.strip()) for t in generated_texts) / max(len(generated_texts), 1),
        "unique_text_fraction": len(set(generated_texts)) / max(len(generated_texts), 1),
        "mean_final_decoder_margin": float(np.nanmean([r["final_decoder_margin_mean"] for r in per_sample])),
        "mean_final_decoder_margin_p10": float(np.nanmean([r["final_decoder_margin_p10"] for r in per_sample])),
        "mean_basin_entry_phase": float(np.nanmean([r["basin_entry_phase_mean"] for r in per_sample])),
        "median_basin_entry_phase": float(np.nanmedian([r["basin_entry_phase_p50"] for r in per_sample])),
        "top_correlations": corr_rows[:20],
    }
    with (args.out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
