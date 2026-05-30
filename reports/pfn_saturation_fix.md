# p_fn saturation fix (Pitfall #5)

Source: CLAUDE.md "Scorer Collapse Under FN-Weighted Loss" + the user's
2026-05-30 task brief.

## What scorer saturation is

`FNWeightedInfoNCELoss` rewards downweighting negatives: the per-pair
weight in the denominator is `1 - p_fn`, so larger `p_fn` ⇒ smaller
denominator ⇒ smaller loss for any embedding configuration. Without a
counterweight the scorer's trivial optimum is to push every `p_fn` to
`max_fn_weight`. Smoke trace `run_smoke_fn_mlp_20260521-195829-d1d901`
showed `p_fn_mean` drifting 0.533 → 0.546 in 5 steps at
`max_fn_weight=1.0` — the saturation signature.

## Why it invalidates interpretation

Once the scorer saturates, the FN-weighted loss is just a globally weakened
InfoNCE. Any AUROC delta we report under H2/H3/H4 may then be attributable
to the weakened global negative pressure, not to actual false-negative
detection. We cannot tell from the contrastive loss alone whether the
scorer is *ranking* suspicious pairs or just suppressing all of them.

## What changed

### Diagnostics (`src/losses/pfn_diagnostics.py`)
`FNWeightedInfoNCELoss` and `EdgeAwareFNWeightedInfoNCELoss` now return a
richer R7 dict including raw and clipped p_fn statistics, the per-step cap,
the negative-weight curve, and entropy:

* `p_fn_raw_{mean,std,min,max}` — scorer output before clipping.
* `p_fn_clipped_{mean,std,min,max}` — values used in the denominator.
* `p_fn_at_cap_fraction` — fraction of raw entries ≥ `max_fn_weight_current`.
* `p_fn_near_zero_fraction` — fraction ≤ 0.05.
* `p_fn_entropy_mean` — Bernoulli entropy averaged over entries.
* `effective_neg_weight_{mean,min}` — `(1 - p_fn_clipped)` on negative slots.
* `max_fn_weight_current` — current cap (with the schedule applied).

All of these flow into `step_metrics.csv` / `step_metrics.json` via
`base_step_metrics` and the existing CSV writer in `train_fn.py` /
`train_edge.py`.

### Regularization
Mean-prior + cap penalties — both gradient-bearing on the raw `p_fn` so
they can pull the scorer back from saturation:

* **Mean prior** `(mean(p_fn_raw) - target_mean)²` — defaults
  `target_mean=0.05`, `lambda_mean=0.1`.
* **Cap penalty** `mean(relu(p_fn_raw / max_cap - cap_margin)²)` —
  defaults `lambda_cap=0.1`, `cap_margin=0.98`.

> Deviation from spec: the spec wrote the cap penalty on `p_fn_clipped`.
> `clamp` is non-differentiable above the cap, so the literal version
> would carry no gradient back to the scorer once it had already saturated
> — the exact state we need to undo. Using the raw (sigmoid-bounded)
> output is gradient-bearing everywhere and is documented in
> `src/losses/pfn_diagnostics.py`.

`pfn_reg_total` carries live gradient; the train loop adds it to the
optimizer objective. The contrastive `loss` key is untouched, so the R3
backward-compat invariant `p_fn=0 ⇒ InfoNCE allclose 1e-5` still holds.

### Schedule
`compute_max_fn_weight_current(epoch, schedule_cfg, static_max_fn_weight)`
in `scripts/train/train_common.py` produces an epoch-driven cap with a
warmup window + linear ramp. Defaults: `warmup_epochs=5, ramp_epochs=10,
start=0.0, end=0.5`. During warmup the loss is exactly InfoNCE (no FN
downweighting), so the scorer has no incentive to inflate `p_fn` until the
encoder/projector have a reasonable embedding to score on.

The train loop calls `compute_max_fn_weight_current` once per epoch and
passes the result as `max_fn_weight_override` to `loss_fn.forward(...)`.

### Scorer init + input detach
* `init_pfn_prior` (default 0.05) bias-initialises the final linear so the
  scorer's mean output starts near 5 %, not the saturation-prone 50 %.
  Applied to `MLPPairScorer`, `KANPairScorer`, `EdgeAwarePairScorer`.
* `detach_inputs` detaches `z` (and `edge_features` on the edge scorer)
  at the scorer boundary so scorer gradients cannot reshape the
  encoder/projector toward features that maximise downweighting.
  Default `True` for the plain FN scorers; **default `False` for
  `EdgeAwarePairScorer`** to keep the H4 scorer→edge→KAN gradient path
  alive. The `test_gradient_flows_through_edge_to_kan_weights` test
  guards this default.

## Files touched

| Area | File |
|------|------|
| Loss | `src/losses/fn_weighted_infonce.py` (rewrite), `src/losses/edge_aware_fn_loss.py` (pass-through), `src/losses/pfn_diagnostics.py` (new) |
| Scorers | `src/models/pair_scorer.py`, `src/models/edge_aware_scorer.py` |
| Train | `scripts/train/train_common.py` (schedule helper + base_step_metrics), `scripts/train/train_fn.py`, `scripts/train/train_edge.py` |
| Configs | `configs/loss/fn_weighted_mlp.yaml`, `configs/loss/edge_aware.yaml`, scorer configs under `configs/model/`, `configs/experiment/debug_pfn_saturation.yaml` |
| Analysis | `scripts/analysis/analyze_pfn_saturation.py` (new) |
| Tests | `tests/test_pfn_saturation_fix.py` (new), updates to `tests/test_pair_scorer.py`, `tests/test_edge_aware_fn_loss.py` |

## How to decide if the fix worked

Run `scripts/analysis/analyze_pfn_saturation.py --run-dir <run>` and look
for the following on a real training run:

| Signal | Healthy | Suspect |
|--------|---------|---------|
| `p_fn_at_cap_fraction` (tail mean) | `< 0.10` | `> 0.25` for ≥ 2 epochs |
| `p_fn_raw_std` | `> 0.05` (scorer ranks pairs) | `< 0.02` with high mean (collapse) |
| `effective_neg_weight_mean` | `> 0.6` | `< 0.4` (most negatives suppressed) |
| `p_fn_raw_mean` | drifts near `target_mean` | pinned to `max_fn_weight` |

Once a run looks healthy by those signals, compare its downstream probe
AUROC against the InfoNCE baseline (R2 — Stage 2 result is the anchor).
The H2/H3/H4 tables can then carry `p_fn_at_cap_fraction` next to AUROC
so a reader can verify the FN scorer was *doing something* rather than
silently shutting off negatives.

## Backward compatibility

Numerical invariants verified by `tests/`:

1. `p_fn=0 ⇒ FNWeightedInfoNCELoss == InfoNCE` (allclose 1e-5), with reg
   on or off.
2. `max_fn_weight_override=0` (warmup) ⇒ FN loss == InfoNCE.
3. EdgeAwareFNWeightedInfoNCELoss with `lambda_edge=lambda_edge_align=0`
   still matches Stage 7 FN loss exactly.
4. H4 scorer→edge→KAN gradient path remains alive at the default
   `detach_inputs=False` for `EdgeAwarePairScorer`.

Legacy `train.lambda_pfn_reg` is still honored (added on top), but the new
in-loss regularization supersedes it. New configs set it to 0.0.
