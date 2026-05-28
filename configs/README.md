# Config Directory

## Structure
- `data/` — Dataset configs (`chestmnist` active; `chexpert` deferred until Stage 9)
- `model/` — Encoder + head + scorer bundles
- `loss/` — Loss function configs
- `experiment/` — Full training configs (`smoke_*` for pipeline tests, `full_*` for ablation)

## Ablation Cell → Config Mapping

| Cell ID | Experiment Config | Model Config | Loss Config | Script |
|---|---|---|---|---|
| mlp_infonce | full_mlp_infonce | mlp_baseline | infonce | train.py |
| kan_infonce | full_kan_infonce | kan_head | infonce | train.py |
| reskan_infonce | full_reskan_infonce | residual_fastkan_warp | infonce | train.py |
| mlp_fn_mlp | full_mlp_fn_mlp | mlp_scorer | fn_weighted_mlp | train_fn.py |
| mlp_fn_kan | full_mlp_fn_kan | kan_scorer | fn_weighted_mlp | train_fn.py |
| reskan_fn_kan | full_reskan_fn_kan | residual_fastkan_warp + kan_scorer | fn_weighted_mlp | train_fn.py |
| zonly_fn | full_zonly_fn | edge_aware_mlp_zonly | edge_aware (λ=0, z-only scorer) | train_edge.py |
| edge_scorer_no_aux | full_edge_scorer_no_aux | edge_aware_mlp | edge_aware (λ=0) | train_edge.py |
| edge_contrastive | full_edge_l005 | edge_aware_mlp | edge_aware (λ_edge=0.05) | train_edge.py |
| edge_align | full_edge_align_l005 | edge_aware_mlp | edge_aware (λ_align=0.05) | train_edge.py |
| edge_contrastive_kan | full_edge_l005_kan_scorer | edge_aware_kan | edge_aware (λ_edge=0.05, KAN scorer) | train_edge.py |

## Smoke Configs
- `smoke_mlp.yaml` — 5-step MLP+InfoNCE (Stage 2 pipeline test)
- `smoke_fn_mlp.yaml` — 5-step FN-weighted (Stage 6 pipeline test)
- `smoke_edge_aware.yaml` — 5-step edge-aware (Stage 7.5 pipeline test)
