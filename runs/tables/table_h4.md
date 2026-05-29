# H4 Edge-aware Scorer Ablation

| cell_id            | scorer    | lambda_edge | lambda_edge_align | n_seeds | macro_auroc  | rare_disease_auroc | mAP          |
| ------------------ | --------- | ----------- | ----------------- | ------- | ------------ | ------------------ | ------------ |
| zonly_fn           | mlp_zonly | 0.0         | 0.0               | 1       | 0.6808 (n=1) | 0.6868 (n=1)       | 0.1032 (n=1) |
| edge_scorer_no_aux | edge_mlp  | 0.0         | 0.0               | 1       | 0.6844 (n=1) | 0.6538 (n=1)       | 0.1076 (n=1) |
| edge_contrastive   | edge_mlp  | 0.05        | 0.0               | 1       | 0.6806 (n=1) | 0.6719 (n=1)       | 0.1061 (n=1) |
| edge_align         | edge_mlp  | 0.0         | 0.05              | 1       | 0.6867 (n=1) | 0.6771 (n=1)       | 0.1054 (n=1) |
