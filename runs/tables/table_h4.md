# H4 Edge-aware Scorer Ablation

| cell_id            | scorer    | lambda_edge | lambda_edge_align | n_seeds | macro_auroc       | rare_disease_auroc | mAP               |
| ------------------ | --------- | ----------- | ----------------- | ------- | ----------------- | ------------------ | ----------------- |
| zonly_fn           | mlp_zonly | 0.0         | 0.0               | 1       | 0.6808 +/- 0.0000 | NA                 | 0.1032 +/- 0.0000 |
| edge_scorer_no_aux | edge_mlp  | 0.0         | 0.0               | 1       | 0.6844 +/- 0.0000 | NA                 | 0.1076 +/- 0.0000 |
| edge_contrastive   | edge_mlp  | 0.05        | 0.0               | 1       | 0.6806 +/- 0.0000 | NA                 | 0.1061 +/- 0.0000 |
| edge_align         | edge_mlp  | 0.0         | 0.05              | 1       | 0.6867 +/- 0.0000 | NA                 | 0.1054 +/- 0.0000 |
