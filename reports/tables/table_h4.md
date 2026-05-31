# H4 Edge-aware Scorer Ablation (test split)

| cell_id            | loss          | scorer    | lambda_edge | lambda_edge_align | n_seeds | final_train_loss | final_val_loss | best_val_loss | macro_auroc  | rare_disease_auroc | mAP          |
| ------------------ | ------------- | --------- | ----------- | ----------------- | ------- | ---------------- | -------------- | ------------- | ------------ | ------------------ | ------------ |
| zonly_fn           | edge_aware_fn | mlp_zonly | 0.0         | 0.0               | 1       | 0.2053 (n=1)     | 0.2551 (n=1)   | 0.2551 (n=1)  | 0.6711 (n=1) | 0.6785 (n=1)       | 0.1026 (n=1) |
| edge_scorer_no_aux | edge_aware_fn | edge_mlp  | 0.0         | 0.0               | 1       | 0.2175 (n=1)     | 0.2461 (n=1)   | 0.2461 (n=1)  | 0.6744 (n=1) | 0.6781 (n=1)       | 0.1074 (n=1) |
| edge_contrastive   | edge_aware_fn | edge_mlp  | 0.05        | 0.0               | 1       | 0.2112 (n=1)     | 0.2417 (n=1)   | 0.2417 (n=1)  | 0.6748 (n=1) | 0.6747 (n=1)       | 0.1068 (n=1) |
| edge_align         | edge_aware_fn | edge_mlp  | 0.0         | 0.05              | 1       | 0.2046 (n=1)     | 0.2398 (n=1)   | 0.2398 (n=1)  | 0.6839 (n=1) | 0.6928 (n=1)       | 0.1045 (n=1) |
