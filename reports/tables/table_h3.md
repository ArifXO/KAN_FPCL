# H3 FN Scorer Comparison (test split)

| cell_id              | head    | loss          | scorer   | params_total | n_seeds | final_train_loss | final_val_loss | best_val_loss | macro_auroc  | mAP          |
| -------------------- | ------- | ------------- | -------- | ------------ | ------- | ---------------- | -------------- | ------------- | ------------ | ------------ |
| mlp_fn_mlp           | mlp     | fn_weighted   | mlp      | 11,508,897   | 1       | 0.1494 (n=1)     | 0.1613 (n=1)   | 0.1613 (n=1)  | 0.6859 (n=1) | 0.1144 (n=1) |
| mlp_fn_kan           | mlp     | fn_weighted   | kan      | 11,508,556   | 1       | 0.1430 (n=1)     | 0.1573 (n=1)   | 0.1573 (n=1)  | 0.6872 (n=1) | 0.1124 (n=1) |
| kan_fn_kan           | fastkan | fn_weighted   | kan      | 11,548,046   | 1       | 0.2233 (n=1)     | 0.2495 (n=1)   | 0.2495 (n=1)  | 0.6772 (n=1) | 0.1048 (n=1) |
| edge_contrastive     | fastkan | edge_aware_fn | edge_mlp | 11,543,539   | 1       | 0.2112 (n=1)     | 0.2417 (n=1)   | 0.2417 (n=1)  | 0.6748 (n=1) | 0.1068 (n=1) |
| edge_contrastive_kan | fastkan | edge_aware_fn | edge_kan | 11,543,244   | 1       | 0.2236 (n=1)     | 0.2340 (n=1)   | 0.2340 (n=1)  | 0.6764 (n=1) | 0.1046 (n=1) |
