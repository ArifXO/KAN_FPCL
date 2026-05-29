# H3 FN Scorer Comparison

| cell_id              | head    | scorer   | params_total | n_seeds | macro_auroc  | mAP          |
| -------------------- | ------- | -------- | ------------ | ------- | ------------ | ------------ |
| mlp_fn_mlp           | mlp     | mlp      | 11,508,897   | 1       | 0.6798 (n=1) | 0.1114 (n=1) |
| mlp_fn_kan           | mlp     | kan      | 11,508,556   | 1       | 0.6858 (n=1) | 0.1128 (n=1) |
| kan_fn_kan           | fastkan | kan      | 11,548,046   | 1       | 0.6779 (n=1) | 0.1026 (n=1) |
| edge_contrastive     | fastkan | edge_mlp | 11,543,539   | 1       | 0.6806 (n=1) | 0.1061 (n=1) |
| edge_contrastive_kan | fastkan | edge_kan | 11,543,244   | 1       | 0.6781 (n=1) | 0.1036 (n=1) |
