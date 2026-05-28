# H3 FN Scorer Comparison

| cell_id              | head    | scorer   | params_total | n_seeds | macro_auroc       | mAP               |
| -------------------- | ------- | -------- | ------------ | ------- | ----------------- | ----------------- |
| mlp_fn_mlp           | mlp     | mlp      | 11,508,897   | 1       | 0.6798 +/- 0.0000 | 0.1114 +/- 0.0000 |
| mlp_fn_kan           | mlp     | kan      | 11,508,556   | 1       | 0.6858 +/- 0.0000 | 0.1128 +/- 0.0000 |
| kan_fn_kan           | fastkan | kan      | 11,548,046   | 1       | 0.6779 +/- 0.0000 | 0.1026 +/- 0.0000 |
| edge_contrastive     | fastkan | edge_mlp | 11,543,539   | 1       | 0.6806 +/- 0.0000 | 0.1061 +/- 0.0000 |
| edge_contrastive_kan | fastkan | edge_kan | 11,543,244   | 1       | 0.6781 +/- 0.0000 | 0.1036 +/- 0.0000 |
