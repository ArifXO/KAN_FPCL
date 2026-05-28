# H2 InfoNCE vs FN-weighted (with rare-class AUROC)

| cell_id     | head | loss        | scorer | n_seeds | macro_auroc       | rare_disease_auroc | mAP               |
| ----------- | ---- | ----------- | ------ | ------- | ----------------- | ------------------ | ----------------- |
| mlp_infonce | mlp  | infonce     | none   | 1       | 0.6749 +/- 0.0000 | 0.6604 +/- 0.0000  | 0.1117 +/- 0.0000 |
| mlp_fn_mlp  | mlp  | fn_weighted | mlp    | 1       | 0.6798 +/- 0.0000 | 0.6608 +/- 0.0000  | 0.1114 +/- 0.0000 |
