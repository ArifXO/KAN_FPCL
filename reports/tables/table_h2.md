# H2 InfoNCE vs FN-weighted, with rare-class AUROC (test split)

| cell_id     | head | loss        | scorer | n_seeds | macro_auroc  | rare_disease_auroc | mAP          |
| ----------- | ---- | ----------- | ------ | ------- | ------------ | ------------------ | ------------ |
| mlp_infonce | mlp  | infonce     | none   | 1       | 0.6773 (n=1) | 0.7053 (n=1)       | 0.1128 (n=1) |
| mlp_fn_mlp  | mlp  | fn_weighted | mlp    | 1       | 0.6859 (n=1) | 0.7093 (n=1)       | 0.1144 (n=1) |
