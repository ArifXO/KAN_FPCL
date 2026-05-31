# H1 Geometry: MLP vs KAN vs res_KAN (test split)

| cell_id          | head             | loss    | params_total | n_seeds | macro_auroc  | alignment    | uniformity    | effective_rank | macro_auroc_knn |
| ---------------- | ---------------- | ------- | ------------ | ------- | ------------ | ------------ | ------------- | -------------- | --------------- |
| mlp_infonce      | mlp              | infonce | 11,499,584   | 1       | 0.6773 (n=1) | 0.2204 (n=1) | -3.8861 (n=1) | 116.95 (n=1)   | 0.6277 (n=1)    |
| kan_infonce      | fastkan          | infonce | 11,498,747   | 1       | 0.6804 (n=1) | 0.2078 (n=1) | -3.7664 (n=1) | 45.49 (n=1)    | 0.6170 (n=1)    |
| kan_wide_infonce | fastkan_wide     | infonce | 11,907,778   | 1       | 0.6801 (n=1) | 0.2224 (n=1) | -3.8262 (n=1) | 73.55 (n=1)    | 0.6172 (n=1)    |
| reskan_infonce   | residual_fastkan | infonce | 11,536,851   | 1       | 0.6743 (n=1) | 0.2265 (n=1) | -3.8846 (n=1) | 116.77 (n=1)   | 0.6286 (n=1)    |
