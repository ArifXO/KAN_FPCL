# H1 Geometry: MLP vs KAN vs res_KAN (VAL split (biased))

| cell_id        | head             | params_total | n_seeds | macro_auroc  | alignment    | uniformity    | effective_rank | macro_auroc_knn |
| -------------- | ---------------- | ------------ | ------- | ------------ | ------------ | ------------- | -------------- | --------------- |
| mlp_infonce    | mlp              | 11,499,584   | 1       | 0.6749 (n=1) | 0.2245 (n=1) | -3.8851 (n=1) | 116.92 (n=1)   | 0.6275 (n=1)    |
| kan_infonce    | fastkan          | 11,498,747   | 1       | 0.6718 (n=1) | 0.2074 (n=1) | -3.7682 (n=1) | 45.53 (n=1)    | 0.6194 (n=1)    |
| reskan_infonce | residual_fastkan | 11,536,851   | 1       | 0.6783 (n=1) | 0.2266 (n=1) | -3.8830 (n=1) | 116.30 (n=1)   | 0.6289 (n=1)    |
