# Overnight run status

Updated 2026-07-26 19:59:41. 23/23 finished.

| arm | model | R2_W | slope | rollout r | tau R2 (inlier) | Vrest R2 (inlier) | clust |
|---|---|---|---|---|---|---|---|
| CX sigma=0 | gnn | 0.5417 | 0.9210 | 1.000 | nan |  | 0.1282 |
| CX sigma=0.05 | gnn | 0.8404 | 1.0136 | 1.000 | nan |  | 0.1282 |
| CX sigma=0.05 | oracle | 1.0000 | 1.0003 | 1.000 | nan | nan | 0.1410 |
| CX sigma=0 | oracle | -2.0684 | 0.9408 | 1.000 | nan | nan | 0.1410 |
| CX sigma=0.5 | gnn | 1.0000 | 1.0014 | 1.000 | nan |  | 0.1346 |
| CX sigma=0.5 | oracle | 1.0000 | 1.0000 | 1.000 | nan | nan | 0.1218 |
| full obs model | oracle | -0.1000 | 0.0123 | 0.044 | nan | 0.7907 | 0.4462 |
| full obs model | gnn | -0.0109 | 0.0000 | -0.010 | -7.5755 | 0.3699 | 0.1344 |
| kernel only | oracle | 0.3305 | 0.7761 | 0.111 | -28.3146 | 0.8783 | 0.6936 |
| kernel+rate 1/25 | oracle | 0.2753 | 0.6113 | 0.459 | -20.0111 | 0.9094 | 0.6645 |
| kernel+rate 1/25 | gnn | -0.0101 | 0.0000 | 0.172 | -5.0617 | 0.8392 | 0.2300 |
| kernel+saturation | oracle | 0.0072 | 0.3329 | -0.014 | -9.7766 | 0.8616 | 0.5305 |
| kernel+saturation | gnn | -0.0101 | -0.0000 | 0.043 | -4.5542 | -2.2390 | 0.1947 |
| kernel+shot g=0.1 | oracle | -0.3753 | 0.1246 | 0.019 | nan | 0.8722 | 0.5699 |
| kernel+shot g=0.1 | gnn | -0.0101 | -0.0000 | 0.041 | -28.7315 | nan | 0.2576 |
| kernel+shot g=0.2 | oracle | -0.3224 | 0.0692 | 0.008 | nan | 0.8633 | 0.4987 |
| kernel+shot g=0.2 | gnn | -0.0101 | 0.0000 | 0.041 | -4.9298 | nan | 0.2835 |
| kernel only | gnn | -0.0101 | -0.0000 | 0.041 | -3.6438 | -3.3507 | 0.2613 |
| deconv kernel-only | oracle | 0.2949 | 1.2318 | 0.647 | -8.2216 | 0.9393 | 0.8594 |
| deconv kernel-only | gnn | 0.0724 | 1.2192 | 0.688 | -4.3028 | 0.4279 | 0.7458 |
| deconv saturation | oracle | -0.9839 | 1.1378 | 0.080 | -3.9865 | 0.8498 | 0.7595 |
| deconv saturation | gnn | -0.0101 | -0.0000 | 0.060 | nan | 0.2682 | 0.3403 |
| voltage control | gnn | 0.9828 | 0.9922 | 1.000 | 0.9923 | 0.9379 | 0.9268 |
