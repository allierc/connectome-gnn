# Experiments

| # | name | purpose | landed | file |
|---|---|---|---|---|
| 0 | gpu_benchmark | decide which GPU to run the campaign on: measure iterations per second on RTX 6000 and A100, and check that bf16 costs no recovered precision at the same seeds | 0/4 | [`exp00_gpu_benchmark.md`](exp00_gpu_benchmark.md) |
| 1 | derivative_target | regenerate slide 5 and 6 of presentation/conductance.pdf with sound training (one parameter changes only) -- it was not -- and look at these plots again | 0/30 | [`exp01_derivative_target.md`](exp01_derivative_target.md) |
| 2 | conductance_lasso | does the general form g_phi = MLP(a_i, a_j, v_i, v_j) recover the circuit as well as the current form once the group lasso is strong enough to prune its extra inputs, across the model-noise sweep; and does that lasso kill the per-edge offset C_ij, read as R2_Vrest against R2_Vrest without the C_i correction | 0/30 | [`exp02_conductance_lasso.md`](exp02_conductance_lasso.md) |
