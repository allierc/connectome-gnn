# Degenerate Connectivity Matrix Generation - COMPLETE ✓

## Summary

Successfully generated **1780 degenerate connectivity variants** from the flyvis connectome using proper cell type assignments.

### Key Achievements

1. **Cell Type Assignment Verified** ✓
   - Loaded 65 cell type names from flyvis connectome JSON
   - Extracted neuron-to-type mapping from zarr files
   - Verified: neurons of same type have IDENTICAL tau/Vrest values
   - Verified: different types are biophysically DISTINCT

2. **Variant Generation Completed** ✓
   - Section 1: 780 single-type variants (52 types × 15 scales)
   - Section 2: 1000 mixed-type variants (top 10 types)
   - Total: 1780 variants (1 ground truth + 1779 variants)
   - Output location: `graphs_data/degenerate_matrix/`

3. **Statistics Tracking** ✓
   - Connectivity R² values calculated for all variants
   - Color-coded output: green (>0.95), orange (>0.50), red (<0.50)
   - Mixed-type variants: mean R² = 0.8387 ± 0.1644

---

## Detailed Results

### Cell Types by Degeneracy Status

**52 Types WITH degenerate groups** (perturbed in variants):
- R cells (R2, R3, R4, R5, R6, R7, R8) - all except R1
- L cells (L1, L2, L4, L5) - all except L3
- Lawf1, Am, C2, C3, CT1(Lo1), CT1(M10)
- Mi cells (Mi2, Mi3, Mi14, Mi15) - not Mi1, Mi4, Mi9, Mi10, Mi11, Mi12, Mi13
- T cells (T2, T2a, T3, T4a, T4b, T4c, T4d, T5a, T5b, T5c, T5d) - not T1
- Tm cells (Tm1, Tm2, Tm3, Tm4, Tm5Y, Tm5a, Tm5c, Tm16, Tm20, Tm28, Tm30, TmY3, TmY4, TmY5a, TmY9, TmY10, TmY13, TmY14, TmY15, TmY18) - not Tm5b, Tm9

**13 Types WITHOUT degenerate groups** (cannot perturb):
- R1, L3, Lawf2, Mi1, Mi4, Mi9, Mi10, Mi11, Mi12, Mi13, T1, Tm5b, Tm9
- These have sparse or one-to-one connectivity patterns

### Section 1: Single-Type Variants

**52 types × 15 scales = 780 variants**

Scale factors: [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

Example results (R² @ scale 8.0):
- R2: 0.9421 (orange)
- R3: 0.9672 (green)
- R4: 0.7408 (orange)
- R5: 0.9921 (green)
- L5: 0.7332 (orange)
- Tm5c: 0.2823 (red - unusual, needs rollout verification)
- C3: 1.0000 (perfect - minimal degenerate structure)

### Section 2: Mixed-Type Variants

**1000 variants perturbing top 10 types simultaneously**

Top 10 types by null space dimension:
1. TmY9 - null_dim: 43,299
2. L5 - null_dim: 25,834
3. Tm5a - null_dim: 20,471
4. Tm5c - null_dim: 15,971
5. Tm1 - null_dim: 15,525
6. Mi2 - null_dim: 14,439
7. T4c - null_dim: 12,564
8. Mi3 - null_dim: 11,889
9. Tm3 - null_dim: 11,068
10. T5c - null_dim: 8,765

**Final statistics:**
- Connectivity R²: mean = 0.8387 ± 0.1644
- Range: min = 0.0568, max = 0.9989
- These are perturbations that PRESERVE connectivity structure

---

## Next Steps

### 1. Compute Rollout R² (Dynamics Preservation)

The connectivity R² values above measure how well the WEIGHTS are preserved.
The rollout R² measures whether the DYNAMICS are preserved.

```bash
python scripts/rollout_degenerate_W.py
```

This will:
- Run ODE simulations for each variant
- Compare trajectory predictions vs ground truth
- Generate rollout R² statistics

### 2. Fill Results in degeneracy_analysis.tex

Once rollout results are available, update the blanks in:
- `docs/degeneracy_analysis.tex` Section 3

Insert:
- Connectivity R² statistics (from above)
- Rollout R² statistics (from rollout script)
- Cell type list and analysis

---

## Technical Notes

### Cell Type Source
- Names: flyvis connectome JSON (`fib25-fib19_v2.2.json`)
- Assignments: zarr files from simulated datasets (`neuron_type.zarr`)
- Validation: tau/Vrest identity within cell types confirms correctness

### Perturbation Strategy
- Sum-preserving within (postsynaptic neuron, presynaptic type) groups
- Maintains network connectivity structure (edges unchanged)
- Exploits within-type redundancy (the null space)
- Log-uniform amplitude sampling for mixed variants

### Connectivity R² Notes
- R² < 0.5 (red): Large perturbations, still preserve connectivity
- 0.5 < R² < 0.95 (orange): Moderate perturbations
- R² > 0.95 (green): Small perturbations, nearly ground truth
- R² ≈ 1.0: Minimal or no degenerate structure in this type

---

## File Organization

```
graphs_data/degenerate_matrix/
├── variant_00_ground_truth/
│   └── ode_params.pt (reference)
├── type_XX_scale_YY/ (780 variants)
│   ├── ode_params.pt
│   └── metadata.pt
└── mixed_types_var_ZZZZ/ (1000 variants)
    ├── ode_params.pt
    └── metadata.pt
```

Each variant includes:
- Modified weight matrix (W)
- Original edge indices (connectivity unchanged)
- ODE parameters (tau, V_rest unchanged)
- Metadata: scale factor, R², null dimension, seed

---

Generated: 2026-04-05
Cell types: 65 (52 with degeneracy)
Variants: 1780 (+ 1 ground truth)
Status: ✓ Complete and ready for rollout analysis
