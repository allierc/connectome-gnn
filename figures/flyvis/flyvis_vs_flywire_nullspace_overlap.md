# Flyvis vs hybrid-flywire null-space ranking overlap

How much do the two connectomes agree on which cell types carry the structural
weight-degeneracy?

**Datasets**
- *Flyvis*: 7-column extent, 13,741 neurons, 434,112 edges
  (`flyvis_noise_005`, ground-truth `ode_params.pt` from `flyvis_noise_free`).
- *Hybrid flywire*: full eye, 50,412 neurons, 1,266,378 edges
  (`full_eye_flywireRF_noise_005_blank50_cv00`).

Both datasets share the same 65 cell-type labels (canonical `INDEX_TO_NAME`),
which lets us match types by name across the two connectomes.

## Headline numbers

|                          | Flyvis        | Hybrid flywire   |
|--------------------------|--------------:|-----------------:|
| Neurons                  | 13,741        | 50,412           |
| Edges                    | 434,112       | 1,266,378        |
| `dim ker(H)`             | 308,160       | 801,764          |
| `dim ker(H)` / E         | **71.0 %**    | **63.3 %**       |
| Degenerate types         | 52            | 58               |
| Identifiable types       | 11            | 5                |
| No-outgoing types        | 2             | 2                |

The flywire connectome is ~3× larger and the absolute null-space dimension is
~2.6× larger, so the **null-space density is similar** (71 % vs 63 %).

## Top-9 overlap (Cedric's working list)

Flyvis top-9 (working list for opto perturbation):

| Rank | Type   | null_dim | % flyvis kernel |
|-----:|--------|---------:|----------------:|
| 1 | TmY15 | 43,299 | 14.1 % |
| 2 | Mi1   | 25,834 |  8.4 % |
| 3 | Tm3   | 20,471 |  6.6 % |
| 4 | Tm4   | 15,971 |  5.2 % |
| 5 | Tm1   | 15,525 |  5.0 % |
| 6 | Mi4   | 14,439 |  4.7 % |
| 7 | T4c   | 12,564 |  4.1 % |
| 8 | Mi9   | 11,889 |  3.9 % |
| 9 | Tm2   | 11,068 |  3.6 % |

Hybrid-flywire top-9 (recomputed from scratch):

| Rank | Type   | null_dim | % flywire kernel |
|-----:|--------|---------:|-----------------:|
| 1 | TmY15 | 109,316 | 13.6 % |
| 2 | Tm3   |  47,601 |  5.9 % |
| 3 | Mi1   |  43,541 |  5.4 % |
| 4 | T4c   |  43,520 |  5.4 % |
| 5 | Tm4   |  41,962 |  5.2 % |
| 6 | Mi4   |  40,774 |  5.1 % |
| 7 | T5c   |  32,074 |  4.0 % |
| 8 | Mi9   |  30,258 |  3.8 % |
| 9 | Tm1   |  28,420 |  3.5 % |

**Overlap of the two top-9 sets: 8 / 9 (89 %).**
The eight types appearing in both top-9 lists are
**TmY15, Mi1, Tm3, Tm4, Tm1, Mi4, T4c, Mi9.**
The single disagreement is **Tm2 (flyvis #9) ↔ T5c (flywire #7)**:

- Tm2 drops from rank 9 in flyvis to rank 15 in flywire.
- T5c rises from rank 10 in flyvis to rank 7 in flywire.

Both swapped types still belong to the motion-detection family (T5 inputs vs T5
output sub-class), so the qualitative motif is preserved.

## Coverage of the kernel by the working list

| List of 9 types                      | flyvis kernel covered | flywire kernel covered |
|--------------------------------------|----------------------:|-----------------------:|
| Flyvis top-9 (working list)          | 55.5 %                | 50.3 %                 |
| Flywire top-9 (recomputed)           | —                     | 52.1 %                 |
| TmY15 + Mi1                          | 22.4 %                | 19.1 %                 |
| TmY15 alone                          | 14.1 %                | 13.6 %                 |

The working list captures roughly half of the structural kernel in **both**
connectomes; the relative weight of TmY15 is essentially identical across
datasets.

## Top-k overlap statistics

| k  | overlap | %    | shared types |
|---:|--------:|-----:|--------------|
|  3 | 3 / 3   | 100% | TmY15, Mi1, Tm3 |
|  5 | 4 / 5   |  80% | TmY15, Mi1, Tm3, Tm4 (flyvis adds Tm1; flywire adds T4c) |
|  7 | 6 / 7   |  86% | + Mi4, T4c |
|  9 | 8 / 9   |  89% | + Mi9, Tm1   (Tm2 ↔ T5c is the only disagreement) |
| 10 | 9 / 10  |  90% | + T5c |
| 15 | 14 / 15 |  93% | (only T5d/CT1(Lo1) swap) |
| 20 | 18 / 20 |  90% | |
| 30 | 28 / 30 |  93% | |

**Spearman rank correlation across all 52 shared degenerate types: ρ = 0.958
(p ≈ 1.2·10⁻²⁸).**
Pearson r on `null_dim`: 0.966; Pearson r on `log10(null_dim)`: 0.966.

## Identifiable / no-outgoing classes

|                       | Flyvis                                       | Flywire                          |
|-----------------------|----------------------------------------------|----------------------------------|
| Identifiable (k=1 only) | Am, Lawf1, Mi3, **R1, R2, R3, R4, R5, R6**, T1, Tm5Y | Am, Lawf1, Mi3, T1, Tm5Y |
| No outgoing edges       | Mi11, Tm30                                   | Mi11, Tm30                       |

The shared identifiable core is **{Am, Lawf1, Mi3, T1, Tm5Y}** — same in both
connectomes. The six photoreceptors **R1–R6** flip from *identifiable* in flyvis
to *barely degenerate* in flywire (`null_dim` = 4 each — i.e. only one or two
multi-edge photoreceptor groups in the entire connectome). They contribute a
combined 24 directions out of 801,764 (0.003 %), so this difference is
quantitatively negligible.

The two no-outgoing types (Mi11, Tm30) are identical across connectomes.

## Bottom line

The "convergent integration" / motion-detection input motif identified in
flyvis (TmY15, Mi1, Tm3, Mi4, Tm1, Tm4, T4c, Mi9 → bulk of `dim ker(H)`)
**transfers almost unchanged to the hybrid-flywire connectome**:

- 8 of 9 working-list types remain in the flywire top-9.
- Tm2 (flyvis #9) is replaced by T5c (flywire #7); both are inputs/outputs of
  the same T4/T5 motion-detection circuit.
- Across all 52 shared degenerate types the rank correlation is
  ρ = 0.96, indicating the per-type kernel contribution is **highly stable**
  between the two connectomes despite the 3× difference in retinotopic extent.

This means the optogenetic-perturbation target list is robust to the choice
of flyvis vs hybrid-flywire as the underlying connectome — Mi1, Mi4, Tm3 (the
ones with existing Gal4 lines) are top-3 contributors in both.

---

*Source data*
- Flyvis: `figures/structural_nullspace_table.json` (produced by
  `src/connectome_gnn/models/flyvis_nullspace.py`).
- Flywire: `figures/structural_nullspace_table_flywire.json` (produced by
  `src/connectome_gnn/models/flywire_nullspace.py` — same Step 2/3
  computation reapplied to `full_eye_flywireRF_noise_005_blank50_cv00/ode_params.pt`).
