# HOWTO: add a new zebrafish circuit

A **circuit** is "which neurons + how they connect": a named, sign-locked,
spectrally-rescaled adjacency template (`W^con`) plus cell-type metadata and the
named sub-populations the model wires its IO gate to. It carries no task and no
IO-mapping info. The canonical reference is the module docstring of
[`circuits.py`](../src/connectome_gnn/generators/circuits.py); this file is the
operational companion, focused on **Procedure B** — adding a circuit that
introduces a *new afferent / input population* (the case we actually need next,
e.g. `HNd`).

---

## 0. Current state

Three circuits are registered; configs select one via a single YAML field.

| `circuit.name` | IPN12 | N | connectome dir | selected by |
|---|---|---|---|---|
| `zebrafish_HD_731_v1` | without | 731 | `figures/zebrafish/zebrafish_connectome_HD` | *(dormant — no config)* |
| `zebrafish_HD_IPN12_839_v1` | with, inhibitory | 839 | `figures/zebrafish/zebrafish_connectome_HD_IPN12` | most configs |
| `zebrafish_HD_IPN12_839_v2` | with, excitatory | 839 | same dir | `*_v2.yaml` |

Selection lives in the `circuit:` block of every config, e.g.
[`zebrafish_hd_si_gnn_ipn12_v1_cv0.yaml`](../config/zebrafish/zebrafish_hd_si_gnn_ipn12_v1_cv0.yaml):

```yaml
circuit:
  name: zebrafish_HD_IPN12_839_v1
```

Resolved at model-build time via `get_circuit(name)`. If `circuit.name` is
omitted, the model falls back to the legacy `simulation.connconstr_datapath`
(raw-CSV) path.

---

## The fork: which procedure do you need?

- **Procedure A — same afferent taxonomy.** The new pool still uses only the
  four L/R afferent subpops `RIPN_L/R`, `ptIPN_L/R`. Covers: re-fetches of the
  same families, IPN12 sign variants, ablations, spectral/Dale retunes. **Just
  steps A1–A5 below; nothing downstream changes.**
- **Procedure B — a new afferent type** (`HNd`, rhombomere-2/3 commissural,
  other pretectal/IPN subtypes). The IO path is hard-bound to the four afferent
  subpops at three coupled seams, plus a hardcoded `n_input` and a positional
  `[:n_dipn]` readout slice; a new type is dropped, rejected, or unexpressable at
  each. **Procedure A + the un-hardcodings (i)–(v) in §B.**

> The seam to widen is the same everywhere: make each consumer read the
> circuit's **declared** subpops (`Circuit.subpops` is already a free-form dict)
> — `afferent_*` on the encoder side, `readout` on the decoder side — instead of
> the hardcoded `RIPN/ptIPN` four and the `[:n_dipn]` readout slice. That is also
> exactly what the extended multi-modal input model in
> [`zebrafish.tex`](zebrafish.tex) (the `(channel × subpop)` gate matrix) needs —
> so **Procedure B and the extended-input refactor are one job.**

---

## Procedure A (same taxonomy) — the baseline

1. **Cache the connectome CSVs** under `figures/zebrafish/<new_dir>/`:
   `neurons.csv` (`bodyId,type,instance,side,somaLocationX/Y/Z`) and
   `connections.csv` (`bodyId_pre,bodyId_post,weight`). Use
   [`fetch_zebrafish_connectivity_HD_IPN12.py`](../figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12.py)
   as the template (needs a neuprint-fish2 token).
2. **Write a `build()` + `register_circuit("<name>_vN", build)`** in
   `circuits.py` (copy `_register_zebrafish_hd_731`, the shortest example); add
   the `_register_*()` call to `_discover_circuits`.
3. **Point a YAML at it:** `circuit:\n  name: <name>_vN`.
4. **Document it** in `zebrafish.tex`.
5. **Regenerate the calcium mapping** (only if neuron set/order changed and you
   train with the obs loss):
   ```bash
   python -m connectome_gnn.generators.make_calcium_dataset \
       --circuit <new_name> --connectome figures/zebrafish/<new_dir> \
       --out graphs_data/zebrafish/<new_calcium_dataset>
   ```
   `calcium_mapping.pt` is bodyId-keyed so it re-derives automatically, but it
   **must** be re-run — a mapping built for another circuit silently gathers the
   wrong model neurons. Then set `training.calcium_dataset` to the new dataset.

---

## Procedure B (NEW afferent type) — the worked path for `HNd`

Do everything in Procedure A, with the following changes.

### B1. Extend the fetch type-list (step A1)

In [`fetch_zebrafish_connectivity_HD_IPN12.py`](../figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12.py),
add the new type(s) to the explicit list (lines 49–66) so the EM fetch actually
pulls those cells and their edges into the closed subgraph:

```python
HND_TYPES   = ["HNd"]                 # dorsal habenula (the big missing input)
# optionally: R2CO_TYPES = ["r2co2", ...]; R3CO_TYPES = ["r3co5", "r3co2", ...]
HD_IPN12_TYPES = (IPND_TYPES + IPNDS_TYPES + RIPN_TYPES + PTIPN_TYPES
                  + IPN12_TYPES + HND_TYPES)
```

Save to a **new** directory (e.g. `zebrafish_connectome_HD_IPN12_HNd/`) and
register a new circuit name (e.g. `zebrafish_HD_IPN12_HNd_<N>_v1`) — never
mutate the existing 839-cell pool in place.

### B2. Un-hardcode the four seams

| # | Seam | File / symbol | What breaks today | Fix |
|---|------|---------------|-------------------|-----|
| (i) | **Loader category map** | `connconstr_data._ZHD_AFFERENT_PREFIXES` ([:296](../src/connectome_gnn/generators/connconstr_data.py#L296)) and `_zhd_category()` (returns `""` for unknown → row rejected) | a `HNd` row is dropped, not loaded | add `"HNd"` to the afferent prefixes + category map; build its `HNd_L/R` subpop in the `afferent_subpop_ix` section (~[:519](../src/connectome_gnn/generators/connconstr_data.py#L519)) alongside the RIPN→PENa / pt-IPN→PENb split |
| (ii) | **Circuit→dict bridge** | `Circuit.to_dict()` afferent keys ([circuits.py:204](../src/connectome_gnn/generators/circuits.py#L204)) hardwires exactly `RIPN_L/R, ptIPN_L/R` | any other declared subpop is silently dropped | pass through **all** `afferent_*` keys the circuit declares, not the fixed four |
| (iii) | **Model velocity gate** | `velocity_gate='pen_4scalar'` in [`zebrafish_hd_task_gnn.py:202`](../src/connectome_gnn/models/zebrafish_hd_task_gnn.py#L202) / `rnn.py:248`; **raises** if the four required keys are absent ([:212](../src/connectome_gnn/models/zebrafish_hd_task_gnn.py#L212)) | builds exactly 4 scalars; assumes the 4-pop taxonomy | replace with a gate over the **N** declared afferent subpops — the per-`(channel × subpop)` matrix `V` of the extended-input model |
| (iv) | **Input dimension** | `self.n_input = 3` hardcoded ([gnn:134](../src/connectome_gnn/models/zebrafish_hd_task_gnn.py#L134), [rnn:147](../src/connectome_gnn/models/zebrafish_hd_task_rnn.py#L147)); config `n_input` ([config.py:618](../src/connectome_gnn/config.py#L618)) ignored | the multi-channel input vector can't be expressed | read `n_input` from config / dataset meta; let the gate span visual + self-motion + cue channels |
| (v) | **Readout** (decoder-side mirror) | `output_from_dipn_only` ([config.py:577](../src/connectome_gnn/config.py#L577)) + positional slice `r[..., :n_dipn]` ([gnn:478](../src/connectome_gnn/models/zebrafish_hd_task_gnn.py#L478), rnn) — the decoder reads the first `n_dipn` cells by **ordering convention**, not a declared set | a circuit whose readout cells differ / are non-contiguous can't be expressed | declare a `readout` subpop and have `W_out` `index_select` it instead of slicing `[:n_dipn]` |

### B3. Declare the new subpop in `build()` (step A2)

`Circuit.subpops` is already free-form — just add the new keys so the
(generalized) bridge and gate can find them:

```python
subpops={
    "bump": ...,
    # encoder side — neurons W_in injects into:
    "afferent_RIPN_L":  ..., "afferent_RIPN_R":  ...,
    "afferent_ptIPN_L": ..., "afferent_ptIPN_R": ...,
    "afferent_HNd_L":   ..., "afferent_HNd_R":   ...,   # NEW afferent
    # decoder side — neurons W_out reads from (replaces output_from_dipn_only):
    "readout": ...,                                     # NEW declared readout set
},
```

> **Encoder/decoder symmetry.** The afferent list (where input is injected) and
> the `readout` list (where the decoder reads) are mirror images — both are
> declared index sets the model should gather, not hardcoded conventions.
> Generalizing the readout (chokepoint v) is the same widen-the-consumer move as
> the afferent gate; do them together.

### B4. New config params

Add a config that selects the new circuit and the wider input. New/used keys:

```yaml
circuit:
  name: zebrafish_HD_IPN12_HNd_<N>_v1
graph_model:
  velocity_gate: <new_generic_mode>   # superset of pen_4scalar over N subpops
  n_input: <C>                         # e.g. 5: ω, swim_fwd, swim_turn, cosθ0, sinθ0
training:
  calcium_dataset: <new_calcium_dataset>
  observation_neurons: all             # 'exclude_afferent' is defined by the
                                       # 4-type taxonomy — revisit if HNd is observed
```

`velocity_gate` is currently a `Literal["none","pen_only","pen_4scalar"]`
([config.py:612](../src/connectome_gnn/config.py#L612)); B(iii) adds the generic
mode to that enum.

### B5. Calcium mapping + observability

Re-run `make_calcium_dataset` for the new circuit (step A5). **Before** wiring
`HNd` as a supervised input, check whether it carries a ZAPBench functional
trace (is it among the 481 matched ROIs?): if yes it's a directly testable
input port; if not, it's an unobserved driver you can model but not supervise.

---

## Worked motivation: why `HNd`

The unrestricted partner census
([`census_zebrafish_partners_HD_IPN12.py`](../figures/zebrafish/census_zebrafish_partners_HD_IPN12.py))
shows the current 839-cell circuit is **not closed**: ~46% of the bump pool's
incoming synapses come from outside the model. The dominant missing input:

| input to bump pool | total synaptic weight |
|---|---|
| all modeled afferents (RIPN + pt-IPN) | 17,820 |
| **HNd alone (dorsal habenula, unmodeled)** | **18,282** |

A single unmodeled type outweighs the entire afferent set — which is the case
for Procedure B.

---

## Checklist

**Procedure A**
- [ ] CSVs cached under a new `figures/zebrafish/<dir>/`
- [ ] `build()` + `register_circuit("<name>_vN")` + added to `_discover_circuits`
- [ ] new YAML with `circuit.name`
- [ ] section in `zebrafish.tex`
- [ ] `make_calcium_dataset --circuit <name>` re-run; `calcium_dataset` updated

**Procedure B (additionally)**
- [ ] fetch type-list extended; new connectome dir + new circuit name
- [ ] (i) loader: prefix + `_zhd_category` + `afferent_subpop_ix` accept the new type
- [ ] (ii) `Circuit.to_dict` passes through all declared afferent subpops
- [ ] (iii) generic N-subpop velocity gate (extends `pen_4scalar`)
- [ ] (iv) `n_input` read from config, not hardcoded
- [ ] (v) declared `readout` subpop + `W_out` gathers it (replaces `output_from_dipn_only` slice)
- [ ] new subpop keys declared in `build()` (afferent **and** `readout`)
- [ ] `velocity_gate` enum + `n_input` set in config
- [ ] observability of the new type checked against the 481 ZAPBench ROIs
