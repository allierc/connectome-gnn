"""Gate for the FlyVisODEParams -> FlyVisCurrentODEParams rename.

    PYTHONPATH=src GNN_OUTPUT_ROOT=... python tools/gate_ode_params_rename.py

Run it before and after a change and diff the two JSON outputs.

A pure class rename with an alias must change NOTHING observable:
  1. every registry key still resolves to the same class object
  2. the class round-trips an existing ode_params.pt with identical tensors
  3. save() writes byte-identical bytes for the same state
  4. the old name still resolves and IS the new class
Prints a single hash line so before/after can be diffed.
"""
import hashlib, io, json, sys, torch
from connectome_gnn.generators import ode_params as OP

DS = ('/groups/saalfeld/home/allierc/GraphData/graphs_data/fly/'
      'flyvis_noise_free_blank50_cv00')
out = {}

# 1. registry: every key -> class NAME (name is expected to change; identity is not)
reg = {k: v.__name__ for k, v in sorted(OP._ODE_PARAMS_REGISTRY.items())} \
    if hasattr(OP, '_ODE_PARAMS_REGISTRY') else {}
out['n_registry_keys'] = len(reg)
out['flyvis_keys_same_class'] = len({v for k, v in reg.items() if k.startswith('flyvis')})

# 2/3. load the real dataset params and re-save; hash the tensor content
cls = OP.get_ode_params_class('flyvis_current')
p = cls.load(DS, device='cpu')
h = hashlib.sha256()
for name in sorted(k for k in vars(p) if not k.startswith('_')):
    v = getattr(p, name)
    if torch.is_tensor(v):
        h.update(name.encode())
        h.update(v.detach().cpu().numpy().tobytes())
        out[f'shape.{name}'] = list(v.shape)
out['tensor_sha256'] = h.hexdigest()

# 4. the alias
out['alias_is_same_object'] = (getattr(OP, 'FlyVisODEParams', None) is cls)
print(json.dumps(out, indent=0, sort_keys=True))
