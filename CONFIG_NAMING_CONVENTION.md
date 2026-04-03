# Config Naming Convention — Consistency Documentation

**Updated: 2026-04-03**

This document confirms that config filenames are now **consistent between the LLM and subprocess (GNN_LLM pipeline)**.

## Quick Reference

For a run like: `python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_known_ode --cluster`

| File Type | Pattern | Example |
|-----------|---------|---------|
| **Config slot files** | `config/fly/{base_config_name}_Claude_00.yaml` to `_03.yaml` | `config/fly/flyvis_noise_005_known_ode_Claude_00.yaml` |
| **Winner config** | `config/fly/{base_config_name}_winner.yaml` | `config/fly/flyvis_noise_005_known_ode_winner.yaml` |
| **Memory file** | `{llm_task_name}_memory.md` (in exploration_dir) | `flyvis_noise_005_known_ode_Claude_memory.md` |
| **Analysis log** | `{llm_task_name}_analysis.md` (in exploration_dir) | `flyvis_noise_005_known_ode_Claude_analysis.md` |

## Variable Mapping

```
CLI argument:       flyvis_noise_005_known_ode
                              ↓
base_config_name:   flyvis_noise_005_known_ode
                              ↓
llm_task_name:      flyvis_noise_005_known_ode_Claude  (auto-generated)
```

## How They're Used

### 1. Pipeline (GNN_LLM.py and pipeline.py)
- Generates full paths using `state.llm_task_name` and `state.base_config_name`
- Creates config slot files: `config/fly/{base_config_name}_Claude_00.yaml` through `_03.yaml`
- Passes these paths to Claude in the prompt

### 2. Claude (in batch_0_prompt and analysis_prompt)
- Receives explicit file paths in the prompt (line 20-21 of prompts.py)
- Instruction file specifies matching filenames with `{base_config_name}` variable
- Edits config files at the paths provided
- Creates/updates memory and analysis logs at provided paths

### 3. Instruction File (instruction_flyvis_noise_005.md)
- **Line 210**: Config files pattern uses `{base_config_name}_Claude_XX`
- **Line 222**: Explains the full path: `config/fly/{base_config_name}_Claude_00.yaml`
- **Line 215-223**: "Variable Names" section clarifies what each variable means
- **Line 341**: Winner file uses `{base_config_name}_winner.yaml`

## Files Modified

1. **LLM/instruction_flyvis_noise_005.md**
   - Updated config file naming from `{name}_00.yaml` to `config/fly/{base_config_name}_Claude_00.yaml`
   - Changed hardcoded `flyvis_noise_005_winner.yaml` to dynamic `{base_config_name}_winner.yaml`
   - Added "Variable Names" section (line 215-223) for clarity
   - Updated "File Structure" references to use `{base_config_name}_Claude_memory.md`

2. **src/connectome_gnn/LLM/pipeline.py** 
   - ✅ No changes needed — already enforces correct naming via:
     - Line 66: `llm_task_name = f'{base_config_name}_Claude'` (auto-adds "_Claude")
     - Line 165: `slot_name = f"{state.llm_task_name}_{slot:02d}"` (adds slot number)

3. **src/connectome_gnn/LLM/prompts.py**
   - ✅ No changes needed — already passes full paths via `{slot_list}` and state paths

## Verification Checklist

- ✅ LLM instruction file tells Claude to edit: `config/fly/{base_config_name}_Claude_00.yaml`
- ✅ Pipeline creates config files at: `config/fly/{base_config_name}_Claude_00.yaml`
- ✅ Prompt passes full paths to Claude in slot_list
- ✅ Winner filename is dynamic: `{base_config_name}_winner.yaml`
- ✅ Memory and analysis files use: `{llm_task_name}_memory.md` and `{llm_task_name}_analysis.md`
- ✅ Variable names clearly explained in instruction file

## For Different Configurations

The naming convention automatically adapts:

| Base Config | llm_task_name | Slot Files |
|-------------|---------------|-----------|
| `flyvis_noise_free_known_ode` | `flyvis_noise_free_known_ode_Claude` | `config/fly/flyvis_noise_free_known_ode_Claude_00.yaml` |
| `flyvis_noise_05_known_ode` | `flyvis_noise_05_known_ode_Claude` | `config/fly/flyvis_noise_05_known_ode_Claude_00.yaml` |
| `flyvis_noise_005_removed_pc_10_known_ode` | `flyvis_noise_005_removed_pc_10_known_ode_Claude` | `config/fly/flyvis_noise_005_removed_pc_10_known_ode_Claude_00.yaml` |

All automatically generated and consistent! 🎯
