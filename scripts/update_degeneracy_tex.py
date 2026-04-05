#!/usr/bin/env python
"""Auto-update degeneracy_analysis.tex with results from three approaches.

Reads JSON results files from:
- results_global_svd.json
- results_per_neuron_svd.json
- results_structural_nullspace.json

And updates the tex file with actual numbers, replacing "(see results)" placeholders.
"""

import os
import sys
import json
import re


def load_results_files(script_dir):
    """Load all three results JSON files."""
    results = {}

    files = {
        "global_svd": "results_global_svd.json",
        "per_neuron_svd": "results_per_neuron_svd.json",
        "structural": "results_structural_nullspace.json",
    }

    for key, filename in files.items():
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[key] = json.load(f)
                print(f"✓ Loaded {filename}")
        else:
            print(f"✗ Missing {filename}")

    return results


def format_results_table(results):
    """Format unified results table for tex."""
    # Extract noise-free results at 99% threshold
    global_svd = results.get("global_svd", {}).get("noise-free", {}).get("99.0%", {})
    per_neuron = results.get("per_neuron_svd", {}).get("noise-free", {}).get("99.0%", {})
    structural = results.get("structural", {}).get("noise-free", {})

    table_rows = []

    # Global SVD row
    if global_svd:
        table_rows.append(
            f"\\textbf{{Step 1:}} Global SVD & Population-level rank upper bound & "
            f"{global_svd.get('null_space_dim', '—')} & 434,112 & "
            f"\\textbf{{{global_svd.get('degree_of_degeneracy', '—')}\\%}} \\\\"
        )

    # Per-neuron SVD row
    if per_neuron:
        table_rows.append(
            f"\\textbf{{Step 2:}} Per-neuron SVD & Individual neuron rank measurement & "
            f"{per_neuron.get('null_space_dim', '—')} & 434,112 & "
            f"\\textbf{{{per_neuron.get('degree_of_degeneracy', '—')}\\%}} \\\\"
        )

    # Structural row
    if structural:
        null_dim = structural.get('null_space_dim', '—')
        deg = structural.get('degree_of_degeneracy', '—')
        table_rows.append(
            f"\\textbf{{Step 3:}} Structural per-type & Within-type correlation counting & "
            f"{null_dim} & 434,112 & "
            f"\\textbf{{{deg}\\%}} \\\\"
        )

    return table_rows


def update_tex_file(tex_path, results):
    """Update tex file with results from JSON files."""

    with open(tex_path, 'r') as f:
        content = f.read()

    # Update per-neuron SVD results table (noise-free at 99%)
    per_neuron = results.get("per_neuron_svd", {}).get("noise-free", {}).get("99.0%", {})
    if per_neuron:
        # Replace the per-neuron results row
        old_pattern = (
            r"99\\% & \\textit\{\(see results\)\} & \\textit\{\(see results\)\} & "
            r"\\textit\{\(see results\)\} & \\textit\{\(see results\)\} \\\\"
        )
        new_row = (
            f"99\\% & {per_neuron.get('mean_effective_rank', '—')} & "
            f"{per_neuron.get('null_space_dim', '—')} & "
            f"{per_neuron.get('degree_of_degeneracy', '—')}\\% & "
            f"{per_neuron.get('fully_identifiable_neurons', '—')} / 13,697 \\\\"
        )
        content = re.sub(old_pattern, new_row, content)
        print(f"✓ Updated per-neuron SVD results table")

    # Update unified results table (all three approaches)
    table_rows = format_results_table(results)
    if table_rows:
        # Find and replace the unified table body
        old_table_pattern = (
            r"Global SVD \(coarse bound\) & \$\\sim\$90,000 & \$\\sim\$21\\% \\\\\s*"
            r"Per-neuron SVD \(empirical\) & \\textit\{\(see results\)\} & \\textit\{\(see results\)\} \\\\\s*"
            r"Structural per-type count & \$\\sim\$121,100 & \$\\sim\$28\\% \\\\"
        )

        new_table_body = " \\\\\n".join(table_rows) + " \\\\"

        # Try to match and replace
        match = re.search(old_table_pattern, content)
        if match:
            content = content[:match.start()] + new_table_body + content[match.end():]
            print(f"✓ Updated unified results table")
        else:
            print(f"✗ Could not find unified results table pattern in tex file")

    # Write updated content
    with open(tex_path, 'w') as f:
        f.write(content)

    print(f"✓ Updated {tex_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, "../docs/degeneracy_analysis.tex")

    print("Loading results files...")
    results = load_results_files(script_dir)

    if not results:
        print("ERROR: No results files found. Run the three analysis scripts first.")
        sys.exit(1)

    print("\nUpdating tex file...")
    update_tex_file(tex_path, results)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
