#!/bin/bash

# Run all three degeneracy analysis approaches and update tex file
# Usage: bash run.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="neural-graph-linux"

echo "================================================================================"
echo "DEGENERACY ANALYSIS - THREE APPROACHES"
echo "================================================================================"
echo ""
echo "Environment: $CONDA_ENV"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda environment exists
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo -e "${YELLOW}ERROR: Conda environment '$CONDA_ENV' not found${NC}"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo -e "${BLUE}Step 1: Global SVD approach${NC}"
echo "=========================================="
cd "$SCRIPT_DIR"
if conda run -n "$CONDA_ENV" python global_svd_null_space.py; then
    echo -e "${GREEN}✓ Global SVD analysis completed${NC}"
    echo ""
else
    echo -e "${YELLOW}✗ Global SVD analysis failed${NC}"
    exit 1
fi

echo -e "${BLUE}Step 2: Per-neuron SVD approach${NC}"
echo "=========================================="
cd "$SCRIPT_DIR"
if conda run -n "$CONDA_ENV" python per_neuron_null_space_svd.py; then
    echo -e "${GREEN}✓ Per-neuron SVD analysis completed${NC}"
    echo ""
else
    echo -e "${YELLOW}✗ Per-neuron SVD analysis failed${NC}"
    exit 1
fi

echo -e "${BLUE}Step 3: Structural per-type analysis${NC}"
echo "=========================================="
cd "$SCRIPT_DIR"
if conda run -n "$CONDA_ENV" python structural_per_type_nullspace.py; then
    echo -e "${GREEN}✓ Structural per-type analysis completed${NC}"
    echo ""
else
    echo -e "${YELLOW}✗ Structural per-type analysis failed${NC}"
    exit 1
fi

# Check if all JSON results files were created
echo -e "${BLUE}Verifying results files${NC}"
echo "=========================================="
RESULTS_FILES=(
    "results_global_svd.json"
    "results_per_neuron_svd.json"
    "results_structural_nullspace.json"
)

for file in "${RESULTS_FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${YELLOW}✗${NC} $file (missing)"
    fi
done
echo ""

# Update tex file with results
echo -e "${BLUE}Step 4: Auto-updating tex file${NC}"
echo "=========================================="
cd "$SCRIPT_DIR"
if python update_degeneracy_tex.py; then
    echo -e "${GREEN}✓ Tex file updated${NC}"
    echo ""
else
    echo -e "${YELLOW}✗ Tex file update failed${NC}"
    exit 1
fi

# Summary
echo "================================================================================"
echo -e "${GREEN}SUCCESS: All degeneracy analyses completed!${NC}"
echo "================================================================================"
echo ""
echo "Generated results files:"
echo "  - $SCRIPT_DIR/results_global_svd.json"
echo "  - $SCRIPT_DIR/results_per_neuron_svd.json"
echo "  - $SCRIPT_DIR/results_structural_nullspace.json"
echo ""
echo "Updated documentation:"
echo "  - docs/degeneracy_analysis.tex"
echo ""
echo "Generated visualizations:"
echo "  - ./svg_global_svd_plots/"
echo "  - ./svg_per_neuron_plots/"
echo "  - ./svg_structural_plots/"
echo ""
echo "Next steps:"
echo "  1. Review updated tex file: git diff docs/degeneracy_analysis.tex"
echo "  2. Commit changes: git add . && git commit -m 'Add degeneracy analysis results'"
echo "  3. Push to remote: git push"
echo ""
