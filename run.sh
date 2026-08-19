#!/bin/bash
set -e 

datasets=("higgs" "epsilon" "helena")

for data in "${datasets[@]}"; do
    echo ""
    echo "Starting experiment $data"
    python main.py --dataset $data --results best_params/$data
done

echo "all runs complete"