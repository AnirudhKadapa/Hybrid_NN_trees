#!/bin/bash
set -e 

datasets=("epsilon" "helena" "adult")

for data in "${datasets[@]}"; do
    echo ""
    echo "Starting experiment $data"
    python main.py --dataset $data --results best_params/$data
done

echo "all runs complete"