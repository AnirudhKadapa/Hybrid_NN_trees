#!/bin/bash

datasets=("higgs" "epsilon" "helena")

for data in "${datasets[@]}"; do
    python main.py --dataset $data --results best_params/$data
done

echo "all runs complete"