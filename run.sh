#!/bin/bash
set -e 

intialseeds=(42 0 1 123 240)

for num in "${intialseeds[@]}"; do
    echo ""
    echo "Starting experiment $num"
    python tune.py --dataset covertype --seed_value $num
done

echo "all runs complete"