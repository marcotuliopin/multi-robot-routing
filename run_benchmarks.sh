#!/bin/bash

MAP_DIR="benchmarks/chao"

for MAP in "$MAP_DIR"/p[4-5]*; do
    for ((i=1; i<=1; i++)); do
        echo "Running iteration $i on map $MAP..."
        python3 main.py \
            --map "$MAP" \
            --total_time "10000" \
            --num_iterations "500" \
            --algorithm 0 \
            --out "out"
    done
done