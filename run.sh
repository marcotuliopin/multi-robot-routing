#!/bin/bash

N=$1
MAP=${2:-"maps/paper_example.txt"}
NUM_AGENTS=${3:-3}
TOTAL_TIME=${4:-600}
SPEEDS=${5:-"1 1 1"}
BUDGET=${6:-"70 70 70"}

if [[ -z "$N" || ! "$N" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <number_of_times>"
    exit 1
fi

for ((i=1; i<=$1; i++)); do
    echo "Running iteration $i..."
    python3 main.py \
        --map "$MAP" \
        --num-agents "$NUM_AGENTS" \
        --total_time "$TOTAL_TIME" \
        --speeds $SPEEDS \
        --budget $BUDGET
done