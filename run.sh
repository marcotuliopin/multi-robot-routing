#!/bin/bash

N=$1
MAP="benchmarks/chao/p1.2.r.txt"
TOTAL_TIME=600
ITERATIONS=100
SPEEDS="2 1 1"
BUDGET="70 70 70"

while getopts "n:m:a:t:s:b:i:" opt; do
  case $opt in
    n) N=$OPTARG ;;
    m) MAP=$OPTARG ;;
    t) TOTAL_TIME=$OPTARG ;;
    s) SPEEDS=$OPTARG ;;
    b) BUDGET=$OPTARG ;;
    i) ITERATIONS=$OPTARG ;;
    \?) echo "Uso: $0 [-n runs] [-m map] [-a agents] [-t time_limit] [-s speeds] [-b budgets] [-i iterations]" 
        exit 1 ;;
  esac
done

if [[ -z "$N" || ! "$N" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <number_of_times>"
    exit 1
fi

for ((i=1; i<=$N; i++)); do
    echo "Running iteration $i on map $MAP..."
    python3 main.py \
        --map "$MAP" \
        --total_time "$TOTAL_TIME" \
        --num_iterations "$ITERATIONS" \
        --speeds $SPEEDS
        # --budget $BUDGET \
done