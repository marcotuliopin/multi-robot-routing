#!/bin/bash

if [ -d "./out" ]; then
    echo "Removing ./out directory..."
    rm -rf ./out
fi

if [ ! -d "./out" ]; then
    echo "Creating ./out directory..."
    mkdir -p ./out
fi

if [ -d "./tests" ]; then
    echo "Removing ./tests directory..."
    rm -rf ./tests
fi

if [ ! -d "./tests" ]; then
    echo "Creating ./tests directory..."
    mkdir -p ./tests
fi

if [ -d "./imgs" ]; then
    echo "Removing ./imgs directory..."
    rm -rf ./imgs
fi

if [ ! -d "./imgs" ]; then
    echo "Creating ./imgs directory and subdirectories..."
    mkdir -p ./imgs/paths
    mkdir -p ./imgs/animations
fi