#!/bin/bash
# run.sh

DATASETS=(mnist fmnist cifar)

for ((i = 0; i < ${#DATASETS[@]}; i++)); do
  echo "Testing on dataset ${DATASETS[i]} ..."
  ./baseline_main.py --model cnn --dataset ${DATASETS[i]} --lr 0.001 --gpu cuda:0 --verbose 1 --seed 42
done

# ./baseline_main.py --model mlp --gpu cuda:0 --verbose 1 --seed 42

