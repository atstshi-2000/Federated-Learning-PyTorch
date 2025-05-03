#!/bin/bash
# run.sh

THRESHOLDS=(0.3,0.4,0.5)
for ((i = 0; i < ${#threshold[@]}; i++)); do
    echo "Testing on threshold ${threshold[i]} ..."
    ./src/federated_main.py --model cnn --dataset cifar --lr 0.03 --seed 42 --el2n 1 --num_users 10 --local_ep 3 --local_bs 32 --verbose 0 --gpu cuda:0 --epoch 50 --threshold ${THRESHOLDS[i]}
done
# ./baseline_main.py --model mlp --gpu cuda:0 --verbose 1 --seed 42
# for ((i = 0; i < ${#DATASETS[@]}; i++)); do
#   echo "Testing on dataset ${DATASETS[i]} ..."
#   ./baseline_main.py --model cnn --dataset ${DATASETS[i]} --lr 0.001 --gpu cuda:0 --verbose 1 --seed 42
# done

