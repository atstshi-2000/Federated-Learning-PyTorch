#!/bin/bash
# run.sh
threshold=(0.02 0.025 0.045)
for ((i = 0; i < ${#threshold[@]}; i++)); do
  echo "Testing on dataset ${threshold[i]} ..."
  ./src/federated_main.py --model cnn --dataset cifar --lr 0.03 --seed 42 --el2n 1 --num_users 10 --local_ep 3 --local_bs 32 --verbose 0 --epoch 50 --threshold ${threshold[i]
  
