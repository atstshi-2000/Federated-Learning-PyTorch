#!/bin/bash
# run.sh
percent=(0.9)
# percent=(0 0.7 0.8 0.9 1.0)
for ((i = 0; i < ${#percent[@]}; i++)); do
  echo "Testing on dataset and percent  ${percent[i]} ..."
  ./src/federated_main.py --model cnn --dataset cifar --lr 0.03 --seed 42 --el2n 5 --num_users 10 --local_ep 3 --local_bs 32 --verbose 0 --epoch 50 --pru_percent ${percent[i]} --gpu cuda:0
done