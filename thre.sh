#!/bin/bash
# run.sh
client=(50000 40000 30000 20000 10000)
# client=(100)
for cid in "${client[@]}"; do
  echo "Testing on dataset and client  ${cid} ..."
  ./src/federated_main.py --model cnn --dataset cifar --lr 0.03 --seed 42 --el2n 0 --num_users 10 --local_ep 3 --local_bs 32 --verbose 0 --epoch 50 --gpu cuda:0 --num_per_client ${cid}
done

# percent=(0.9)
# # percent=(0 0.7 0.8 0.9 1.0)
# for ((i = 0; i < ${#percent[@]}; i++)); do
#   echo "Testing on dataset and percent  ${percent[i]} ..."
#   ./src/federated_main.py --model cnn --dataset cifar --lr 0.03 --seed 42 --el2n 0 --num_users 10 --local_ep 3 --local_bs 32 --verbose 0 --epoch 50 --pru_percent ${percent[i]} --gpu cuda:0
# done