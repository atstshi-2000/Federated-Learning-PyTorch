#!/bin/bash
# run.sh

#!/bin/bash

client=(10000)
el2n=(5)
prune_rates=(0.05)
local_ep_values=(3 3 3 3 3) # <-- 3のみで実行するように変更

for cid in "${client[@]}"; do
  for el in "${el2n[@]}"; do
    if [ "$el" -eq 0 ]; then
      # el2n=0の場合はprune_rateに関係なく実行
      for lep in "${local_ep_values[@]}"; do
        echo "Testing on dataset with client ${cid}, el2n ${el}, local_ep ${lep} ..."
        ./src/subprocess_ccs.py \
          --model cnn \
          --dataset cifar \
          --lr 0.03 \
          --seed 42 \
          --el2n "${el}" \
          --num_users 10 \
          --local_ep "${lep}" \
          --local_bs 32 \
          --verbose 0 \
          --epoch 50 \
          --gpu cuda:0 \
          --num_per_client "${cid}" \
          --prune_rate 0 \
          --iid 0 \
          --unequal 0
          done
      else
      # el2n!=0の場合はprune_ratesの値の回数分実行
      for rate in "${prune_rates[@]}"; do
        for lep in "${local_ep_values[@]}"; do
          echo "Testing on dataset with client ${cid}, prune_rate ${rate}, el2n ${el}, local_ep ${lep} ..."
          ./src/subprocess_ccs.py \
            --model cnn \
            --dataset cifar \
            --lr 0.03 \
            --seed 42 \
            --el2n "${el}" \
            --num_users 10 \
            --local_ep "${lep}" \
            --local_bs 32 \
            --verbose 0 \
            --epoch 50 \
            --gpu cuda:0 \
            --num_per_client "${cid}" \
            --prune_rate "${rate}" \
            --iid 0 \
            --unequal 0
        done
      done
    fi
  done
done


# client=(10000 20000 30000 40000 50000)
# # client=(2000)
# for cid in "${client[@]}"; do
#   echo "Testing on dataset and client  ${cid} ..."
#   ./src/new_ccs_lim.py --model cnn --dataset cifar --lr 0.03 --seed 42 --el2n 5 --num_users 10 --local_ep 3 --local_bs 32 --verbose 0 --epoch 50 --gpu cuda:0 --num_per_client ${cid} --prune_rate 0.1
# done
# percent=(0 0.7 0.8 0.9 1.0)
# for ((i = 0; i < ${#percent[@]}; i++)); do
#   echo "Testing on dataset and percent  ${percent[i]} ..."
#   ./src/new_ccs.py --model cnn --dataset cifar --lr 0.03 --seed 42 --el2n 5 --num_users 10 --local_ep 3 --local_bs 32 --verbose 0 --epoch 50 --pru_percent ${percent[i]} --gpu cuda:0 --num_per_client 1000
# done

#!/bin/bash
# run.sh

# client=(40000 50000)
# # prune_rates=(0.05)
# el2n=(0 5)
# prune_rates=(0.05 0.1 0.15 0.2)

# for cid in "${client[@]}"; do
#   for rate in "${prune_rates[@]}"; do
#     for el in "${el2n[@]}"; do
#       echo "Testing on dataset with client ${cid}, prune_rate ${rate}, and el2n ${el} ..."
#       ./src/subprocess_ccs.py \
#         --model cnn \
#         --dataset cifar \
#         --lr 0.03 \
#         --seed 42 \
#         --el2n "${el}" \
#         --num_users 10 \
#         --local_ep 3 \
#         --local_bs 32 \
#         --verbose 0 \
#         --epoch 50 \
#         --gpu cuda:0 \
#         --num_per_client "${cid}" \
#         --prune_rate "${rate}"
#     done    
#   done
# done
#!/bin/bash

# client=(10000 20000 30000 40000 50000)
# el2n=(0 5)
# prune_rates=(0.05 0.1 0.15)
# local_ep_values=(3) # <-- local_epの値を配列として定義

# for cid in "${client[@]}"; do
#   for el in "${el2n[@]}"; do
#     if [ "$el" -eq 0 ]; then
#       # el2n=0の場合はprune_rateに関係なく実行
#       for lep in "${local_ep_values[@]}"; do # <-- local_epのループを追加
#         echo "Testing on dataset with client ${cid}, el2n ${el}, local_ep ${lep} ..."
#         ./src/subprocess_ccs.py \
#           --model cnn \
#           --dataset cifar \
#           --lr 0.03 \
#           --seed 42 \
#           --el2n "${el}" \
#           --num_users 10 \
#           --local_ep "${lep}" \
#           --local_bs 32 \
#           --verbose 0 \
#           --epoch 50 \
#           --gpu cuda:0 \
#           --num_per_client "${cid}" \
#           --prune_rate 0
#       done
#     else
#       # el2n!=0の場合はprune_ratesの値の回数分実行
#       for rate in "${prune_rates[@]}"; do
#         for lep in "${local_ep_values[@]}"; do # <-- local_epのループを追加
#           echo "Testing on dataset with client ${cid}, prune_rate ${rate}, el2n ${el}, local_ep ${lep} ..."
#           ./src/subprocess_ccs.py \
#             --model cnn \
#             --dataset cifar \
#             --lr 0.03 \
#             --seed 42 \
#             --el2n "${el}" \
#             --num_users 10 \
#             --local_ep "${lep}" \
#             --local_bs 32 \
#             --verbose 0 \
#             --epoch 50 \
#             --gpu cuda:0 \
#             --num_per_client "${cid}" \
#             --prune_rate "${rate}"
#           done
#         done
#       fi
#     done
#   done
# done

# client=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 40000 50000)
# el2n=(0 5)
# prune_rates=(0.05 0.1 0.15)

# for cid in "${client[@]}"; do
#   for el in "${el2n[@]}"; do
#     if [ "$el" -eq 0 ]; then
#       # el2n=0の場合はprune_rateに関係なく1回のみ実行
#       echo "Testing on dataset with client ${cid}, el2n ${el} ..."
#       ./src/subprocess_ccs.py \
#         --model cnn \
#         --dataset cifar \
#         --lr 0.03 \
#         --seed 42 \
#         --el2n "${el}" \
#         --num_users 10 \
#         --local_ep 2 \
#         --local_bs 32 \
#         --verbose 0 \
#         --epoch 50 \
#         --gpu cuda:0 \
#         --num_per_client "${cid}" \
#         --prune_rate 0
#     else
#       # el2n!=0の場合はprune_ratesの値の回数分実行
#       for rate in "${prune_rates[@]}"; do
#         echo "Testing on dataset with client ${cid}, prune_rate ${rate}, and el2n ${el} ..."
#         ./src/subprocess_ccs.py \
#           --model cnn \
#           --dataset cifar \
#           --lr 0.03 \
#           --seed 42 \
#           --el2n "${el}" \
#           --num_users 10 \
#           --local_ep 2 \
#           --local_bs 32 \
#           --verbose 0 \
#           --epoch 50 \
#           --gpu cuda:0 \
#           --num_per_client "${cid}" \
#           --prune_rate "${rate}"
#       done
#     fi
#   done
# done