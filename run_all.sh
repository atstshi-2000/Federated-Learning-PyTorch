#!/bin/bash
# run_all.sh - 全ての実験を管理し、実験ごとにクライアントを再起動する最終完成版

# --- 実験パラメータ ---
client_num_per_client_values=(40000 50000)
el2n_values=(0)
prune_rate_values=(0.05)
local_ep_values=(3)

# --- ここから実験パラメータのループを開始 ---
for cid in "${client_num_per_client_values[@]}"; do
  for el in "${el2n_values[@]}"; do
    for rate in "${prune_rate_values[@]}"; do
      for lep in "${local_ep_values[@]}"; do
        
        echo "================================================================="
        echo "STARTING EXPERIMENT:"
        echo "  - num_per_client: ${cid}, el2n: ${el}, prune_rate: ${rate}, local_ep: ${lep}"
        echo "================================================================="

        # --- ステップ1: サーバーをバックグラウンドで起動 ---
        echo "▶️ Step 1: Starting server in the background..."
        python ./src/subpro_https.py \
          --model cnn --dataset cifar --lr 0.03 --seed 42 \
          --el2n "${el}" --num_users 4 --local_ep "${lep}" \
          --local_bs 32 --verbose 0 --epochs 50 --gpu 0 \
          --num_per_client "${cid}" --prune_rate "${rate}" \
          --iid 0 --alpha 0.5 --unequal 0 &
        
        PYTHON_PID=$!
        echo "Server process for this experiment started with PID: ${PYTHON_PID}"

        # --- ステップ2: サーバーが起動するのを少し待つ ---
        echo "⏳ Step 2: Waiting 15 seconds for the server to initialize..."
        sleep 15

        # --- ステップ3: この実験のためにクライアントを起動 ---
        echo "🚀 Step 3: Starting all clients for this experiment..."
        bash start_clients.sh

        # --- ステップ4: Pythonサーバープロセスが終了するのを待つ ---
        echo "📊 Experiment is running. Monitoring server process..."
        wait $PYTHON_PID

        # --- ステップ5: この実験の全プロセスをクリーンアップ ---
        echo "🛑 Step 5: Experiment finished. Stopping all clients and server..."
        bash stop_all.sh
        
        echo "Waiting 10 seconds before next run..."
        sleep 10
        
      done
    done
  done
done

echo "🎉 All experiment series completed."