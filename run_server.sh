#!/bin/bash
# run_server.sh - サーバーを監視・管理しながら実験ループを実行する最終版スクリプト

# --- 実験パラメータ ---
client_num_per_client_values=(30000 40000 50000)
el2n_values=(0)
prune_rate_values=(0.05)
local_ep_values=(3)

# --- ループ処理 ---
for cid in "${client_num_per_client_values[@]}"; do
  for el in "${el2n_values[@]}"; do
    for rate in "${prune_rate_values[@]}"; do
      for lep in "${local_ep_values[@]}"; do
        
        echo "================================================================="
        echo "STARTING EXPERIMENT:"
        echo "  - num_per_client: ${cid}, el2n: ${el}, prune_rate: ${rate}, local_ep: ${lep}"
        echo "================================================================="

        # --- サーバー起動と監視 ---
        # 以前の完了ファイルを削除
        rm -f ./src/SERVER_DONE.lock

        # サーバー(subpro_https.py)をバックグラウンドで起動
        python ./src/subpro_https.py \
          --model cnn --dataset cifar --lr 0.03 --seed 42 \
          --el2n "${el}" --num_users 4 --local_ep "${lep}" \
          --local_bs 32 --verbose 0 --epochs 50 --gpu 0 \
          --num_per_client "${cid}" --prune_rate "${rate}" \
          --iid 0 --alpha 0.5 --unequal 0 &
        
        PYTHON_PID=$!
        echo "Server process for this experiment started with PID: ${PYTHON_PID}"

        # 完了ファイルができるまで、またはプロセスが死ぬまで監視
        echo "📊 Experiment is running. Monitoring for completion signal..."
        while [ ! -f ./src/SERVER_DONE.lock ]; do
            # プロセスが予期せず終了していないか確認
            if ! ps -p $PYTHON_PID > /dev/null; then
                echo "⚠️ Server process (PID: ${PYTHON_PID}) disappeared unexpectedly."
                break
            fi
            sleep 5 # 5秒ごとに確認
        done

        # --- クリーンアップ ---
        echo "✅ Completion signal received or process ended. Cleaning up..."
        # ハングしている可能性のあるサーバープロセスを確実に強制終了
        kill -9 $PYTHON_PID
        # 完了ファイルを削除
        rm -f ./src/SERVER_DONE.lock
        
        echo "Waiting 10 seconds before next run..."
        sleep 10
        
      done
    done
  done
done

echo "🎉 All experiments completed!"