#!/bin/bash
# start_clients.sh - 全てのクライアントを遠隔で起動するスクリプト

# --- 設定項目 ---
CLIENT_HOSTS=("kappa" "iwashi" "saba" "aji")

# --- 実行処理 ---
CLIENT_ID=0
for host in "${CLIENT_HOSTS[@]}"; do
  echo "🚀 Starting client ${CLIENT_ID} on host ${host}..."

  # ホスト名に応じてプロジェクトパスを切り替える
  case ${host} in
    "kappa")
      PROJECT_PATH="~/Fed/Federated-Learning-PyTorch"
      ;;
    *)
      PROJECT_PATH="~/Federated-Learning-PyTorch"
      ;;
  esac

  # リモートで実行するコマンド
  COMMAND="cd ${PROJECT_PATH} && { source venv/bin/activate && python src/ctrain_https.py --client_id ${CLIENT_ID}; } > src/client_${CLIENT_ID}.log 2>&1 &"
  
  # ▼▼▼ このコマンド行を最終修正しました ▼▼▼
  # sshに "-f" オプションを追加して、即座にバックグラウンドに移行させる
  ssh -f "${host}" "${COMMAND}"
  
  if [ $? -eq 0 ]; then
    echo "✅ Client ${CLIENT_ID} start command sent successfully."
  else
    echo "❌ Failed to send start command to ${host}."
  fi
  
  CLIENT_ID=$((CLIENT_ID + 1))
done

echo "All client start commands have been sent."