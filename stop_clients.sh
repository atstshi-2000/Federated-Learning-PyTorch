#!/bin/bash
# stop_clients.sh - 全てのクライアントを遠隔で停止するスクリプト

# --- 設定項目 (start_clients.shと合わせてください) ---
# ▼▼▼ ここをIPアドレスからホスト名に変更 ▼▼▼
CLIENT_HOSTS=("kappa" "aji" "saba" "iwashi")

# --- 実行処理 ---
for host in "${CLIENT_HOSTS[@]}"; do
  echo "🛑 Stopping clients on host ${host}..."
  
  # ▼▼▼ プロセス名を ctrain_https.py に修正 & ユーザー名の指定を削除 ▼▼▼
  # configファイルでUserが指定されているので、ここでの指定は不要
  ssh "${host}" "pkill -f ctrain_https.py"
  
  echo "✅ Stop command sent to ${host}."
done

echo "All client stop commands have been sent."