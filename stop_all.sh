#!/bin/bash
# stop_all.sh - 全ての関連プロセスを完全に停止するスクリプト

echo "🛑 Stopping all related processes..."

# --- 1. 全てのクライアントを停止 ---
echo "--> Stopping remote clients..."
bash stop_clients.sh

# --- 2. サーバープロセス(作業員)を停止 ---
# "subpro_https.py"という名前が含まれるプロセスを全て停止
echo "--> Stopping python server process (subpro_https.py)..."
pkill -f subpro_https.py

# --- 3. サーバー管理者プロセス(親方)を停止 ---
# "run_server.sh"という名前が含まれるプロセスを全て停止
echo "--> Stopping server management script (run_server.sh)..."
pkill -f run_server.sh

# --- 4. (念のため) 大親方も停止 ---
echo "--> Stopping main script (run_all.sh)..."
pkill -f run_all.sh

echo "✅ All processes should be stopped."