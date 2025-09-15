#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import sys
import pickle
import numpy as np
from tqdm import tqdm
import json
import psutil
from collections import defaultdict
import slackweb
import threading
import matplotlib
import matplotlib.pyplot as plt
import gzip
import io
import signal

import torch
import torchvision
from flask import Flask, request, send_from_directory, jsonify

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from ccs_utils import IncrementalCCS
import requests
# ===================================================================
# === APIサーバー関連のコード (変更なし) ===
# ===================================================================
app = Flask(__name__)
base_dir = os.path.dirname(__file__)
lock = threading.Lock()
server_state = {
    "status": "initializing",
    "current_epoch": -1,
    "selected_clients": set(),
    "completed_clients_this_round": set(),
    "checked_in_clients": set(),  # ← この行を追加
    "args": None
}
# subpro_https.py のAPI定義セクションに追加

@app.route('/check_in/<int:client_id>', methods=['POST'])
def check_in(client_id):
    """クライアントからの起動報告を受け付けるAPI"""
    with lock:
        server_state["checked_in_clients"].add(client_id)
        num_checked_in = len(server_state["checked_in_clients"])
        num_total = server_state["args"].num_users if server_state["args"] else 'N/A'
    
    print(f"[ヘルスチェック] クライアント {client_id} が起動しました。(現在 {num_checked_in}/{num_total} 台)")
    return jsonify({"status": "checked in"}), 200
@app.route('/files/<path:filepath>')
def download_file(filepath):
    try: return send_from_directory(base_dir, filepath, as_attachment=True)
    except FileNotFoundError: return "File not found", 404

@app.route('/get_config', methods=['GET'])
def get_config():
    with lock:
        if server_state["args"]: return jsonify(vars(server_state["args"]))
        else: return jsonify({"error": "Server is not ready yet"}), 503
@app.route('/get_task/<int:client_id>', methods=['GET'])
def get_task(client_id):
    with lock:
        status = server_state["status"]
        current_epoch = server_state["current_epoch"]
        is_selected = client_id in server_state["selected_clients"]
        is_completed = client_id in server_state["completed_clients_this_round"]
    if status == "finished": return jsonify({"task": "shutdown"})
    if status == "running" and is_selected and not is_completed:
        return jsonify({"task": "train", "epoch": current_epoch})
    else: return jsonify({"task": "wait", "retry_after": 30})

@app.route('/upload/<int:client_id>/<int:epoch>', methods=['POST'])
def upload_files(client_id, epoch):
    """クライアントが学習結果をアップロードするためのAPI"""
    try:
        compressed_model_file = request.files['model']
        metrics_file = request.files['metrics']
        
        # 受信したモデルデータをメモリ上でGZIP解凍
        decompressed_model_data = gzip.decompress(compressed_model_file.read())
        
        # メモリ上のバイトデータからモデルの重みをロード
        buffer = io.BytesIO(decompressed_model_data)
        buffer.seek(0)
        local_state_dict = torch.load(buffer)

        # メトリクスはJSONとしてロード
        metrics = json.loads(metrics_file.read().decode('utf-8'))
        
        # (この後の処理は後続のステップで使うため、一時的にグローバル変数などに保存するなどの工夫が必要ですが、
        #  まずは通信部分の修正として)
        # ここでは受け取ったデータをファイルに保存する
        pth_dir = os.path.join(base_dir, "PTH")
        json_dir = os.path.join(base_dir, "JSON")
        torch.save(local_state_dict, os.path.join(pth_dir, f"client_{client_id}_round{epoch}_model.pth"))
        with open(os.path.join(json_dir, f"metrics_{client_id}_round{epoch}.json"), "w") as f:
            json.dump(metrics, f)

        with lock:
            server_state["completed_clients_this_round"].add(client_id)
            
        print(f"\n[API] Client {client_id} (Round {epoch}) から圧縮結果を受信・解凍しました。")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"\n[API ERROR] Client {client_id} からのアップロードに失敗: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    # ▼▼▼【追加】サーバーをスレッドから安全に停止させるためのAPI▼▼▼
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        # 開発サーバー以外で実行されている場合は、この方法は使えない
        # 代わりにプロセスを終了させることで対応
        print("Shutdown function not found, exiting process.")
        os._exit(0)
    func()

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

# ===================================================================
# === 元のプログラムのヘルパー関数 (変更なし) ===
# ===================================================================
SLACKURL = 'https://hooks.slack.com/services/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'
def slackPost(message):
    try:
        slack = slackweb.Slack(url=SLACKURL)
        slack.notify(text=message)
    except Exception as e: print(f"Slack notification failed: {e}")
def compute_keep_ratios(exec_times, cpu_usages, mem_usages, **kwargs):
    cids = list(exec_times.keys()); n = len(cids)
    if n == 0: return {}
    def normalize(x):
        mn, mx = x.min(), x.max()
        return np.ones_like(x) / n if (mx - mn) < 1e-8 else (x - mn) / (mx - mn)
    inv_time = normalize(1.0 / (np.array([exec_times.get(c, 0) for c in cids]) + 1e-8))
    inv_cpu = normalize(1.0 / (np.array([cpu_usages.get(c, 0) for c in cids]) + 1e-8))
    inv_mem = normalize(1.0 / (np.array([mem_usages.get(c, 0) for c in cids]) + 1e-8))
    mixed = (kwargs.get('alpha_time', 0.4) * inv_time + kwargs.get('alpha_cpu', 0.3) * inv_cpu + kwargs.get('alpha_mem', 0.3) * inv_mem)
    mixed_sum = mixed.sum()
    normed = mixed / mixed_sum if mixed_sum > 1e-8 else np.ones_like(mixed) / n
    min_r, max_r = kwargs.get('min_ratio', 0.2), kwargs.get('max_ratio', 0.9)
    scaled = min_r + (max_r - min_r) * normed
    scaled_sum = scaled.sum()
    final = scaled / scaled_sum if scaled_sum > 1e-8 else np.ones_like(scaled) / n
    return {cid: float(final[i]) for i, cid in enumerate(cids)}
def log_metrics(round_num, train_time, model, accuracies, log_path):
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    model_size = len(pickle.dumps(model.state_dict())) / 1024
    accuracy_mean = np.mean(accuracies) if accuracies else 0.0
    accuracy_std = np.std(accuracies) if accuracies else 0.0
    log_data = {"round": round_num, "train_time_sec": train_time, "memory_MB": memory_usage, "comm_KB": model_size, "accuracy_mean": accuracy_mean, "accuracy_std": accuracy_std}
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, f"round_{round_num}.json"), "w") as f: json.dump(log_data, f, indent=4)
    return memory_usage, model_size

# ===================================================================
# === メインの学習処理 (ここから) ===
# ===================================================================
def training_loop(args):
    with lock: server_state["args"] = args
    
    # 1. 初期設定
    exp_details(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    print(f"Using device: {device}")
    
    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.model == 'cnn':
        if args.dataset == 'cifar': global_model = torchvision.models.resnet18(num_classes=args.num_classes)
        elif args.dataset == 'mnist': global_model = CNNMnist(args=args)
        else: global_model = CNNFashion_Mnist(args=args)
    elif args.model == 'mlp':
        len_in = int(np.prod(train_dataset[0][0].shape))
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else: exit('Error: unrecognized model')
    global_model.to(device)
    
    # ▼▼▼【修正】ここからサーバーの起動・停止をループの外に移動▼▼▼
    # --- 2. APIサーバーをバックグラウンドで起動 ---
    server_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False))
    server_thread.daemon = True
    server_thread.start()
    print("\nAPI server started for the duration of this experiment.")
    time.sleep(5) # サーバーが完全に起動するのを待つ
    
    # 2. ディレクトリと変数の準備
    pruned_dir = os.path.join(base_dir, "pruned_data")
    json_dir = os.path.join(base_dir, "JSON")
    pth_dir = os.path.join(base_dir, "PTH")
    os.makedirs(pruned_dir, exist_ok=True); os.makedirs(json_dir, exist_ok=True); os.makedirs(pth_dir, exist_ok=True)

    train_loss, train_accuracy = [], []
    ccs_times = {cid: 0.0 for cid in range(args.num_users)}
    train_times = {cid: 0.0 for cid in range(args.num_users)}
    cpu_usages = {cid: 0.0 for cid in range(args.num_users)}
    mem_usages = {cid: 0.0 for cid in range(args.num_users)}
    mem_history, size_history = [], []
    last_client_keep = defaultdict(list)
    i_ccs = IncrementalCCS(num_groups=100, seed=args.seed)
    local_models = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=None, client_id=idx) for idx in range(args.num_users)]

    # 3. Round 0 用データ作成
    for cid in range(args.num_users):
        full_idxs = local_models[cid].idxs_train
        pruned_data = { "idxs": full_idxs, "data": [train_dataset[i][0] for i in full_idxs], "labels": [train_dataset[i][1] for i in full_idxs] }
        torch.save(pruned_data, os.path.join(pruned_dir, f"client_{cid}_round0.pt"))
# ▼▼▼【追加】ここから起動確認のヘルスチェック▼▼▼
    print("\n" + "="*50)
    print("全クライアントの起動確認中... (タイムアウト: 60秒)")
    health_check_start_time = time.time()
    all_clients_checked_in = False

    while time.time() - health_check_start_time < 60: # 60秒
        with lock:
            if len(server_state["checked_in_clients"]) == args.num_users:
                all_clients_checked_in = True
                break
        time.sleep(5)

    if all_clients_checked_in:
        print("✅ 全てのクライアントの起動を確認しました。学習を開始します。")
    else:
        with lock:
            all_client_ids = set(range(args.num_users))
            missing_clients = all_client_ids - server_state["checked_in_clients"]
            print(f"❌ タイムアウトしました。以下のクライアントが未接続です: {sorted(list(missing_clients))}")
            print("実験を中止します。")
        
        # プログラムを安全に終了させる
        with lock:
            server_state["status"] = "finished"
        return # training_loopスレッドを終了
    print("="*50 + "\n")
    # ▲▲▲【追加】ここまで起動確認のヘルスチェック▲▲▲
    # 4. メイン学習ループ
    with lock: server_state["status"] = "running"
    
    start_time = time.time()
    for epoch in tqdm(range(args.epochs), desc="Global Rounds"):
        global_model.train()
        with lock:
            server_state["current_epoch"] = epoch
            server_state["completed_clients_this_round"].clear()
        
        torch.save(global_model.state_dict(), os.path.join(pth_dir, f"global_model_round{epoch}.pth"))
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        with lock: server_state["selected_clients"] = set(idxs_users)
        
        # ▼▼▼【移植】ここからEL2N/CCSプルーニングのロジック▼▼▼
        start_prune_round = getattr(args, 'start_prune_round', 10)
        prune_interval = getattr(args, 'prune_interval', 10)
        do_prune = (args.el2n != 0 and epoch >= (start_prune_round - 1) and (epoch - (start_prune_round - 1)) % prune_interval == 0)

        if do_prune:
            t_prune_op_start = time.monotonic()
            client_keep = defaultdict(list)

            if args.el2n == 5: # 提案手法 (EL2N-CCS)
                print(f"\n | [Round {epoch+1}] Performing Pruning using Proposed Method (EL2N-CCS)... |")
                all_scores, global_to_local = [], []
                for cid in range(args.num_users):
                    if len(local_models[cid].trainloader.dataset) > 0:
                        scores_np = local_models[cid].compute_el2n_scores_4(global_model).detach().cpu().numpy()
                        all_scores.append(scores_np)
                        for li in range(len(scores_np)): global_to_local.append((cid, li))
                    else:
                        print(f"[Info] Client {cid} has no data, skipping score calculation.")
                
                # ▼▼▼【修正】スコアが1件も集まらなかった場合の安全装置▼▼▼
                if not all_scores:
                    print("⚠️ No client has data to compute scores. Skipping pruning for this round.")
                    client_keep = defaultdict(list) # 空のままにする
                else:
                    all_scores = np.concatenate(all_scores, axis=0)
                    prune_level = (epoch - (start_prune_round - 1)) // prune_interval + 1
                    reduction_rate = min(args.prune_rate * prune_level, 0.8) 
                    num_to_keep = int(len(all_scores) * (1 - reduction_rate))
                    
                    keep_ratio_per_client = compute_keep_ratios(exec_times=train_times, cpu_usages=cpu_usages, mem_usages=mem_usages)
                    client_keep = i_ccs.update_and_select(scores=all_scores, num_to_keep=num_to_keep, global_to_local=global_to_local, keep_ratio_per_client=keep_ratio_per_client)
            
            # (ここに el2n == 1 のランダムプルーニングのロジックなども必要に応じて追加)

            if client_keep:
                last_client_keep = client_keep.copy()
                for cid in range(args.num_users):
                    keep_idx = client_keep.get(cid, [])
                    local_models[cid].update_dataset_sub(keep_idx, el2n=args.el2n)
                    global_keep_idxs = local_models[cid].pruned_idxs
                    pruned_data = {'idxs': global_keep_idxs, 'data': [train_dataset[i][0] for i in global_keep_idxs], 'labels': [train_dataset[i][1] for i in global_keep_idxs]}
                    torch.save(pruned_data, os.path.join(pruned_dir, f"client_{cid}_round{epoch}.pt"))
            
            pruning_time = time.monotonic() - t_prune_op_start
            for cid in range(args.num_users): ccs_times[cid] += pruning_time
        # ▲▲▲【移植】ここまでEL2N/CCSプルーニングのロジック▲▲▲

        print(f"\n[Round {epoch+1}/{args.epochs}] クライアント {list(idxs_users)} の学習を待機中...")
        
        wait_start_time = time.time()
        while True:
            time.sleep(5)
            with lock:
                if server_state["selected_clients"].issubset(server_state["completed_clients_this_round"]):
                    print(f"[Round {epoch+1}] 選択された全クライアントから結果を受信しました。")
                    break
            if time.time() - wait_start_time > 1800:
                print(f"!! [Round {epoch+1}] タイムアウトしました。")
                break
        # ▼▼▼【ここに追加】▼▼▼
        # 完了したクライアントのIDセットをリストに変換して active_clients を定義します
        active_clients = list(server_state["completed_clients_this_round"])

        # 5. 結果の読み込みと集約
        local_weights, local_losses = [], []
        for cid in active_clients:
            metrics_file = os.path.join(json_dir, f"metrics_{cid}_round{epoch}.json")
            model_file = os.path.join(pth_dir, f"client_{cid}_round{epoch}_model.pth")
            if os.path.exists(metrics_file) and os.path.exists(model_file):
                with open(metrics_file, "r") as f: info = json.load(f)
                cpu_usages[cid] = info["avg_cpu"]
                mem_usages[cid] = info["avg_mem"]
                train_times[cid] += info["train_time"]
                local_losses.append(info["loss"])
                local_weights.append(torch.load(model_file, map_location=device))
        
        if len(local_weights) > 0:
            # ▼▼▼【デバッグ用】集約前のクライアントの重みを出力▼▼▼
            if active_clients:
                print(f"[デバッグ] Client {active_clients[0]} の重みの一部: {local_weights[0]['conv1.weight'][0,0,0,:5]}")
            
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)

            # ▼▼▼【デバッグ用】集約後のグローバルモデルの重みを出力▼▼▼
            print(f"[デバッグ] 集約後のGlobal Modelの重みの一部: {global_model.state_dict()['conv1.weight'][0,0,0,:5]}")

        # 6. 評価
        global_model.eval()
        clients_for_eval = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=None, client_id=idx) for idx in range(args.num_users)]
        list_acc = [clients_for_eval[c].inference(model=global_model)[0] for c in range(args.num_users)]
        train_accuracy.append(sum(list_acc)/len(list_acc))
        print(f'Train Accuracy: {100*train_accuracy[-1]:.2f}%')
        
        mem_mb, model_kb = log_metrics(epoch + 1, time.time() - start_time, global_model, list_acc, json_dir)
        mem_history.append(mem_mb); size_history.append(model_kb)

    # 7. 最終処理 (ループ終了後)
    with lock: server_state["status"] = "finished"
    
    # ▼▼▼【移植】ここから最終レポート、保存、グラフ描画のロジック▼▼▼
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    total_train_time = sum(train_times.values())
    total_ccs_time = sum(ccs_times.values())
    total_wall_clock_time = time.time() - start_time
    
    report_lines = [
        "```", f"Results after {args.epochs} global rounds:", "",
        f"|---- Avg Train Accuracy: {100*train_accuracy[-1]:.2f}%",
        f"|---- Test Accuracy: {100*test_acc:.2f}%",
        f"CCS time (avg per client): {total_ccs_time / args.num_users:.3f}s",
        f"Train time (sum of clients): {total_train_time:.3f}s",
        f"pruning_rate: {args.prune_rate}, el2n: {args.el2n}",
        f"--num_per_client: {args.num_per_client} ,local_ep: {args.local_ep}",
        f"Total Wall Clock Time: {total_wall_clock_time:.3f}s", "```"
    ]
    report = "\n".join(report_lines)
    print(report)
    slackPost(report)

        # ▼▼▼【修正】ここからグラフ保存処理▼▼▼
    # 保存先のファイル名を生成
    plot_filename = f'../save/plots/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_acc.png'
    
    # 【追加】保存先のディレクトリが存在しない場合に作成する
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    # グラフの描画と保存
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(plot_filename)
# --- 6. APIサーバーをシャットダウン ---
    print("\nExperiment finished. Shutting down the API server.")
    try:
        requests.post('http://127.0.0.1:5000/shutdown')
    except requests.exceptions.ConnectionError:
        print("Server already down.")
    server_thread.join(timeout=5)
    
    # ▲▲▲【修正】ここまで▲▲▲
if __name__ == '__main__':
    args = args_parser()
    train_thread = threading.Thread(target=training_loop, args=(args,))
    train_thread.start()
    print(f"APIサーバーを http://0.0.0.0:5000 で起動します。")
    app.run(host='0.0.0.0', port=5000, debug=False)
    train_thread.join()
    print("プログラムを終了します。")