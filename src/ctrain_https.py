#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6

import os
import argparse
import time
import json
import torch
import psutil
import numpy as np
import copy
import requests
import gzip
import io

import torchvision.models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from models import MLP, CNNMnist, CNNFashion_Mnist

# ===================================================================
# === 引数パーサー (クライアント起動時に最低限必要な情報のみ) ===
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument('--client_id', type=int, required=True, help='このクライアントの固有ID (0, 1, 2...)')
    parser.add_argument('--server_address', type=str, default="http://192.168.30.191:5000", help='サーバーのIPアドレスとポート')
    return parser.parse_args()

# ===================================================================
# === 通信ヘルパー関数 (変更なし) ===
# ===================================================================
def download_file_from_server(server_address, remote_path, local_path):
    url = f"{server_address}/files/{remote_path}"
    try:
        print(f"Downloading: {url} -> {local_path}")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"!! ファイルのダウンロードに失敗: {e}")
        return False

def upload_files_to_server(server_address, client_id, epoch, model_state_dict, metrics_dict):
    url = f"{server_address}/upload/{client_id}/{epoch}"
    try:
        print(f"Uploading compressed results to: {url}")
        
        # モデルの重みをメモリ上でGZIP圧縮
        buffer = io.BytesIO()
        torch.save(model_state_dict, buffer)
        buffer.seek(0)
        compressed_model_data = gzip.compress(buffer.read())

        # メトリクスはJSONとしてエンコード
        metrics_data = json.dumps(metrics_dict).encode('utf-8')

        files = {
            'model': ('model.gz', compressed_model_data, 'application/gzip'),
            'metrics': ('metrics.json', metrics_data, 'application/json')
        }
        
        r = requests.post(url, files=files, timeout=120) # タイムアウトを少し延長
        r.raise_for_status()
        print("アップロードに成功しました。")
        return True
    except requests.exceptions.RequestException as e:
        print(f"!! ファイルのアップロードに失敗: {e}")
        return False

# ===================================================================
# === データセットクラス (修正済み) ===
# ===================================================================
class PrunedDataset(Dataset):
    def __init__(self, data_list, label_list, transform=None):
        self.data_list, self.label_list, self.transform = data_list, label_list, transform
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        # データは既にTensor形式なので、そのまま扱う
        x, y = self.data_list[idx], self.label_list[idx]
        if self.transform:
            # Tensorに直接データ拡張を適用する
            x = self.transform(x)
        return x, y

# ===================================================================
# === 1ラウンド分の学習を実行するメイン関数 (修正済み) ===
# ===================================================================
def run_training(client_id, server_address, epoch, args_from_server):
    print(f"\n--- [Client {client_id}] ラウンド {epoch+1} の学習を開始 ---")
    
    # 1. サーバーから渡された設定(args)を反映
    args = argparse.Namespace(**args_from_server)
    base_dir = os.path.dirname(__file__)
    print("[デバッグ] ステップ1：設定を反映しました。")

    # 2. このラウンドで必要なファイルパスを動的に決定
    global_model_filename = f"global_model_round{epoch}.pth"
    pruning_epoch = 0
    if args.el2n != 0 and epoch >= (args.start_prune_round - 1):
        offset = epoch - (args.start_prune_round - 1)
        pruning_epoch = (args.start_prune_round - 1) + (offset // args.prune_interval) * args.prune_interval
    pruned_data_filename = f"client_{client_id}_round{pruning_epoch}.pt"
    print("[デバッグ] ステップ2：ファイルパスを決定しました。")

    # 3. 必要なファイルをサーバーからダウンロード
    if not download_file_from_server(server_address, f"PTH/{global_model_filename}", os.path.join(base_dir, "PTH", global_model_filename)): return
    if not download_file_from_server(server_address, f"pruned_data/{pruned_data_filename}", os.path.join(base_dir, "pruned_data", pruned_data_filename)): return
    print("[デバッグ] ステップ3：ファイルのダウンロードが完了しました。")

    # 4. デバイス設定とモデル構築
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    torch.manual_seed(args.seed + epoch + client_id)
    print("[デバッグ] ステップ4：モデルの構築が完了しました。")
    if args.model == 'cnn':
        if args.dataset == 'cifar': global_model = torchvision.models.resnet18(num_classes=args.num_classes)
        elif args.dataset == 'mnist': global_model = CNNMnist(args=args)
        else: global_model = CNNFashion_Mnist(args=args)
    elif args.model == 'mlp':
        len_in = int(np.prod((3, 32, 32) if args.dataset == 'cifar' else (1, 28, 28)))
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else: exit("Unknown model type")
    global_model.load_state_dict(torch.load(os.path.join(base_dir, "PTH", global_model_filename), map_location=device))
    
    # 5. データセットとデータローダーの準備
    pruned_data_path = os.path.join(base_dir, "pruned_data", pruned_data_filename)
    pruned = torch.load(pruned_data_path, map_location="cpu", weights_only=False)
    print("[デバッグ] ステップ5：データローダーの準備が完了しました。")
    # ▼▼▼【修正点2】データ拡張(Transform)の修正 (Normalizeを削除)▼▼▼
    transform = None
    if args.dataset == 'cifar':
        # データは既に正規化済みTensorなので、ToPILImageやToTensor、Normalizeは不要。
        # Tensorに直接適用できるデータ拡張のみを使用する。
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    
    train_dataset = PrunedDataset(pruned["data"], pruned["labels"], transform=transform)
        # ▼▼▼【修正】データが空の場合の処理を追加▼▼▼
    if len(train_dataset) == 0:
        print("⚠️ データセットが空のため、このラウンドの学習をスキップします。")
        # 空の結果をサーバーに即座に報告
        metrics = {"loss": 0, "avg_cpu": 0, "avg_mem": 0, "train_time": 0}
        
        # 空のモデルファイルとメトリクスを作成
        output_model_path = os.path.join(base_dir, "PTH", f"client_{client_id}_round{epoch}_model.pth")
        output_metrics_path = os.path.join(base_dir, "JSON", f"metrics_{client_id}_round{epoch}.json")
        
        # モデルはグローバルモデルをそのまま使う
        torch.save(global_model.state_dict(), output_model_path) 
        with open(output_metrics_path, "w") as f: json.dump(metrics, f, indent=4)
        
        upload_files_to_server(server_address, client_id, epoch, output_model_path, output_metrics_path)
        print(f"--- [Client {client_id}] ラウンド {epoch+1} をスキップ報告しました ---")
        return # このラウンドの処理を終了
    # ▲▲▲【修正】ここまで▲▲▲
    train_loader  = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    
    # 6. ローカル学習の実行
    model = copy.deepcopy(global_model).to(device)
    model.train()

    # ▼▼▼【デバッグ用】学習前のモデルの重みを出力▼▼▼
    print(f"[デバッグ] 学習前のLocal Modelの重みの一部: {model.state_dict()['conv1.weight'][0,0,0,:5]}")

    lr = args.lr
    if epoch >= 40: lr *= 0.01
    elif epoch >= 25: lr *= 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) if args.optimizer == 'sgd' else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    print("[デバッグ] ステップ6：ローカル学習ループを開始します...")
    # ▼▼▼【修正点1】学習時間(train_time)の計測開始▼▼▼
    start_train_time = time.time()
    total_cpu, total_mem, count_samples, epoch_loss = 0.0, 0.0, 0, []
    for _ in range(args.local_ep):
        batch_loss = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            cpu_before, mem_before = psutil.cpu_percent(interval=None), psutil.Process().memory_info().rss / (1024 ** 2)
            optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step()
            batch_loss.append(loss.item())
            cpu_after, mem_after = psutil.cpu_percent(interval=None), psutil.Process().memory_info().rss / (1024 ** 2)
            total_cpu += (cpu_before + cpu_after) / 2.0; total_mem += (mem_before + mem_after) / 2.0; count_samples += 1
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    
    # ▼▼▼【修正点1】学習時間の計測終了▼▼▼
    end_train_time = time.time()
    # ▼▼▼【デバッグ用】学習後のモデルの重みを出力▼▼▼
    print(f"[デバッグ] 学習後のLocal Modelの重みの一部: {model.state_dict()['conv1.weight'][0,0,0,:5]}")
    print("[デバッグ] ステップ6：ローカル学習ループが完了しました。")
    # 7. 結果の計算とファイル保存
    avg_loss = float(np.mean(epoch_loss)) if epoch_loss else 0.0
    avg_cpu  = float(total_cpu / max(count_samples, 1))
    avg_mem  = float(total_mem / max(count_samples, 1))
    print("[デバッグ] ステップ7：結果の保存が完了しました。")
    # ▼▼▼【修正点1】メトリクスにtrain_timeを追加▼▼▼
    metrics = {
        "loss": avg_loss, "avg_cpu": avg_cpu, "avg_mem": avg_mem,
        "train_time": end_train_time - start_train_time
    }
    
    output_model_path = os.path.join(base_dir, "PTH", f"client_{client_id}_round{epoch}_model.pth")
    output_metrics_path = os.path.join(base_dir, "JSON", f"metrics_{client_id}_round{epoch}.json")
    torch.save(model.state_dict(), output_model_path)
    with open(output_metrics_path, "w") as f: json.dump(metrics, f, indent=4)
        
    # 8. 学習結果をサーバーにアップロード
    print("[デバッグ] ステップ8：結果のアップロードを開始します...")
    # upload_files_to_server(server_address, client_id, epoch, output_model_path, output_metrics_path) # ← 古い呼び出し
    upload_files_to_server(server_address, client_id, epoch, model.state_dict(), metrics) # ← 新しい呼び出し
    print(f"--- [Client {client_id}] ラウンド {epoch+1} の学習を完了・報告しました ---")
# ===================================================================
# === クライアントの常駐ループ (修正済み) ===
# ===================================================================
if __name__ == '__main__':
    client_args = parse_args()
    print(f"クライアント {client_args.client_id} を起動しました。サーバー ({client_args.server_address}) に接続します。")

    # サーバーから基本設定を取得
    try:
        response = requests.get(f"{client_args.server_address}/get_config", timeout=10)
        response.raise_for_status()
        args_from_server = response.json()
        print("サーバーから基本設定の取得に成功しました。")
    except requests.exceptions.RequestException as e:
        print(f"!! 致命的エラー: サーバーから設定を取得できませんでした: {e}")
        exit()
    # ▼▼▼【追加】サーバーに起動完了を報告する▼▼▼
    try:
        print("サーバーに起動完了を報告しています...")
        requests.post(f"{client_args.server_address}/check_in/{client_args.client_id}", timeout=10)
        print("報告が完了しました。")
    except requests.exceptions.RequestException as e:
        print(f"!! 起動報告に失敗しました: {e}")
        # 報告に失敗した場合は、実験に参加できないため終了するのが安全
        exit()
    # ▲▲▲【追加】ここまで▲▲▲
    # 無限ループでサーバーにタスクを問い合わせる
    while True:
        try:
            res = requests.get(f"{client_args.server_address}/get_task/{client_args.client_id}", timeout=10)
            res.raise_for_status()
            task_info = res.json()
            
            task = task_info.get("task")
            if task == "train":
                current_epoch = task_info.get("epoch")
                run_training(client_args.client_id, client_args.server_address, current_epoch, args_from_server)
            elif task == "wait":
                retry_after = task_info.get("retry_after", 30)
                print(f". (待機中... {retry_after}秒後に再試行)")
                time.sleep(retry_after)
            elif task == "shutdown":
                print("サーバーから終了シグナルを受信しました。プログラムを終了します。")
                break
            else:
                print(f"不明なタスク '{task}' を受信しました。30秒待機します。")
                time.sleep(30)
        except requests.exceptions.RequestException as e:
            print(f"!! サーバーとの通信に失敗しました: {e}。60秒後に再試行します。")
            time.sleep(60)