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
from utils import get_dataset

from update import LocalUpdate         # 元リポジトリの LocalUpdate を利用
from utils import get_dataset          # get_dataset をそのまま使う
from models import MLP, CNNMnist, CNNFashion_Mnist
import torchvision.models              # ResNet の読み込み用
import copy

# 元のメイン側で使っている args_parser をインポート
from options import args_parser as main_args_parser
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    # ──────────────── 既存のクライアント用引数 ────────────────
    parser.add_argument("--client_id", type=int, required=True,
                        help="クライアントのID (0～num_users-1)")
    parser.add_argument("--epoch", type=int, required=True,
                        help="現在のフェデレーションラウンド番号")
    parser.add_argument("--pruned_data_path", type=str, required=True,
                        help="CCS 後に絞り込んだデータセット (.pt) のパス")
    parser.add_argument("--device", type=str, default="cpu",
                        help="使用デバイス文字列 (例: 'cuda:0' or 'cpu')")
    parser.add_argument("--seed", type=int, required=True,
                        help="乱数シード (main と合わせること)")
    parser.add_argument("--model", type=str, choices=["cnn","mlp"], required=True,
                        help="モデルの種類 (cnn or mlp)")
    parser.add_argument("--dataset", type=str, choices=["mnist","fmnist","cifar"], required=True,
                        help="データセット名 (mnist, fmnist, cifar)")
    parser.add_argument("--lr", type=float, required=True,
                        help="学習率 (main と合わせること)")
    parser.add_argument("--optimizer", type=str, choices=["sgd","adam"], required=True,
                        help="オプティマイザの種類 (sgd or adam)")
    parser.add_argument("--local_ep", type=int, required=True,
                        help="ローカルエポック数 (main と合わせること)")
    parser.add_argument("--global_model_path", type=str, required=True,
                        help="グローバルモデル重みファイルのパス")
    # ────────────────────────────────────────────────

    # ───────────── get_dataset 用に新たに追加 ──────────────
    parser.add_argument("--iid", type=int, choices=[0,1], default=1,
                        help="IID 分割するなら 1, Non-IID なら 0")
    parser.add_argument("--unequal", type=int, choices=[0,1], default=0,
                        help="Unequal Non-IID 分割をするなら 1, しないなら 0")
    parser.add_argument("--num_users", type=int, required=True,
                        help="全クライアント数(メインと合わせること)")
    parser.add_argument("--num_per_client", type=int, default=0,
                        help="IID 分割時の1クライアントあたりサンプル数 (IID=1 のとき使用)")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="データセットのクラス数")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="学習済みモデルの保存先パス")
    parser.add_argument("--output_metrics_path", type=str, required=True,
                        help="メトリクスJSONの保存先パス")
    return parser.parse_args()

class PrunedDataset(Dataset):
    def __init__(self, data_list, label_list, transform=None):
        self.data_list  = data_list
        self.label_list = label_list
        self.transform  = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx]
        y = self.label_list[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

if __name__ == '__main__':
    # print("DEBUG0: main.py started, parsing args...")
    args = parse_args()

    # exp_details(args) は client 側では不要なので呼ばない

    # ――― デバイス設定 ―――
    # print("DEBUG1: parse_args OK, args =", args)
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

       # ――― get_dataset で元の全データを一度に取得 ―――
    # print("DEBUG3: About to call get_dataset")
    # train_dataset, test_dataset, user_groups = get_dataset(args)
    # print("DEBUG4: get_dataset OK, len(train_dataset) =", len(train_dataset))
    # ――― Global モデルの構築 & 重みロード ―――
    # print("DEBUG5: About to define + load global_model")
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # ↓ ここを修正 ↓
            global_model = torchvision.models.resnet18(num_classes=args.num_classes)
        else:
            raise ValueError("Unknown dataset")

    elif args.model == 'mlp':
        img_size = 3072  # CIFAR-10 の場合は 32x32x3 = 3072
        len_in = 1
        for x in img_size:
            len_in *= x
        # ↓ ここを修正 ↓
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        raise ValueError("Unknown model type")

    # 重複していた2つ目のモデル定義ブロックは削除する

    try:
        global_model.load_state_dict(torch.load(args.global_model_path, map_location=device))
    except Exception as e:
        print(f"[Client {args.client_id}] ERROR loading global model: {e}")
        exit(1)

    global_model.to(device)
    global_model.train()
    # print("DEBUG6: Loaded global_model from", args.global_model_path)

    # print("DEBUG7: About to load pruned_data from", args.pruned_data_path)
    pruned = torch.load(args.pruned_data_path, map_location="cpu")
    # print("DEBUG8: pruned_data keys =", list(pruned.keys()))
    data_list  = pruned["data"]
    label_list = pruned["labels"]
    # print("DEBUG9: data_list length =", len(data_list), "label_list length =", len(label_list))

    if args.dataset == 'cifar':
    # 注意：.ptファイルに保存されているデータがすでに正規化済みのため、
    # ここではNormalizeは含めません。
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    else:
    # 他のデータセットの場合は適切なAugmentationを設定
        transform = None
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # print("DEBUG10: Creating PrunedDataset + DataLoader")
    train_dataset = PrunedDataset(data_list, label_list, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # print("DEBUG11: train_loader created, len(train_loader) =", len(train_loader))

    # ――― LocalUpdate オブジェクトを生成 ―――
    cid = args.client_id
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()

    # # オプティマイザ設定
    lr = args.lr  # 毎ラウンド、まず初期値にリセットされる
    # lr = args.lr * (0.5 ** (args.epoch // 10))
    # 例えば50エポック実行する場合、25ラウンド目と40ラウンド目で学習率を1/10にする
    if args.epoch >= 40:
        lr *= 0.01
    elif args.epoch >= 25:
        lr *= 0.1
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    total_cpu = 0.0
    total_mem = 0.0
    count_samples = 0
    epoch_loss = []
    # print("DEBUG12: About to enter local training loop")
    # ――― 各ローカルエポックごとに学習 ―――
    for _ in range(args.local_ep):
        batch_loss = []
        print("len(train_loader.dataset):", len(train_loader.dataset))
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # (A) 学習前に CPU% とメモリをサンプリング
            cpu_before = psutil.cpu_percent(interval=None)
            mem_before = psutil.Process().memory_info().rss / (1024 ** 2)  # MB

            # 通常の SGD ステップ
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

            # (B) 学習後に CPU% とメモリをサンプリング
            cpu_after = psutil.cpu_percent(interval=None)
            mem_after = psutil.Process().memory_info().rss / (1024 ** 2)

            # (C) 前後の平均を累積
            total_cpu += (cpu_before + cpu_after) / 2.0
            total_mem += (mem_before + mem_after) / 2.0
            count_samples += 1

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    # ――― 平均ロス／平均 CPU%／平均メモリ を計算 ―――
    avg_loss = float(np.mean(epoch_loss))
    avg_cpu  = float(total_cpu / max(count_samples, 1))
    avg_mem  = float(total_mem / max(count_samples, 1))

    # ――― 学習後のローカルモデルを保存 ―――
    out_model_path = args.output_model_path
    try:
        torch.save(model.state_dict(), out_model_path)
    except Exception as e:
        print(f"[Client {cid}] ERROR saving model to '{out_model_path}': {e}")
        exit(1)

    # ――― メトリクスを JSON で書き出し ―――
    metrics = {
        "loss": avg_loss,
        "avg_cpu": avg_cpu,
        "avg_mem": avg_mem
    }
    out_metrics_path = args.output_metrics_path
    try:
        with open(out_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        print(f"[Client {cid}] ERROR writing metrics to '{out_metrics_path}': {e}")
        exit(1)

    # ――― 最後に STDOUT へサマリを出力（subprocess 側で読むため） ―――
    print(f"CLIENT {cid} ROUND {args.epoch} => loss={avg_loss:.4f}, avg_cpu={avg_cpu:.1f}, avg_mem={avg_mem:.1f}")
