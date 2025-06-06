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

from update import LocalUpdate         # 元リポジトリの LocalUpdate を利用
from utils import get_dataset          # get_dataset をそのまま使う
from models import MLP, CNNMnist, CNNFashion_Mnist
import torchvision.models              # ResNet の読み込み用
import copy

# 元のメイン側で使っている args_parser をインポート
from options import args_parser as main_args_parser


def parse_args():
    parser = argparse.ArgumentParser()  # ✅ 正しくは ArgumentParser
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--pruned_data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # exp_details(args) は client 側では不要なので呼ばない

    # ――― デバイス設定 ―――
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ――― データセットと user_groups を取得 ―――
    train_dataset, _, user_groups = get_dataset(args)

    # ――― Global モデルの構築 & 重みロード ―――
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # CIFAR の場合、main 側では num_classes=args.num_users で ResNet18 を作っている想定
            global_model = torchvision.models.resnet18(num_classes=args.num_users)
        else:
            raise ValueError("Unknown dataset")
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        dim_in = 1
        for s in img_size:
            dim_in *= s
        global_model = MLP(dim_in=dim_in, dim_hidden=64, dim_out=args.num_users)
    else:
        raise ValueError("Unknown model type")

    # Global model の重みをサブプロセス側に渡されたパスから読み込む
    try:
        global_model.load_state_dict(torch.load(args.global_model_path, map_location=device))
    except Exception as e:
        print(f"[Client {args.cid}] ERROR: failed to load global model at '{args.global_model_path}': {e}")
        exit(1)

    global_model.to(device)

    # ――― LocalUpdate オブジェクトを生成 ―――
    cid = args.cid
    local_update = LocalUpdate(
        args=args,
        dataset=train_dataset,
        idxs=user_groups[cid],
        logger=None,
        client_id=cid
    )

    # ――― Update weights（ほぼ元の update_weights をコピー） ―――
    model = copy.deepcopy(global_model)
    model.train()

    # オプティマイザ設定
    lr = args.lr * (0.5 ** (args.round // 10))
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    total_cpu = 0.0
    total_mem = 0.0
    count_samples = 0
    epoch_loss = []

    # ――― 各ローカルエポックごとに学習 ―――
    for _ in range(args.local_ep):
        batch_loss = []
        for batch_idx, (_, images, labels) in enumerate(local_update.trainloader):
            images, labels = images.to(device), labels.to(device)

            # (A) 学習前に CPU% とメモリをサンプリング
            cpu_before = psutil.cpu_percent(interval=None)
            mem_before = psutil.Process().memory_info().rss / (1024 ** 2)  # MB

            # 通常の SGD ステップ
            optimizer.zero_grad()
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
    out_model_path = f"client_{cid}_round{args.round}_model.pth"
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
    out_metrics_path = f"metrics_{cid}_round{args.round}.json"
    try:
        with open(out_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        print(f"[Client {cid}] ERROR writing metrics to '{out_metrics_path}': {e}")
        exit(1)

    # ――― 最後に STDOUT へサマリを出力（subprocess 側で読むため） ―――
    print(f"CLIENT {cid} ROUND {args.round} => loss={avg_loss:.4f}, avg_cpu={avg_cpu:.1f}, avg_mem={avg_mem:.1f}")
