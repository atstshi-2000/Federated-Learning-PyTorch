#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import sympy as sp

import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

from sklearn.preprocessing import StandardScaler
from ccs_utils import coverage_centric_selection
import psutil
import json
from torch.utils.data import DataLoader
from collections import defaultdict


def main():
    # 1. 引数の取得
    args = args_parser()

    # 2. デバイス設定
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # 3. データロード & クライアント分割
    train_dataset, test_dataset, users_groups = get_dataset(args)
    idxs_users = list(dict_users.keys())

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            #global_model = torchvision.models.resnet18(
            #    num_classes=args.num_classes)
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            # global_model = torchvision.models.resnet18(
                # num_classes=args.num_classes)
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # global_model = CNNCifar(args=args)
            # global_model = torchvision.models.resnet18(
                # num_classes=1000, pretrained=True)
                # num_classes=1000, pretrained=True)
            global_model = torchvision.models.resnet18(
                num_classes=args.num_classes)
            # global_model = torchvision.models.resnet18(
                # weights = 'IMAGENET1K_V1',num_classes=args.num_classes)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')    
    # 5. ローカルモデルオブジェクト生成
    local_models = {uid: LocalUpdate(args, dataset_train, dict_users[uid]) 
                    for uid in idxs_users}

    # 6. グローバルトレーニングラウンド
    for epoch in range(args.epochs):
        # (通常のFedAvg等の処理：省略可)
        ...

    # 7. el2n == 5 の場合のみ CCS 処理
    if args.el2n == 5:
        print(">>> Running CCS selection")

        # 7.1 各クライアントから el2n スコア収集
        el2n_scores = {}
        for uid in idxs_users:
            scores = local_models[uid].compute_el2n_scores_4(global_model)
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().tolist()
            el2n_scores[uid] = scores

        # 7.2 flatten + map 作成
        flattened_scores = []
        global_to_local = []
        for uid, scores in el2n_scores.items():
            for i, s in enumerate(scores):
                flattened_scores.append(s)
                global_to_local.append((uid, i))
        flattened_scores = np.array(flattened_scores)

        # 7.3 CCS 実行
        keep_ratio = getattr(args, "el2n_ratio", 0.5)
        num_to_keep = int(len(flattened_scores) * keep_ratio)
        keep_global = coverage_centric_selection(flattened_scores, num_to_keep, num_groups=100)

        # 7.4 クライアント別インデックス再構成
        client_keep = defaultdict(list)
        for gi in keep_global:
            uid, li = global_to_local[gi]
            client_keep[uid].append(li)

        # 7.5 各クライアントに伝えてデータ再構成
        for uid in idxs_users:
            local_models[uid].update_dataset(client_keep[uid], el2n=args.el2n)

    # 8. 以降、必要なら再トレーニングなど
    ...

if __name__ == "__main__":
    main()
