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
from ccs_utils import IncrementalCCS
import psutil
import json
from torch.utils.data import DataLoader
from collections import defaultdict
import slackweb



# slack送信メソッド
def slackPost(message):
    slack = slackweb.Slack(url = SLACKURL)
    slack.notify(text = message)

def log_metrics(round_num, train_time, model, accuracies, log_path="logs"):
    """
    Log training metrics for each round.

    Args:
    - round_num: Current round number.
    - train_time: Training time in seconds.
    - model: Global model.
    - accuracies: List of accuracies from all clients.
    - log_path: Path to save the log file.
    """
    # メモリ使用量
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # MB単位

    # 通信サイズ
    model_size = len(pickle.dumps(model.state_dict())) / 1024  # KB単位

    # 精度のばらつき
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)

    # ログを保存
    log_data = {
        "round": round_num,
        "train_time_sec": train_time,
        "memory_MB": memory_usage,
        "comm_KB": model_size,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std
    }
    # JSON 保存
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, f"round_{round_num}.json"), "w") as f:
        json.dump(log_data, f, indent=4)

    # 返り値を追加
    return memory_usage, model_size


if __name__ == '__main__':
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    #print(args)
    exp_details(args)
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device(f"{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # el2n scores setup
    el2n_scores = {uid: [] for uid in range(args.num_users)}

    print("num of train dataset = ", len(train_dataset))
    print("num of test dataset = ", len(test_dataset))

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

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print("global_model = ",global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    time_list = []
    keep_idxs ,prun_num = [], []
    # keep_idxs_1, keep_idxs_2, keep_idxs3 = [], [], []
    pri_pru = [0] * args.num_users
    stock_accuracy = [0] * args.num_users
    choice_users = [i for i in range(args.num_users)]
    pru_num = 0
    total_pru = 0
    print_every = 1
    pru_count = 0
    val_loss_pre, counter = 0, 0
    global_round = 0
    ta,tt,te_all,fin = 0,0,0,0
    total_data = 0
    local_models = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, client_id=idx) for idx in range(args.num_users)]
    
    # クライアントオブジェクトのリストを作成
    clients = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, client_id=idx) for idx in range(args.num_users)]
    # 事前：全クライアントのCCS時間・学習時間をためる辞書を用意
    ccs_times   = {cid: 0.0 for cid in range(args.num_users)}
    train_times = {cid: 0.0 for cid in range(args.num_users)}
    # ループ前に空のリストを用意
    mem_history  = []   # 各ラウンドの memory_MB を格納
    size_history = []   # 各ラウンドの comm_KB    を格納

    # CCSの初期化
    i_ccs = IncrementalCCS(num_groups=100, seed=args.seed)

    # if args.dataset == 'cifar':
    org_num = 40000 / args.num_users
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        org_num = 48000 / args.num_users
    el2n = args.el2n
    percent = args.percent
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        print(f"\n | Global Training Round : {epoch+1} |\n")

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        num_users = args.num_users
        idxs_users = np.random.choice(choice_users, m, replace=False)
        print("Selected users:", idxs_users)
        do_prune = (global_round >= 10 and global_round % 10 == 0)
        client_keep = defaultdict(list)
        local_weights, local_losses = [], []

        # ========== [1] EL2Nスコア収集と CCS 実行（10エポックごと） ==========
        if do_prune:
            all_scores = []
            global_to_local = []  # (client_id, local_idx) マッピング

            t_ccs_all_start = time.monotonic()
            
            # for cid in idxs_users:全数対象
            for cid in range(args.num_users):
                scores_np = local_models[cid].compute_el2n_scores_4(global_model).detach().cpu().numpy()
                all_scores.append(scores_np)
                for li in range(len(scores_np)):
                    global_to_local.append((cid, li))
            all_scores = np.concatenate(all_scores, axis=0)

            # CCS 実行
            prune_rate = args.prune_rate
            total_points = all_scores.shape[0]
            num_to_keep = int(total_points * (1 - prune_rate * (epoch // 10)))
            num_groups  = 100

            # ===== 新規追加: 各クライアントのリソースに応じて保持比率を決定 =====
            cpu_usages = {cid: psutil.cpu_percent(interval=0.1) for cid in range(args.num_users)}
            max_usage = max(cpu_usages.values()) + 1e-8
            keep_ratio_per_client = {cid: 0.5 + 0.5 * (1.0 - cpu_usages[cid] / max_usage) for cid in range(args.num_users)}
            # ===== 新規追加: 各クライアントのリソースに応じて保持比率を決定 =====
            
            # CCS 実行
            client_keep = i_ccs.update_and_select(
                scores=all_scores,
                num_to_keep=num_to_keep,
                global_to_local=global_to_local,
                keep_ratio_per_client=keep_ratio_per_client
            )
            #インクリメンタルCCS
            # keep_global = i_ccs.update_and_select(all_scores, num_to_keep)
            # keep_global = coverage_centric_selection(
            #     normalized,
            #     num_to_keep=num_to_keep,
            #     num_groups=num_groups
            # )

            # # クライアントごとに戻す
            # for gidx in keep_global:
            #     cid, lid = global_to_local[gidx]
            #     client_keep[cid].append(lid)
            
            # CCS 全体の終了
            t_ccs_all_end = time.monotonic()
            # 各クライアントに「同じ CCS 全体時間」を加算しておく
            for cid in range(args.num_users):
                ccs_times[cid] += (t_ccs_all_end - t_ccs_all_start)

        # ========== [2] 再構築（全クライアント） ==========
        if do_prune and cid in client_keep:
            t_recon_start = time.monotonic()
            for cid in range(args.num_users):
                keep_idx = client_keep.get(cid, [])
                local_models[cid].update_dataset(keep_idx, el2n=args.el2n)
            t_recon_end = time.monotonic()
            for cid in range(args.num_users):
                ccs_times[cid] += (t_recon_end - t_recon_start)
            last_client_keep = client_keep.copy()  # ← ここで保存
        # ========== [3] 各クライアントのローカル学習 ==========
        for cid in idxs_users:
            # 学習部分の計測
            t_train_start = time.monotonic()
            w, loss = local_models[cid].update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch
            )
            t_train_end = time.monotonic()
            train_times[cid] += (t_train_end - t_train_start)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            # クライアントごとの時間を都度出力
            print(f"[Epoch {epoch+1}][Client {cid}] "
                f"CCS time: {ccs_times[cid]:.3f}s, "
                f"Train time: {t_train_end - t_train_start:.3f}s")

        # ========== [3] グローバルモデル更新 ==========
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # ========== [4] 評価 & ログ ==========
        list_acc = []
        global_model.eval()
        for c in range(args.num_users):
            acc, _ = clients[c].inference(model=global_model)
            list_acc.append(acc)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {loss_avg:.4f}')
        print(f'Train Accuracy: {100*train_accuracy[-1]:.2f}%\n')
        total_ccs   = sum(ccs_times.values())/num_users
        total_train = sum(train_times.values())
        print(f"=== Overall CCS time: {total_ccs:.3f}s ===")
        print(f"=== Overall Train time: {total_train:.3f}s ===")
        global_round += 1

        mem_mb, model_kb = log_metrics(
            round_num=epoch + 1,
            train_time=time.time() - start_time,
            model=global_model,
            accuracies=list_acc,
            log_path="logs")
# ラウンドごとにリストへ追加
        mem_history.append(mem_mb)
        size_history.append(model_kb)


    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    end_time = time.time()
    all_time = end_time - start_time
    print(f' \n Results after {global_round} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    # print("train time =",ta)
    print(f"Train time = : {total_train:.3f}s")
    print(f"el2n time = : {total_ccs:.3f}s")
    print(f"total time = : {all_time:.3f}s")


    # Slack 通知用レポート生成
    report_lines = [
        "```",
        f"python src/federated_main.py --dataset {args.dataset} --el2n {args.el2n}"
        f" --epoch {args.epochs} --threshold {args.threshold}",
        "",
        f"Results after {global_round} global rounds of training:",
        "",
        f" |---- 割り当てたデータ数: {args.num_per_client}",
        f" |---- プルーニング割合: {args.prune_rate}",
        f"|---- Avg Train Accuracy: {100*train_accuracy[-1]:.2f}",
        f"Training Loss : {np.mean(np.array(train_loss)):.4f}",
        f"|---- Test Accuracy: {100*test_acc:.2f}",
        f"el2n time = : {total_ccs:.3f}s",
        f"Train time = : {total_train:.3f}s",
        f"total time = : {all_time:.3f}s",
        "",
        "各クライアントデータセット:"
    ]

    # 各クライアントの保持サンプル数を追加
    for cid in range(args.num_users):
        kept = len(last_client_keep.get(cid, []))
        total_data += kept
        report_lines.append(f"  Client {cid}: {kept}")
    report_lines.append(f"Total data points: {total_data}")
    report_lines.append("```")
    # 最終レポート生成の直前
    avg_mem = sum(mem_history) / len(mem_history)
    final_mem = mem_history[-1]
    avg_size = sum(size_history) / len(size_history)
    final_size = size_history[-1]

    report_lines += [
        "",
        f"|---- 平均メモリ使用量: {avg_mem:.1f} MB",
        f"|---- 最終ラウンドメモリ使用量: {final_mem:.1f} MB",
        f"|---- 平均モデルサイズ  : {avg_size:.1f} KB",
        f"|---- 最終ラウンドモデルサイズ: {final_size:.1f} KB",
    ]
    
    # 最終的な report
    report = "\n".join(report_lines)
    slackPost(report)
    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_Client[{}]_iid[{}]_Epoch[{}]_Batch_s[{}]_lr[{}]__el2n[{}]_num_users[{}]_threshold[{}]_percent[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                args.local_ep, args.local_bs, args.lr ,args.el2n, args.num_users, args.threshold,args.percent)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)  # ディレクトリを作成
    # Ensure the directory exists before saving the file
    file_dir = os.path.dirname(file_name)
    os.makedirs(file_dir, exist_ok=True)  # Create the directory if it doesn't exist

    try:
        if args.el2n != 0:
            with open(file_name, 'wb') as f:
                pickle.dump([train_loss, train_accuracy, test_acc, keep_idxs, prun_num, args.threshold, args.percent], f)
        else:
            with open(file_name, 'wb') as f:
                pickle.dump([train_loss, train_accuracy, test_acc, keep_idxs, prun_num, args.threshold], f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Failed to save file at {file_name}. Please check the directory structure.")


    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(args.dataset, args.model, args.epochs, args.frac,args.iid, args.local_ep, args.local_bs))