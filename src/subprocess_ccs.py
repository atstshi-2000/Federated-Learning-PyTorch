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
import subprocess

# Slack Webhook URL
SLACKURL = 'YOUR_SLACK_WEBHOOK_URL_HERE'

# slack送信メソッド
def slackPost(message):
    slack = slackweb.Slack(url=SLACKURL)
    slack.notify(text=message)

def compute_keep_ratios(
    exec_times: dict,
    cpu_usages: dict,
    mem_usages: dict,
    alpha_time=0.4,
    alpha_cpu=0.3,
    alpha_mem=0.3,
    min_ratio=0.2,
    max_ratio=0.9
):
    """
    三種の指標（累積学習時間、CPU使用率、メモリ使用量）を混合し、
    最終的に ∑ keep_ratio_per_client == 1.0 となるように返す。

    Args:
        exec_times (dict[cid -> float]): 累積学習時間
        cpu_usages (dict[cid -> float]): CPU 使用率（直近ラウンド or 累積）
        mem_usages (dict[cid -> float]): メモリ使用量（直近ラウンド or 累積）
        alpha_time, alpha_cpu, alpha_mem (float): 各指標の重み合計は 1.0
        min_ratio (float): 比率の最小値（例: 0.2）
        max_ratio (float): 比率の最大値（例: 0.9）

    Returns:
        keep_ratio_per_client (dict[cid -> float])：
            各クライアントの保持比率。合計は 1.0 になる。
    """
    import numpy as np

    cids = list(exec_times.keys())
    n = len(cids)

    # 1) 実行時間の逆数
    times = np.array([exec_times[cid] for cid in cids])
    inv_time = 1.0 / (times + 1e-8)

    # 2) CPU 使用率の逆数
    cpus = np.array([cpu_usages[cid] for cid in cids])
    inv_cpu = 1.0 / (cpus + 1e-8)

    # 3) メモリ使用量の逆数
    mems = np.array([mem_usages[cid] for cid in cids])
    inv_mem = 1.0 / (mems + 1e-8)

    # - 正規化（0～1 にスケーリング）
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-8:
            return np.ones_like(x)  # 全て同じなら一律 1
        return (x - mn) / (mx - mn)

    norm_time = normalize(inv_time)
    norm_cpu  = normalize(inv_cpu)
    norm_mem  = normalize(inv_mem)

    # 4) 重みづけ合成（0～1 の範囲）
    mixed = alpha_time * norm_time + alpha_cpu * norm_cpu + alpha_mem * norm_mem

    # 5) すべて非負にしておく（念のため）
    mixed = np.clip(mixed, 0.0, None)

    # 6) 「クライアント間で合計＝1」に正規化
    total = mixed.sum()
    if total < 1e-8:
        # もし全て 0 に近いなら一律均等割り
        normed = np.ones_like(mixed) / n
    else:
        normed = mixed / total

    # 7) 最後に [min_ratio, max_ratio] にマッピングすると合計が1を外れる可能性があるので、
    #    まず min_ratio～max_ratio の範囲にスケールし、その後でクライアント間で再調整して合計1.0へ
    #    (a) 仮に min_ratio～max_ratio にリニア変換してみる
    scaled = min_ratio + (max_ratio - min_ratio) * normed  # まだ ∑≠1

    # (b) ここで ∑scaled を S とすると、最終的に keep_ratio_per_client[i] = scaled[i]/S で合計が1になる
    S = scaled.sum()
    if S < 1e-8:
        # 万一全て 0 なら一律均等
        final = np.ones_like(scaled) / n
    else:
        final = scaled / S

    # 8) 結果を辞書で返す
    keep_ratio_per_client = {cid: float(final[i]) for i, cid in enumerate(cids)}
    return keep_ratio_per_client

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            # num_classes=args.num_classes)
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
    ccs_times = {cid: 0.0 for cid in range(args.num_users)}
    train_times = {cid: 0.0 for cid in range(args.num_users)}
    # ① 各指標の初期化
    cpu_usages    = {cid: 0.0 for cid in range(args.num_users)}
    mem_usages    = {cid: 0.0 for cid in range(args.num_users)}
    # 事前準備: smoothed_cpu を 0–100 の適当な初期値で定義しておく
    smoothed_cpu = {cid: 50.0 for cid in range(args.num_users)}  # e.g. 初期値 50%
    client_thread_limits = {
    # 各クライアントのスレッド数制限を設定(max 12)
    0: 8,
    1: 10,
    2: 12,
    3: 6,
    4: 11,
    5: 9,
    6: 8,
    7: 6,
    8: 10,
    9: 9,
    10: 11,
}
    alpha = 0.3  # EMA 平滑係数
    # ループ前に空のリストを用意
    mem_history = [] # 各ラウンドの memory_MB を格納
    size_history = [] # 各ラウンドの comm_KB を格納
    # CCSの初期化
    i_ccs = IncrementalCCS(num_groups=100, seed=args.seed)
    last_client_keep = defaultdict(list)  # 最後のクライアント保持サンプルを保存するための変数
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
            global_to_local = [] # (client_id, local_idx) マッピング
            t_ccs_all_start = time.monotonic()
            # for cid in idxs_users:全数対象
            for cid in range(args.num_users):
                scores_np = local_models[cid].compute_el2n_scores_4(global_model).detach().cpu().numpy()
                all_scores.append(scores_np)
                for li in range(len(scores_np)):
                    global_to_local.append((cid, li))
            all_scores = np.concatenate(all_scores, axis=0)
            N_total = len(all_scores)
            # CCS 実行
            prune_rate = args.prune_rate
            remaining_ratio = 1.0 - prune_rate * (epoch // 10)
            remaining_ratio = max(0.0, remaining_ratio)
            num_to_keep = int(N_total * remaining_ratio)
            num_groups = 100
            # ===== 新規追加: 各クライアントのリソースに応じて保持比率を決定 =====
            keep_ratio_per_client = compute_keep_ratios(
            exec_times=train_times,
            cpu_usages=cpu_usages,
            mem_usages=mem_usages,
            alpha_time=0.4,
            alpha_cpu=0.3,
            alpha_mem=0.3,
            min_ratio=0.2,
            max_ratio=0.9
        )
            # ===== 新規追加: 各クライアントのリソースに応じて保持比率を決定 =====
            client_keep = i_ccs.update_and_select(
                scores=all_scores,
                num_to_keep=num_to_keep,
                global_to_local=global_to_local,
                keep_ratio_per_client=keep_ratio_per_client
            )
            #インクリメンタルCCS
            # keep_global = i_ccs.update_and_select(all_scores, num_to_keep)
            # keep_global = coverage_centric_selection(
            # normalized,
            # num_to_keep=num_to_keep,
            # num_groups=num_groups
            # )

            # # クライアントごとに戻す
            # for gidx in keep_global:
            # cid, lid = global_to_local[gidx]
            # client_keep[cid].append(lid)
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
            last_client_keep = client_keep.copy() # ← ここで保存
        # ========== [3] 各クライアントのローカル学習 ==========
        # まず、今ラウンドのグローバルモデルをファイルに保存しておく
        for cid in idxs_users:
            model_path = f"client_{cid}_round{epoch}_model.pth"
            torch.save(local_models[cid].state_dict(), model_path)
            if not os.path.exists(model_path):
                print(f"[Warning] {model_path} not found. Skipping this client.")
                continue  # またはエラー処理へ
            # ── (1) クライアントごとに使うスレッド数を取得 ──
            num_threads = client_thread_limits.get(cid, 1)

            # ── (2) サブプロセス起動時の環境変数をコピーし、スレッド数を固定 ──
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(num_threads)
            env["MKL_NUM_THREADS"] = str(num_threads)

            # ── (3) サブプロセスで client_train.py を呼び出す ──
            #      必要な引数は「クライアントID」「現ラウンド」「グローバルモデルのパス」
            cmd = [
                "python", "client_train.py",
                "--cid", str(cid),
                "--round", str(epoch),
                "--global_model_path", model_path
            ]

            print(f"[Epoch {epoch+1}][Client {cid}] start training with {num_threads} threads")
            t_train_start = time.monotonic()
            result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)   # ←ここで "src/" をカレントにする
        )
            if result.returncode != 0:
                print(f"!! Client {cid} failed !! stderr:\n{result.stderr}")
            else:
                print(f"Client {cid} stdout summary:\n{result.stdout.splitlines()[-1]}")
            t_train_end = time.monotonic()

            if result.returncode != 0:
                print(f"!! Client {cid} failed !! stderr:\n{result.stderr}")
                # 必要ならここで raise するかスキップする
            else:
                # サブプロセス側で last line に学習結果を print しているので出力表示
                print(f"Client {cid} stdout summary:\n{result.stdout.splitlines()[-1]}")

            # ── (4) サブプロセスで作成された「クライアントモデル」と「メトリクス」を読み込む ──
            #  - client_{cid}_round{epoch}_model.pth  (重みファイル)
            #  - metrics_{cid}_round{epoch}.json       (loss, avg_cpu, avg_mem を含むJSON)
            local_weights.append(torch.load(model_path, map_location=device))

            info = json.loads(open(f"metrics_{cid}_round{epoch}.json", "r").read())
            cpu_usages[cid] = info["avg_cpu"]
            mem_usages[cid] = info["avg_mem"]

            print(f"[Epoch {epoch+1}][Client {cid}] "
                f"avg_cpu={cpu_usages[cid]:.1f}%, avg_mem={cpu_usages[cid]:.1f}MB, "
                f"train_time={(t_train_end - t_train_start):.3f}s")

            train_times[cid] += (t_train_end - t_train_start)
            local_losses.append(info["loss"])
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
        total_ccs = sum(ccs_times.values())/num_users
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
        report_lines.append(f" Client {cid}: {kept}")
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
        f"|---- 平均モデルサイズ : {avg_size:.1f} KB",
        f"|---- 最終ラウンドモデルサイズ: {final_size:.1f} KB",
    ]


    # 最終的な report
    report = "\n".join(report_lines)
    slackPost(report)
    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_Client[{}]_iid[{}]_Epoch[{}]_Batch_s[{}]_lr[{}]__el2n[{}]_num_users[{}]_threshold[{}]_percent[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs, args.lr ,args.el2n, args.num_users, args.threshold,args.percent)
    os.makedirs(os.path.dirname(file_name), exist_ok=True) # ディレクトリを作成
    # Ensure the directory exists before saving the file
    file_dir = os.path.dirname(file_name)
    os.makedirs(file_dir, exist_ok=True) # Create the directory if it doesn't exist



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