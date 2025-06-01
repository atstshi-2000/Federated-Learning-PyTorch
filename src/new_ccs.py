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

# Slack Webhook URL
SLACKURL = 'YOUR_SLACK_WEBHOOK_URL_HERE'
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

# NEW: calculate_burden_aware_keep_ratios 関数 (前回提案のものをベースに)
def calculate_burden_aware_keep_ratios(
    all_client_ids, client_smoothed_metrics, system_ram_pressure_percent,
    w_time, w_cpu, w_ram, min_r, max_r, default_r
):
    # (前回の提案と同様のロジック、process_cpu_usage を使用)
    output_ratios = {cid: default_r for cid in all_client_ids}
    clients_with_metrics_ids = [
        cid for cid in all_client_ids 
        if cid in client_smoothed_metrics and \
           client_smoothed_metrics[cid].get('time_per_sample') is not None and \
           not np.isinf(client_smoothed_metrics[cid].get('time_per_sample')) and \
           client_smoothed_metrics[cid].get('process_cpu_usage') is not None
    ]
    if not clients_with_metrics_ids: return output_ratios

    times_values = np.array([client_smoothed_metrics[cid]['time_per_sample'] for cid in clients_with_metrics_ids])
    process_cpus_values = np.array([client_smoothed_metrics[cid]['process_cpu_usage'] for cid in clients_with_metrics_ids])

    norm_times = np.full_like(times_values, 0.5, dtype=float)
    if times_values.size > 0:
        min_t, max_t = np.min(times_values), np.max(times_values)
        if times_values.size > 1 and (max_t - min_t) > 1e-8: norm_times = (times_values - min_t) / (max_t - min_t + 1e-8)
    
    norm_process_cpus = np.full_like(process_cpus_values, 0.5, dtype=float)
    if process_cpus_values.size > 0:
        min_cpu, max_cpu = np.min(process_cpus_values), np.max(process_cpus_values)
        if process_cpus_values.size > 1 and (max_cpu - min_cpu) > 1e-8: norm_process_cpus = (process_cpus_values - min_cpu) / (max_cpu - min_cpu + 1e-8)

    norm_system_ram = system_ram_pressure_percent / 100.0
    burden_scores_list = []
    for i in range(len(clients_with_metrics_ids)):
        score = (w_time * norm_times[i] + w_cpu * norm_process_cpus[i] + w_ram * norm_system_ram)
        burden_scores_list.append(score)
    burden_scores = np.clip(np.array(burden_scores_list), 0.0, 1.0)
    calculated_ratios = np.clip(max_r - burden_scores * (max_r - min_r), min_r, max_r)
    for i, cid in enumerate(clients_with_metrics_ids): output_ratios[cid] = calculated_ratios[i]
    return output_ratios


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
    last_client_keep = defaultdict(list) # CCS結果を保持

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

    # NEW: クライアントごとの平滑化メトリクス用ストレージ
    client_smoothed_metrics = {
        cid: {'time_per_sample': None, 'process_cpu_usage': None, 'last_updated_round': -1}
        for cid in range(args.num_users)
    }

    local_models = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, client_id=idx) for idx in range(args.num_users)]
    i_ccs = IncrementalCCS(num_groups=getattr(args, 'num_groups_ccs', 100), seed=args.seed) # CHANGED: num_groupsをargsから


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
        start_prune_round = getattr(args, 'start_prune_round', 10) 
        prune_interval = getattr(args, 'prune_interval', 10)    
        do_prune = (epoch >= (start_prune_round -1) and \
                    (epoch - (start_prune_round - 1)) % prune_interval == 0)

        # このラウンドで計算された keep_ratio (CCS実行時のみ意味を持つ)
        current_keep_ratio_per_client = {cid: args.default_keep_ratio for cid in range(args.num_users)}

        client_keep = defaultdict(list)
        local_weights, local_losses = [], []
        t_ccs_all_start = time.monotonic()
        # ========== [1] (do_prune時) 負担スコアベースの保持比率計算 ==========
        if do_prune:
            system_ram_pressure = psutil.virtual_memory().percent # NEW: RAM使用率取得
            # print(f"System RAM pressure for round {epoch+1} burden calculation: {system_ram_pressure:.1f}%")

            current_keep_ratio_per_client = calculate_burden_aware_keep_ratios( # NEW: 呼び出し
                all_client_ids=list(range(args.num_users)),
                client_smoothed_metrics=client_smoothed_metrics,
                system_ram_pressure_percent=system_ram_pressure,
                w_time=args.burden_w_time, # CHANGED: options.py で定義した引数名に合わせる
                w_cpu=args.burden_w_cpu,   # CHANGED
                w_ram=getattr(args, 'burden_w_ram', 0.3), 
                min_r=args.min_keep_ratio, # CHANGED
                max_r=args.max_keep_ratio, # CHANGED
                default_r=args.default_keep_ratio # CHANGED
            )
            # print(f"Calculated keep_ratios for CCS (round {epoch+1}): {current_keep_ratio_per_client}")
            # REMOVED: 以前の cpu_usages に基づく keep_ratio_per_client の計算は削除
        

       # ========== [2] (do_prune時) EL2Nスコア収集とCCS実行 ==========
        # client_keep はこのラウンドのCCS結果。元コードの defaultdict(list) を流用。
        # client_keep = defaultdict(list) # ループの先頭で初期化されているはず

        if do_prune:
            # print("CCS Phase: Collecting EL2N scores and performing selection...")
            all_scores_list_for_ccs = [] 
            global_to_local_for_ccs = []  

            for cid_el2n in range(args.num_users): # 全クライアントからスコア収集
                # LocalUpdate.compute_el2n_scores_4 が現在のtrainloaderのデータでスコアを計算
                if local_models[cid_el2n].trainloader and len(local_models[cid_el2n].trainloader.dataset) > 0:
                    scores_tensor = local_models[cid_el2n].compute_el2n_scores_4(global_model)
                    if scores_tensor.numel() > 0: # スコアが空でないか確認
                        scores_np_for_client = scores_tensor.numpy()
                        all_scores_list_for_ccs.append(scores_np_for_client)
                        for local_idx in range(len(scores_np_for_client)):
                            global_to_local_for_ccs.append((cid_el2n, local_idx))
            
            if not all_scores_list_for_ccs or not global_to_local_for_ccs:
                print("No EL2N scores collected. Skipping CCS data pruning for this round.")
            else:
                all_scores_concatenated_np = np.concatenate(all_scores_list_for_ccs, axis=0)
                if all_scores_concatenated_np.size == 0:
                    print("Concatenated EL2N scores are empty. Skipping CCS.")
                else:
                    # CHANGED: num_to_keep_globally の計算をより明確に
                    num_to_keep_globally = all_scores_concatenated_np.shape[0] # デフォルトは全保持
                    if epoch >= (start_prune_round -1) :
                         prune_level_idx = (epoch - (start_prune_round - 1)) // prune_interval
                         # args.prune_rate は1回のCCSでの削減率のベース (例: 0.1 なら10%)
                         # 削減率は徐々に増やす (最大 max_overall_reduction_rate まで)
                         current_overall_reduction_target = min(args.prune_rate * (prune_level_idx +1), # +1で初回から削減
                                                               getattr(args, 'max_overall_reduction_rate', 0.8))
                         num_to_keep_globally = int(all_scores_concatenated_np.shape[0] * (1 - current_overall_reduction_target))
                         num_to_keep_globally = max(num_to_keep_globally, 
                                                   getattr(args, 'min_samples_after_global_prune', 
                                                           int(all_scores_concatenated_np.shape[0]*0.1))) # 最低10%は残すなど

                    print(f"CCS: Total points for selection={all_scores_concatenated_np.shape[0]}, Num to keep globally={num_to_keep_globally}")

                    # client_keep にCCSの結果を代入 (元コードの client_keep を使用)
                    client_keep = i_ccs.update_and_select(
                        scores=all_scores_concatenated_np,
                        num_to_keep=num_to_keep_globally,
                        global_to_local=global_to_local_for_ccs,
                        keep_ratio_per_client=current_keep_ratio_per_client # 計算済みの比率
                    )
                    last_client_keep = client_keep.copy() # slackレポート用に保持

                    # ========== [3] (CCS実行した場合) データセット再構築 ==========
                    # print("Updating client datasets based on CCS results...")
                    for cid_update_ds in range(args.num_users):
                        # client_keep.get(cid, []) で、そのクライアントが保持すべき「ローカルインデックス」リスト取得
                        # このローカルインデックスは、compute_el2n_scores_4 で使われたデータセット内の0からのインデックス。
                        local_models[cid_update_ds].update_dataset(
                            client_keep.get(cid_update_ds, []), 
                            el2n=args.el2n # el2n引数は元々あったので維持
                        )
        # CCS 全体の終了
        t_ccs_all_end = time.monotonic()
        # 各クライアントに「同じ CCS 全体時間」を加算しておく
        for cid_el2n in range(args.num_users):
                ccs_times[cid_el2n] += (t_ccs_all_end - t_ccs_all_start)

        # ========== [4] 各クライアントのローカル学習 ==========
        local_weights, local_losses = [], [] 
        temp_client_raw_metrics_this_round = {} # NEW: このラウンドで学習したクライアントの生メトリクス
        t_train_start = time.monotonic()
        for cid_train in idxs_users: 
            t_train_start = time.monotonic()
            # print(f"Client {cid_train} starting local training...")
            w, loss, perf_metrics = local_models[cid_train].update_weights( # CHANGED: perf_metrics を受け取る
                model=copy.deepcopy(global_model),
                global_round=epoch
            )
            t_train_end = time.monotonic()
            train_times[cid_train] += (t_train_end - t_train_start)
            if perf_metrics['num_samples'] > 0: # 学習データがあった場合のみ
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                temp_client_raw_metrics_this_round[cid_train] = perf_metrics # NEW: メトリクス保存
                # (ログは元コードのものをベースに調整)
            # else:
                # print(f"Client {cid_train} had no samples, skipped training contribution.")


        # ========== [5] グローバルモデル更新 ==========
        if local_weights: 
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            loss_avg_this_round = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg_this_round)
        # (学習参加クライアントなしの場合の処理は前回の提案通り)
        elif train_loss: 
            loss_avg_this_round = train_loss[-1] 
            train_loss.append(loss_avg_this_round)
        else: 
            loss_avg_this_round = float('nan')
            train_loss.append(loss_avg_this_round)

        # ========== [6] クライアントごとの平滑化メトリクス更新 ==========
        # NEW: (実際に学習に参加したクライアントに対してのみ更新)
        # print("Updating smoothed metrics for clients that participated...")
        for cid_metrics_update, raw_metrics in temp_client_raw_metrics_this_round.items():
            if raw_metrics['num_samples'] > 0:
                current_time_per_sample = raw_metrics['train_time'] / raw_metrics['num_samples']
            else:
                current_time_per_sample = float('inf') 
            current_process_cpu_usage = raw_metrics['process_cpu_usage']

            # time_per_sample の平滑化
            if client_smoothed_metrics[cid_metrics_update]['time_per_sample'] is None or \
               np.isinf(client_smoothed_metrics[cid_metrics_update]['time_per_sample']): 
                client_smoothed_metrics[cid_metrics_update]['time_per_sample'] = current_time_per_sample
            else: 
                client_smoothed_metrics[cid_metrics_update]['time_per_sample'] = \
                    args.burden_alpha * current_time_per_sample + \
                    (1 - args.burden_alpha) * client_smoothed_metrics[cid_metrics_update]['time_per_sample']

            # process_cpu_usage の平滑化
            if client_smoothed_metrics[cid_metrics_update]['process_cpu_usage'] is None:
                client_smoothed_metrics[cid_metrics_update]['process_cpu_usage'] = current_process_cpu_usage
            else:
                client_smoothed_metrics[cid_metrics_update]['process_cpu_usage'] = \
                    args.burden_alpha * current_process_cpu_usage + \
                    (1 - args.burden_alpha) * client_smoothed_metrics[cid_metrics_update]['process_cpu_usage']
            client_smoothed_metrics[cid_metrics_update]['last_updated_round'] = epoch
        # print(f"Smoothed metrics after round {epoch+1}: {client_smoothed_metrics}")

        # ========== [7] 評価 & ログ ==========
        list_acc_this_round = [] # このラウンドの全クライアントのテスト精度を格納
        list_loss_this_round = [] # このラウンドの全クライアントのテストロスを格納 (必要であれば)

        global_model.eval() # グローバルモデルを評価モードに

        for client_idx_eval in range(args.num_users):
            # LocalUpdate.inference は (accuracy, loss) を返すと仮定
            accuracy_eval, loss_eval = local_models[client_idx_eval].inference(model=global_model)
            list_acc_this_round.append(accuracy_eval)
            list_loss_this_round.append(loss_eval) # テストロスも収集する場合

        #このラウンドの平均精度と平均ロス
        current_avg_accuracy = np.mean(list_acc_this_round) if list_acc_this_round else 0.0
        current_avg_loss_on_test = np.mean(list_loss_this_round) if list_loss_this_round else float('nan') # 必要に応じて

        train_accuracy.append(current_avg_accuracy) # 全ラウンドの平均精度リストに追加
        total_ccs   = sum(ccs_times.values())/num_users
        total_train = sum(train_times.values())
        # ターミナルへの出力 (loss_avg_this_round はローカル学習の平均ロス)
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss (avg local losses): {loss_avg_this_round:.4f}') # このラウンドの平均ローカル学習ロス
        print(f'Test Accuracy (avg over client test sets): {100*current_avg_accuracy:.2f}%')
        print(f'Test Loss (avg over client test sets): {current_avg_loss_on_test:.4f}') # 必要に応じてテストロスも出力
        # ログファイルへの記録 (log_metrics 関数を使用)
        # train_time は、このラウンド開始時からの経過時間ではなく、
        # グローバルな学習開始 (start_time) からの総経過時間
        round_wall_time = time.time() - start_time # このラウンド終了までの総壁時計時間

        # log_metrics に渡す accuracies は、このラウンドの全クライアントの精度リスト
        mem_mb_log, model_kb_log = log_metrics(
            round_num=epoch + 1,
            train_time=round_wall_time,
            model=global_model,
            accuracies=list_acc_this_round, # このラウンドの精度リスト
            log_path="logs" # args.log_dir などで指定できるようにしても良い
        )
        mem_history.append(mem_mb_log)
        size_history.append(model_kb_log)


    # Test inference after completion of training
    test_acc, test_loss_on_test_set = test_inference(args, global_model, test_dataset) # CHANGED: 変数名
    end_time = time.time()
    overall_wall_clock_time = end_time - start_time # CHANGED: 変数名
    print(f' \n Results after {epoch + 1} global rounds of training:') # CHANGED: global_round -> epoch + 1
    # train_accuracyリストの最後の要素が最終的な平均精度
    if train_accuracy: # train_accuracyが空でないことを確認
        print("|---- Avg Train Accuracy (on client test sets): {:.2f}%".format(100*train_accuracy[-1]))
    else:
        print("|---- Avg Train Accuracy (on client test sets): N/A")
    print("|---- Test Accuracy (on global test set): {:.2f}%".format(100*test_acc))
    print(f"Total Client Train Time (sum of reported from LocalUpdate): {total_train:.3f}s") # CHANGED
    print(f"Total CCS Related Time (approx from old logic): {total_ccs:.3f}s") # CHANGED
    print(f"Total Wall-clock Time: {overall_wall_clock_time:.3f}s")


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
        f"total time = : {overall_wall_clock_time:.3f}s",
        "",
        "各クライアントデータセット:"
        f"Relevant CCS Params: start_prune_round={getattr(args, 'start_prune_round', 10)}, prune_interval={getattr(args, 'prune_interval', 10)}, prune_rate={args.prune_rate}",
        f"Burden Params: w_time={args.burden_w_time}, w_cpu={args.burden_w_cpu}, w_ram={getattr(args, 'burden_w_ram', 0.3)}, min_r={args.min_keep_ratio}, max_r={args.max_keep_ratio}",
        "",
        f"Results after {epoch + 1} global rounds of training:", # CHANGED
        ""
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
    if train_accuracy:
        report_lines.append(f"|---- Avg Train Accuracy (client test): {100*train_accuracy[-1]:.2f}%")
    if train_loss: # train_lossが空でないことを確認
        report_lines.append(f"|---- Final Avg Local Training Loss : {train_loss[-1] if train_loss else float('nan'):.4f}") # CHANGED: 最終ラウンドの平均ローカルロス
    report_lines.append(f"|---- Global Test Accuracy            : {100*test_acc:.2f}%")
    report_lines.append(f"|---- Total Client Train Time         : {total_train:.3f}s")
    report_lines.append(f"|---- Total CCS Related Time (approx) : {total_ccs:.3f}s")
    report_lines.append(f"|---- Total Wall-clock Time           : {overall_wall_clock_time:.3f}s")
    report_lines.append("")
    report_lines.append("Data points per client after final CCS (if CCS occurred):")

    # 最終的な report
    report = "\n".join(report_lines)
    slackPost(report)
    # Saving the objects train_loss and train_accuracy:
    # CHANGED: ファイル名をより詳細に、argsの値を多く含めるように変更
    # また、el2n引数よりもCCS関連の主要パラメータを含める方が適切かもしれない
    filename_suffix = f"dataset[{args.dataset}]_model[{args.model}]_epochs[{args.epochs}]_" \
                      f"users[{args.num_users}]_frac[{args.frac}]_iid[{args.iid}]_" \
                      f"E[{args.local_ep}]_B[{args.local_bs}]_lr[{args.lr}]_seed[{args.seed}]"
    
    # CCS関連のパラメータもファイル名に含めると管理しやすい
    if do_prune: # CCSが実行された可能性がある場合のみCCSパラメータをファイル名に
        filename_suffix += f"_ccs[start{getattr(args, 'start_prune_round', 10)}_int{getattr(args, 'prune_interval', 10)}_rate{args.prune_rate}]"
        filename_suffix += f"_burden[t{args.burden_w_time}_c{args.burden_w_cpu}_r{getattr(args, 'burden_w_ram',0.3)}_min{args.min_keep_ratio}]"

    file_name = f'../save/objects/{filename_suffix}.pkl'
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # 保存するオブジェクトを明確化
    # keep_idxs, prun_num, pri_pru などは元コードの特定のel2n分岐のロジックに依存していたため、
    # 新しいCCSロジックでは last_client_keep や client_smoothed_metrics を保存する方が適切かもしれない。
    # ここでは、元コードの保存形式を参考にしつつ、主要な結果を保存。
    objects_to_save = {
        'args': args, # 実験設定の保存
        'train_loss_rounds': train_loss,
        'train_accuracy_rounds': train_accuracy,
        'global_test_accuracy': test_acc,
        'global_test_loss': test_loss_on_test_set,
        'client_smoothed_metrics_final': client_smoothed_metrics,
        'last_client_keep_indices': dict(last_client_keep), # defaultdictを通常のdictに変換
        'wall_clock_time': overall_wall_clock_time,
        'mem_history': mem_history,
        'size_history': size_history
    }

    try:
        with open(file_name, 'wb') as f:
            pickle.dump(objects_to_save, f)
        print(f"Results saved to {file_name}")
    except Exception as e: # CHANGED: FileNotFoundError だけでなく一般的な例外をキャッチ
        print(f"Error saving results: {e}")
        print(f"Failed to save file at {file_name}.")


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
    matplotlib.use('Agg') # サーバーサイドなどGUIなし環境用

    if train_accuracy: # CHANGED: 空でないか確認
        plt.figure()
        plt.plot(range(len(train_accuracy)), train_accuracy, color='k', marker='o', linestyle='-') # CHANGED: マーカーと線種追加
        plt.title(f'Avg Client Test Accuracy vs. Global Rounds\n{args.dataset} - {args.model}') # CHANGED: タイトル追加
        plt.ylabel('Average Client Test Accuracy')
        plt.xlabel('Communication Rounds')
        plt.grid(True) # CHANGED: グリッド追加
        # CHANGED: ファイル名をより詳細に
        plot_filename = f'../save/plot_acc_{filename_suffix}.png'
        plt.savefig(plot_filename)
        print(f"Accuracy plot saved to {plot_filename}")
        plt.close() # CHANGED: メモリ解放のために閉じる

    if train_loss: # CHANGED: 空でないか確認
        # nanを除外してプロット (loss_avg_this_roundがnanになる場合があるため)
        valid_train_loss = [loss for loss in train_loss if not np.isnan(loss)]
        if valid_train_loss:
            plt.figure()
            plt.plot(range(len(valid_train_loss)), valid_train_loss, color='r', marker='x', linestyle='--') # CHANGED
            plt.title(f'Avg Local Training Loss vs. Global Rounds\n{args.dataset} - {args.model}') # CHANGED
            plt.ylabel('Average Local Training Loss')
            plt.xlabel('Communication Rounds')
            plt.grid(True) # CHANGED
            # CHANGED: ファイル名をより詳細に
            plot_filename_loss = f'../save/plot_loss_{filename_suffix}.png'
            plt.savefig(plot_filename_loss)
            print(f"Loss plot saved to {plot_filename_loss}")
            plt.close() # CHANGED
