#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
# import sympy as sp # 未使用のためコメントアウト
import subprocess # NEW: subprocess をインポート

import torch
import torchvision
# from torchvision.models import resnet18, ResNet18_Weights # resnet以外では未使用ならコメントアウト可
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference # LocalUpdate は client_task.py 内で使われる
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

# from sklearn.preprocessing import StandardScaler # 未使用のためコメントアウト
from ccs_utils import IncrementalCCS
import psutil
import json
from torch.utils.data import DataLoader
from collections import defaultdict
import slackweb # SLACKURLが未定義、または設定が必要なら有効化

# Slack Webhook URL (必要であれば設定)
SLACKURL = 'YOUR_SLACK_WEBHOOK_URL_HERE'

def slackPost(message):
    if 'SLACKURL' in globals() and SLACKURL != 'YOUR_SLACK_WEBHOOK_URL_HERE':
        slack = slackweb.Slack(url=SLACKURL)
        slack.notify(text=message)
    else:
        print("Slack URL not configured. Message not sent.")

def log_metrics(round_num, current_wall_time, model, accuracies, log_path="logs"): # CHANGED: train_time -> current_wall_time
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    model_size = len(pickle.dumps(model.state_dict())) / 1024
    accuracy_mean = np.mean(accuracies) if accuracies else 0.0 # CHANGED: accuracies が空の場合の処理
    accuracy_std = np.std(accuracies) if accuracies else 0.0   # CHANGED

    log_data = {
        "round": round_num,
        "wall_time_sec": current_wall_time, # CHANGED
        "memory_MB": memory_usage,
        "comm_KB": model_size,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std
    }
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, f"round_{round_num}.json"), "w") as f:
        json.dump(log_data, f, indent=4)
    return memory_usage, model_size

# calculate_burden_aware_keep_ratios 関数 (前回提案のものをベースに、RAMメトリクス名変更の可能性を考慮)
def calculate_burden_aware_keep_ratios(
    all_client_ids, client_smoothed_metrics, system_ram_pressure_percent,
    w_time, w_cpu, w_ram, # w_ram はクライアント個別RAM用かシステムRAM用か明確に
    min_r, max_r, default_r
):
    output_ratios = {cid: default_r for cid in all_client_ids}
    clients_with_metrics_ids = [
        cid for cid in all_client_ids 
        if cid in client_smoothed_metrics and \
           client_smoothed_metrics[cid].get('time_per_sample') is not None and \
           not np.isinf(client_smoothed_metrics[cid].get('time_per_sample')) and \
           client_smoothed_metrics[cid].get('process_cpu_usage') is not None and \
           client_smoothed_metrics[cid].get('client_ram_rss_mb') is not None # NEW: クライアントRAMも考慮
    ]
    if not clients_with_metrics_ids: return output_ratios

    times_values = np.array([client_smoothed_metrics[cid]['time_per_sample'] for cid in clients_with_metrics_ids])
    process_cpus_values = np.array([client_smoothed_metrics[cid]['process_cpu_usage'] for cid in clients_with_metrics_ids])
    client_ram_values = np.array([client_smoothed_metrics[cid]['client_ram_rss_mb'] for cid in clients_with_metrics_ids]) # NEW

    # 正規化 (0-1, 高いほど高負担)
    norm_times = np.full_like(times_values, 0.5, dtype=float)
    if times_values.size > 0:
        min_t, max_t = np.min(times_values), np.max(times_values)
        if times_values.size > 1 and (max_t - min_t) > 1e-8: norm_times = (times_values - min_t) / (max_t - min_t + 1e-8)
    
    norm_process_cpus = np.full_like(process_cpus_values, 0.5, dtype=float)
    if process_cpus_values.size > 0:
        min_cpu, max_cpu = np.min(process_cpus_values), np.max(process_cpus_values)
        if process_cpus_values.size > 1 and (max_cpu - min_cpu) > 1e-8: norm_process_cpus = (process_cpus_values - min_cpu) / (max_cpu - min_cpu + 1e-8)

    # クライアント個別RAMの正規化 (高いほど高負担) -> RSSが高いほど高負担なのでそのままで良い
    norm_client_ram = np.full_like(client_ram_values, 0.5, dtype=float) # NEW
    if client_ram_values.size > 0: # NEW
        min_ram, max_ram = np.min(client_ram_values), np.max(client_ram_values) # NEW
        if client_ram_values.size > 1 and (max_ram - min_ram) > 1e-8: # NEW
            norm_client_ram = (client_ram_values - min_ram) / (max_ram - min_ram + 1e-8) # NEW
            
    # norm_system_ram = system_ram_pressure_percent / 100.0 # システムRAMは別途考慮も可

    burden_scores_list = []
    for i in range(len(clients_with_metrics_ids)):
        # CHANGED: norm_client_ram を使用し、w_ram がクライアント個別RAMの重みを指すように変更
        # system_ram_pressure_percent は別途全体的なペナルティとして使うか、このスコアに含めない場合は w_ram=0 と同等
        score = (w_time * norm_times[i] + 
                 w_cpu * norm_process_cpus[i] + 
                 w_ram * norm_client_ram[i]) # w_ramはクライアントRAMの重み
        # もしシステムRAMもスコアに含めるなら、別途 w_sys_ram * norm_system_ram を加える
        burden_scores_list.append(score)
        
    burden_scores = np.clip(np.array(burden_scores_list), 0.0, 1.0) # 重み合計が1を超える場合もあるのでクリップ
    calculated_ratios = np.clip(max_r - burden_scores * (max_r - min_r), min_r, max_r)
    for i, cid in enumerate(clients_with_metrics_ids): output_ratios[cid] = calculated_ratios[i]
    return output_ratios


if __name__ == '__main__':
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    args = args_parser()
    exp_details(args)
    
    # NEW: client_cpu_limits_str を辞書にパース
    args.client_cpu_limits_dict = {}
    if hasattr(args, 'client_cpu_limits_str') and args.client_cpu_limits_str:
        try:
            import ast
            # 文字列内のキーが整数であることを保証するために少し工夫が必要な場合がある
            # 例: "{0:20, 1:50}" のような形式を期待
            args.client_cpu_limits_dict = ast.literal_eval(args.client_cpu_limits_str)
            if not isinstance(args.client_cpu_limits_dict, dict):
                args.client_cpu_limits_dict = {}; print("Warning: client_cpu_limits_str was not parsed to a dict.")
        except Exception as e:
            print(f"Warning: Could not parse client_cpu_limits_str ('{args.client_cpu_limits_str}'): {e}. Using defaults.")

    device_to_use = 'cpu' # デフォルトはCPU
    if hasattr(args, 'gpu') and args.gpu is not None:
        if isinstance(args.gpu, str) and args.gpu.lower() != 'none' and args.gpu.lower() != 'cpu':
            try:
                # "cuda:0" のような文字列から数値部分を抽出
                gpu_id_str = args.gpu.split(':')[-1]
                gpu_id = int(gpu_id_str)
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    torch.cuda.set_device(gpu_id) # 整数で指定
                    device_to_use = f'cuda:{gpu_id}'
                else:
                    print(f"Warning: GPU id {gpu_id} is not available. Found {torch.cuda.device_count()} GPUs. Using CPU.")
                    # args.gpu = None # CPU使用にフォールバックする場合
            except ValueError:
                print(f"Warning: Invalid GPU format '{args.gpu}'. Expected format like '0', '1', or 'cuda:0'. Using CPU.")
                # args.gpu = None
        elif isinstance(args.gpu, int): # 既に整数の場合
            if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
                torch.cuda.set_device(args.gpu)
                device_to_use = f'cuda:{args.gpu}'
            else:
                print(f"Warning: GPU id {args.gpu} is not available. Using CPU.")
                # args.gpu = None
    
    # device 変数には最終的に 'cuda:X' または 'cpu' が入る
    device = torch.device(device_to_use) 
    # print(f"Script using device: {device}")

    # LocalUpdateクラスの__init__内や、モデルを .to(device) する箇所でも同様の注意が必要
    # LocalUpdateのコンストラクタで args.gpu を渡す場合、LocalUpdate内でも同様のパース処理を行うか、
    # メインスクリプト側でパース済みの device オブジェクトを渡すようにする。
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print("num of train dataset = ", len(train_dataset))
    print("num of test dataset = ", len(test_dataset))

    # BUILD MODEL (変更なし)
    if args.model == 'cnn':
        if args.dataset == 'mnist': global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist': global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar': global_model = torchvision.models.resnet18(num_classes=args.num_classes)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape; len_in = 1; [len_in := len_in * x for x in img_size]
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else: exit('Error: unrecognized model')
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    # Training lists and metrics
    train_loss, train_accuracy = [], []
    mem_history, size_history = [], []
    choice_users = list(range(args.num_users))
    last_client_keep = defaultdict(list)

    # NEW: クライアントごとの平滑化メトリクス用ストレージ (RAM情報も追加)
    client_smoothed_metrics = {
        cid: {'time_per_sample': None, 'process_cpu_usage': None, 
              'client_ram_rss_mb': None, 'last_updated_round': -1}
        for cid in range(args.num_users)
    }

    # local_models は、各クライアントの現在のデータセット状態を管理するために依然として使用
    # ただし、学習自体はサブプロセスで行う
    local_models = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], 
                                logger=logger, client_id=idx) for idx in range(args.num_users)]
    
    i_ccs = IncrementalCCS(num_groups=getattr(args, 'num_groups_ccs', 100), seed=args.seed)
    
    # NEW: 時間集計用変数
    cumulative_total_client_train_time = 0.0
    cumulative_total_ccs_processing_time = 0.0

    # NEW: 一時ファイル保存用ディレクトリ (argsから取得)
    TEMP_CLIENT_IO_DIR = getattr(args, 'temp_io_dir', './client_io_temp_run_ccs_default')
    os.makedirs(TEMP_CLIENT_IO_DIR, exist_ok=True)

    start_time = time.time() # 実験全体の開始時間

    for epoch in tqdm(range(args.epochs)):
        print(f"\n | Global Training Round : {epoch+1} |\n")
        global_model.train()
        
        m = max(int(args.frac * args.num_users), 1)
        if not choice_users : break
        idxs_users = np.random.choice(choice_users, min(m, len(choice_users)), replace=False) # CHANGED: 選択可能数を超えないように
        print("Selected users for training this round:", idxs_users)
        
        start_prune_round = getattr(args, 'start_prune_round', 10) 
        prune_interval = getattr(args, 'prune_interval', 10)    
        do_prune = (epoch >= (start_prune_round -1) and \
                    (epoch - (start_prune_round - 1)) % prune_interval == 0)

        current_keep_ratio_per_client = {cid: args.default_keep_ratio for cid in range(args.num_users)}
        round_ccs_processing_duration = 0.0

        if do_prune:
            t_ccs_round_start = time.monotonic() # CCS関連処理の時間計測開始
            system_ram_pressure = psutil.virtual_memory().percent 
            current_keep_ratio_per_client = calculate_burden_aware_keep_ratios(
                all_client_ids=list(range(args.num_users)),
                client_smoothed_metrics=client_smoothed_metrics,
                system_ram_pressure_percent=system_ram_pressure, # システム全体のRAM負荷
                w_time=args.burden_w_time, 
                w_cpu=args.burden_w_cpu,   
                w_ram=args.burden_w_ram, # クライアント個別RAMの重み (calculate関数内で使用)
                min_r=args.min_keep_ratio, 
                max_r=args.max_keep_ratio, 
                default_r=args.default_keep_ratio
            )
            
            all_scores_list_for_ccs, global_to_local_for_ccs = [], []
            for cid_el2n in range(args.num_users):
                if local_models[cid_el2n].trainloader and len(local_models[cid_el2n].trainloader.dataset) > 0:
                    scores_tensor = local_models[cid_el2n].compute_el2n_scores_4(global_model)
                    if scores_tensor.numel() > 0:
                        scores_np_for_client = scores_tensor.numpy()
                        all_scores_list_for_ccs.append(scores_np_for_client)
                        for local_idx in range(len(scores_np_for_client)):
                            global_to_local_for_ccs.append((cid_el2n, local_idx))
            
            if all_scores_list_for_ccs and global_to_local_for_ccs:
                all_scores_concatenated_np = np.concatenate(all_scores_list_for_ccs, axis=0)
                if all_scores_concatenated_np.size > 0:
                    num_to_keep_globally = all_scores_concatenated_np.shape[0]
                    if epoch >= (start_prune_round -1) :
                         prune_level_idx = (epoch - (start_prune_round - 1)) // prune_interval
                         current_overall_reduction_target = min(args.prune_rate * (prune_level_idx +1), getattr(args, 'max_overall_reduction_rate', 0.8))
                         num_to_keep_globally = int(all_scores_concatenated_np.shape[0] * (1 - current_overall_reduction_target))
                         num_to_keep_globally = max(num_to_keep_globally, getattr(args, 'min_samples_after_global_prune', max(1, int(all_scores_concatenated_np.shape[0]*0.1))))
                    
                    client_keep_indices_this_round = i_ccs.update_and_select(
                        scores=all_scores_concatenated_np, num_to_keep=num_to_keep_globally,
                        global_to_local=global_to_local_for_ccs, keep_ratio_per_client=current_keep_ratio_per_client
                    )
                    last_client_keep = client_keep_indices_this_round.copy()
                    for cid_update_ds in range(args.num_users):
                        local_models[cid_update_ds].update_dataset(
                            client_keep_indices_this_round.get(cid_update_ds, []), el2n=args.el2n
                        )
            round_ccs_processing_duration = time.monotonic() - t_ccs_round_start
            cumulative_total_ccs_processing_time += round_ccs_processing_duration
            print(f"Round {epoch+1} CCS processing time: {round_ccs_processing_duration:.3f}s")


        # ========== [4] 各クライアントのローカル学習 (サブプロセス化) ==========
        local_weights_this_round = []
        local_losses_this_round = []
        temp_client_raw_metrics_this_round = {}

        # 現在のグローバルモデルのstate_dictを一時ファイルに保存
        # TEMP_CLIENT_IO_DIR は事前に定義・作成されていること (例: getattr(args, 'temp_io_dir', './client_io_temp_run'))
        global_model_path_for_clients = os.path.join(TEMP_CLIENT_IO_DIR, f'gmodel_r{epoch}.pt')
        try:
            torch.save(global_model.state_dict(), global_model_path_for_clients)
        except Exception as e:
            print(f"FATAL: Could not save global model for round {epoch}: {e}")
            break 

        active_client_processes = []
        cpulimit_processes_to_manage = []

        for client_id_to_train in idxs_users:
            try:
                current_client_indices = local_models[client_id_to_train].client_specific_global_idxs
                if not current_client_indices:
                    # print(f"Client {client_id_to_train} has no data indices for round {epoch}, skipping subprocess.") # DEBUG
                    continue
            except AttributeError:
                print(f"FATAL: local_models[{client_id_to_train}] does not have 'client_specific_global_idxs'. Ensure LocalUpdate is correctly initialized.")
                continue 
            except IndexError:
                 print(f"FATAL: client_id_to_train {client_id_to_train} out of range for local_models. Skipping.")
                 continue


            client_data_indices_path = os.path.join(TEMP_CLIENT_IO_DIR, f'cdata_{client_id_to_train}_r{epoch}.pkl')
            try:
                with open(client_data_indices_path, 'wb') as f:
                    pickle.dump(current_client_indices, f)
            except Exception as e:
                print(f"Error saving data indices for client {client_id_to_train} round {epoch}: {e}")
                continue

            client_script_to_run_path = getattr(args, 'client_script_path', 'src/update_sub.py') # options.pyでのデフォルトを期待

            cmd_list = ['python3', client_script_to_run_path,
                        f'--client_id={client_id_to_train}', 
                        f'--global_model_path={global_model_path_for_clients}',
                        f'--current_epoch={epoch}', 
                        f'--data_indices_path={client_data_indices_path}',
                        f'--output_dir={TEMP_CLIENT_IO_DIR}']
            
            arg_dict = vars(args)
            # サブプロセスに渡すべきでない、またはメインプロセス側で制御する引数リスト
            excluded_args_for_client_script = [
                'help', 'client_id', 'global_model_path', 'current_epoch', 
                'data_indices_path', 'output_dir', 'client_script_path', 
                'temp_io_dir', 
                'client_cpu_limits_str', # 文字列版は渡さない
                'client_cpu_limits_dict', # CHANGED: パース済みの辞書も渡さない (サブプロセスは自身のIDのみ知れば良い)
                'num_users', 'frac', 'epochs', 
                'burden_alpha', 'burden_w_time', 'burden_w_cpu', 'burden_w_ram',
                'min_keep_ratio', 'max_keep_ratio', 'default_keep_ratio',
                'start_prune_round', 'prune_interval', 
                'max_overall_reduction_rate', 'min_samples_after_global_prune',
                'num_groups_ccs',
                # args.el2n に応じた分岐で使われていた引数も、その分岐がメイン側なら除外
                'threshold', 'percent', 'acc_thre1', 'acc_thre2', 'acc_thre3', 
                'start_accuracy', 'pru_percent' 
            ]

            for arg_name, arg_val in arg_dict.items():
                if arg_name not in excluded_args_for_client_script:
                    if isinstance(arg_val, bool):
                        if arg_val: 
                            cmd_list.append(f'--{arg_name}')
                    elif arg_val is not None:
                        cmd_list.append(f'--{arg_name}={str(arg_val)}')
            
            try:
                # print(f"Round {epoch} Client {client_id_to_train} CMD: {' '.join(cmd_list)}") # DEBUG
                client_proc = subprocess.Popen(cmd_list)
                active_client_processes.append({'pid': client_proc.pid, 'process': client_proc, 'cid': client_id_to_train})
                
                cpu_limit_val = args.default_client_cpu_limit
                # args.client_cpu_limits_dict はメインスクリプトの最初で文字列からパース済みと仮定
                if hasattr(args, 'client_cpu_limits_dict') and client_id_to_train in args.client_cpu_limits_dict:
                    cpu_limit_val = args.client_cpu_limits_dict[client_id_to_train]
                
                if cpu_limit_val < 100 and os.name == 'posix':
                    # cpulimitコマンドが存在するか確認
                    if subprocess.call("command -v cpulimit >/dev/null 2>&1", shell=True, executable='/bin/bash') == 0:
                        cpulimit_cmd = ['cpulimit', '-p', str(client_proc.pid), '-l', str(cpu_limit_val), '-b', '-q']
                        # print(f"Applying cpulimit to PID {client_proc.pid} with limit {cpu_limit_val}%") # DEBUG
                        cpulimit_subproc = subprocess.Popen(cpulimit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        cpulimit_processes_to_manage.append(cpulimit_subproc)
                    else:
                        print(f"Warning: cpulimit command not found. Client {client_id_to_train} will run without CPU limit.")
                elif cpu_limit_val < 100:
                    print(f"Warning: cpulimit is typically for POSIX systems. Client {client_id_to_train} may run without CPU limit on OS '{os.name}'.")

            except FileNotFoundError as e: 
                print(f"Error starting client {client_id_to_train} (FileNotFound for '{client_script_to_run_path}' or 'python3'): {e}.")
            except Exception as e: 
                 print(f"Error launching or applying cpulimit for client {client_id_to_train}: {e}")

        # 全ての起動したクライアントサブプロセスの終了を待つ
        for client_info in active_client_processes:
            try:
                # print(f"Waiting for client {client_info['cid']} (PID {client_info['pid']})...") # DEBUG
                client_info['process'].wait(timeout=getattr(args, 'client_timeout', 600)) # options.pyで定義
                
                client_output_filename = f'client_{client_info["cid"]}_round_{epoch}_output.pkl'
                client_output_path = os.path.join(TEMP_CLIENT_IO_DIR, client_output_filename)
                
                if os.path.exists(client_output_path):
                    with open(client_output_path, 'rb') as f: results = pickle.load(f)
                    
                    if 'perf_metrics' in results and results['perf_metrics'].get('num_samples', 0) > 0:
                        local_weights_this_round.append(results['weights'])
                        local_losses_this_round.append(results['loss'])
                        temp_client_raw_metrics_this_round[results['client_id']] = results['perf_metrics']
                    elif 'perf_metrics' in results:
                        # print(f"Client {results.get('client_id', client_info['cid'])} reported 0 samples or missing num_samples in perf_metrics.") # DEBUG
                        # エラー情報が含まれている可能性があるので、メトリクスは保存しておく
                        temp_client_raw_metrics_this_round[results.get('client_id', client_info['cid'])] = results['perf_metrics'] 
                    else:
                        print(f"Client {results.get('client_id', client_info['cid'])}: perf_metrics not found in results. Output: {results}")
                    
                    try:
                        os.remove(client_output_path)
                    except OSError as e_rm:
                        print(f"Warning: Could not remove client output file {client_output_path}: {e_rm}")
                else:
                    print(f"Output file not found for client {client_info['cid']} (PID {client_info['pid']}). Client might have failed or timed out before saving.")

            except subprocess.TimeoutExpired:
                print(f"Client {client_info['cid']} (PID {client_info['pid']}) timed out. Killing process.")
                client_info['process'].kill(); client_info['process'].wait()
            except FileNotFoundError: 
                print(f"Output file for client {client_info['cid']} (PID {client_info['pid']}) was unexpectedly missing after wait.")
            except Exception as e:
                print(f"Error processing results or waiting for client {client_info['cid']} (PID {client_info['pid']}): {e}")
                if client_info['process'].poll() is None: client_info['process'].kill(); client_info['process'].wait()
            
            client_data_indices_path_to_remove = os.path.join(TEMP_CLIENT_IO_DIR, f'cdata_{client_info["cid"]}_r{epoch}.pkl')
            if os.path.exists(client_data_indices_path_to_remove):
                try: os.remove(client_data_indices_path_to_remove)
                except OSError as e_rm_idx: print(f"Warning: Could not remove client data index file {client_data_indices_path_to_remove}: {e_rm_idx}")

        # cpulimitプロセスを終了させる
        for cp_lim_proc in cpulimit_processes_to_manage:
            if cp_lim_proc.poll() is None: 
                try: cp_lim_proc.terminate(); cp_lim_proc.wait(timeout=1)
                except: cp_lim_proc.kill()
        
        if os.path.exists(global_model_path_for_clients):
            try: os.remove(global_model_path_for_clients)
            except OSError as e_rm_gmodel: print(f"Warning: Could not remove global model temp file {global_model_path_for_clients}: {e_rm_gmodel}")

        
        
        # ========== [5] グローバルモデル更新 ==========
        if local_weights_this_round:
            global_weights = average_weights(local_weights_this_round)
            global_model.load_state_dict(global_weights)
            loss_avg_this_round = sum(local_losses_this_round) / len(local_losses_this_round)
            train_loss.append(loss_avg_this_round)
        elif train_loss: 
            loss_avg_this_round = train_loss[-1]; train_loss.append(loss_avg_this_round)
        else: 
            loss_avg_this_round = float('nan'); train_loss.append(loss_avg_this_round)

        # ========== [6] クライアントごとの平滑化メトリクス更新 ==========
        for cid_metrics_update, raw_metrics in temp_client_raw_metrics_this_round.items():
            cumulative_total_client_train_time += raw_metrics['train_time'] # NEW: 集計
            
            if raw_metrics['num_samples'] > 0:
                current_time_per_sample = raw_metrics['train_time'] / raw_metrics['num_samples']
            else: current_time_per_sample = float('inf') 
            current_process_cpu_usage = raw_metrics['process_cpu_usage']
            current_client_ram_rss_mb = raw_metrics.get('client_ram_rss_mb', None)

            alpha = args.burden_alpha
            # time_per_sample
            prev_tps = client_smoothed_metrics[cid_metrics_update].get('time_per_sample')
            if prev_tps is None or np.isinf(prev_tps): client_smoothed_metrics[cid_metrics_update]['time_per_sample'] = current_time_per_sample
            else: client_smoothed_metrics[cid_metrics_update]['time_per_sample'] = alpha * current_time_per_sample + (1 - alpha) * prev_tps
            # process_cpu_usage
            prev_cpu = client_smoothed_metrics[cid_metrics_update].get('process_cpu_usage')
            if prev_cpu is None: client_smoothed_metrics[cid_metrics_update]['process_cpu_usage'] = current_process_cpu_usage
            else: client_smoothed_metrics[cid_metrics_update]['process_cpu_usage'] = alpha * current_process_cpu_usage + (1 - alpha) * prev_cpu
            # client_ram_rss_mb
            if current_client_ram_rss_mb is not None:
                prev_ram = client_smoothed_metrics[cid_metrics_update].get('client_ram_rss_mb')
                if prev_ram is None: client_smoothed_metrics[cid_metrics_update]['client_ram_rss_mb'] = current_client_ram_rss_mb
                else: client_smoothed_metrics[cid_metrics_update]['client_ram_rss_mb'] = alpha * current_client_ram_rss_mb + (1 - alpha) * prev_ram
            client_smoothed_metrics[cid_metrics_update]['last_updated_round'] = epoch
        
        # ========== [7] 評価 & ログ (前回提示の修正済みコードをここに) ==========
        list_acc_this_round = []
        global_model.eval()
        for client_idx_eval in range(args.num_users):
            accuracy_eval, _ = local_models[client_idx_eval].inference(model=global_model) # loss_evalは使わないなら_
            list_acc_this_round.append(accuracy_eval)
        current_avg_accuracy = np.mean(list_acc_this_round) if list_acc_this_round else 0.0
        train_accuracy.append(current_avg_accuracy)
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss (avg local losses): {loss_avg_this_round:.4f}')
        print(f'Test Accuracy (avg over client test sets): {100*current_avg_accuracy:.2f}%')
        
        current_wall_time_log = time.time() - start_time
        mem_mb_log, model_kb_log = log_metrics(
            round_num=epoch + 1, current_wall_time=current_wall_time_log, # CHANGED
            model=global_model, accuracies=list_acc_this_round
        )
        mem_history.append(mem_mb_log); size_history.append(model_kb_log)

    # --- ループ終了後 ---
    # Test inference after completion of training
    test_acc, test_loss_on_test_set = test_inference(args, global_model, test_dataset)
    overall_wall_clock_time = time.time() - start_time

    # CHANGED: 実際の集計値を使用
    final_total_client_train_time = cumulative_total_client_train_time
    final_total_ccs_processing_time = cumulative_total_ccs_processing_time

    print(f' \n Results after {epoch + 1} global rounds of training:')
    if train_accuracy: print("|---- Avg Train Accuracy (on client test sets): {:.2f}%".format(100*train_accuracy[-1]))
    else: print("|---- Avg Train Accuracy (on client test sets): N/A")
    print("|---- Test Accuracy (on global test set): {:.2f}%".format(100*test_acc))
    print(f"Total Client Actual Train Time: {final_total_client_train_time:.3f}s")
    print(f"Total Actual CCS Processing Time: {final_total_ccs_processing_time:.3f}s")
    print(f"Total Wall-clock Time: {overall_wall_clock_time:.3f}s")

    # Slack 通知用レポート生成 (主要な時間情報を更新)
    report_lines = [
        "```",
        f"python src/new_ccs.py --dataset {args.dataset} --model {args.model} --epochs {args.epochs} --frac {args.frac} --lr {args.lr} ...",
        f"CCS Params: start_prune_round={getattr(args, 'start_prune_round', 10)}, interval={getattr(args, 'prune_interval', 10)}, rate={args.prune_rate}",
        f"Burden Params: w_t={args.burden_w_time}, w_cpu={args.burden_w_cpu}, w_ram={args.burden_w_ram}, min_r={args.min_keep_ratio}, max_r={args.max_keep_ratio}",
        f"CPU Limits used: {hasattr(args, 'client_cpu_limits_dict') and bool(args.client_cpu_limits_dict)}",
        "",
        f"Results after {epoch + 1} global rounds of training:",
        ""
    ]
    if train_accuracy: report_lines.append(f"|---- Avg Train Accuracy (client test): {100*train_accuracy[-1]:.2f}%")
    if train_loss and not np.isnan(train_loss[-1]): report_lines.append(f"|---- Final Avg Local Training Loss : {train_loss[-1]:.4f}")
    report_lines.append(f"|---- Global Test Accuracy            : {100*test_acc:.2f}%")
    report_lines.append(f"|---- Total Client Actual Train Time  : {final_total_client_train_time:.3f}s")
    report_lines.append(f"|---- Total Actual CCS Processing Time: {final_total_ccs_processing_time:.3f}s")
    report_lines.append(f"|---- Total Wall-clock Time           : {overall_wall_clock_time:.3f}s")
    report_lines.append("")
    report_lines.append("Data points per client after final CCS (if CCS occurred):")
    total_data_after_final_ccs = 0
    if last_client_keep:
        for cid_report in range(args.num_users):
            kept_count = len(last_client_keep.get(cid_report, []))
            report_lines.append(f"  Client {cid_report}: {kept_count}")
            total_data_after_final_ccs += kept_count
        report_lines.append(f"Total data points after final CCS: {total_data_after_final_ccs}")
    else: report_lines.append("  CCS was not performed or no data kept.")
    
    if mem_history:
        report_lines.append(f"|---- Avg Memory Usage (MB)         : {np.mean(mem_history):.1f}")
        report_lines.append(f"|---- Final Round Memory Usage (MB) : {mem_history[-1]:.1f}")
    if size_history:
        report_lines.append(f"|---- Avg Model Size (KB)           : {np.mean(size_history):.1f}")
        report_lines.append(f"|---- Final Round Model Size (KB)   : {size_history[-1]:.1f}")
    report_lines.append("```")
    report = "\n".join(report_lines)
    slackPost(report) # 必要なら有効化

    # 結果の保存
    filename_suffix = f"ds[{args.dataset}]_m[{args.model}]_ep[{args.epochs}]_usr[{args.num_users}]_fr[{args.frac}]_iid[{args.iid}]_lep[{args.local_ep}]_lbs[{args.local_bs}]_lr[{args.lr}]_sd[{args.seed}]"
    if do_prune: # 最後にdo_pruneがTrueだったか、あるいはCCSが一度でも実行されたかのフラグで判断
        filename_suffix += f"_ccs[s{start_prune_round}_i{prune_interval}_r{args.prune_rate}]_bd[t{args.burden_w_time}_c{args.burden_w_cpu}_r{args.burden_w_ram}_min{args.min_keep_ratio}]"
    file_name = f'../save/objects/{filename_suffix}.pkl'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    objects_to_save = {
        'args': args, 'train_loss_rounds': train_loss, 'train_accuracy_rounds': train_accuracy,
        'global_test_accuracy': test_acc, 'global_test_loss': test_loss_on_test_set,
        'client_smoothed_metrics_final': client_smoothed_metrics,
        'last_client_keep_indices': dict(last_client_keep),
        'wall_clock_time': overall_wall_clock_time,
        'total_client_train_time': final_total_client_train_time,
        'total_ccs_processing_time': final_total_ccs_processing_time,
        'mem_history': mem_history, 'size_history': size_history
    }
    try:
        with open(file_name, 'wb') as f: pickle.dump(objects_to_save, f)
        print(f"Results saved to {file_name}")
    except Exception as e: print(f"Error saving results: {e}")
    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg') # サーバーサイドなどGUIなし環境用
    # グラフ描画
    if train_accuracy:
        plt.figure(); plt.plot(range(len(train_accuracy)), train_accuracy, color='k', marker='o', linestyle='-')
        plt.title(f'Avg Client Test Acc vs. Rounds\n{args.dataset}-{args.model}'); plt.ylabel('Avg Client Test Acc'); plt.xlabel('Rounds'); plt.grid(True)
        plt.savefig(f'../save/plot_acc_{filename_suffix}.png'); plt.close()
    if train_loss:
        valid_train_loss = [loss for loss in train_loss if not np.isnan(loss)]
        if valid_train_loss:
            plt.figure(); plt.plot(range(len(valid_train_loss)), valid_train_loss, color='r', marker='x', linestyle='--')
            plt.title(f'Avg Local Train Loss vs. Rounds\n{args.dataset}-{args.model}'); plt.ylabel('Avg Local Train Loss'); plt.xlabel('Rounds'); plt.grid(True)
            plt.savefig(f'../save/plot_loss_{filename_suffix}.png'); plt.close()
            
    print("\nFederated training process has finished.")