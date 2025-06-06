#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import time
import psutil
import argparse
import pickle
import torchvision # torchvision.models を使う場合に備えてインポート

# --- モジュールのインポートとフォールバック ---
try:
    from options import args_parser as main_args_parser_for_defaults_setup
except ImportError:
    print("ERROR in update_sub.py: Could not import args_parser from options.py. "
          "Ensure options.py is in the Python path or the same directory.")
    def main_args_parser_for_defaults_setup(): # フォールバック
        parser = argparse.ArgumentParser(add_help=False)
        # options.py に定義されている主要な引数のデフォルト値をここで設定
        parser.add_argument('--gpu', default=None)
        parser.add_argument('--dataset', type=str, default='cifar')
        parser.add_argument('--model', type=str, default='cnn')
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--local_ep', type=int, default=3)
        parser.add_argument('--local_bs', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.03)
        parser.add_argument('--optimizer', type=str, default='sgd')
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--verbose', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_channels', type=int, default=3)
        parser.add_argument('--kernel_num', type=int, default=9)
        parser.add_argument('--kernel_sizes', type=str, default='3,4,5')
        parser.add_argument('--norm', type=str, default='batch_norm')
        parser.add_argument('--num_filters', type=int, default=32)
        parser.add_argument('--max_pool', type=str, default='True')
        parser.add_argument('--iid', type=int, default=1)
        parser.add_argument('--unequal', type=int, default=0)
        parser.add_argument('--dim_hidden', type=int, default=64)
        parser.add_argument('--num_per_client', type=int, default=10000)
        # エラーメッセージのusageに表示されていた他の引数も必要に応じて追加
        # options.py に定義されているが、上記フォールバックに含まれていないもの
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--num_users', type=int, default=10)
        parser.add_argument('--frac', type=float, default=0.5)
        parser.add_argument('--stopping_rounds', type=int, default=10)
        parser.add_argument('--el2n', type=int, default=5)
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--percent', type=float, default=0.5)
        parser.add_argument('--acc_thre1', type=float, default=0.7)
        parser.add_argument('--acc_thre2', type=float, default=0.75)
        parser.add_argument('--acc_thre3', type=float, default=0.8)
        parser.add_argument('--start_accuracy', type=float, default=0.6)
        parser.add_argument('--pru_percent', type=float, default=0.6)
        parser.add_argument('--prune_rate', type=float, default=0.1)
        # メインプロセス専用の引数 (サブプロセスでは使わないが、options.py に存在するなら定義しておく)
        parser.add_argument('--client_script_path', type=str, default='src/update_sub.py')
        parser.add_argument('--temp_io_dir', type=str, default='./client_io_temp_ccs')
        parser.add_argument('--client_cpu_limits_str', type=str, default='{}')
        parser.add_argument('--default_client_cpu_limit', type=int, default=100)
        parser.add_argument('--client_timeout', type=int, default=600)
        parser.add_argument('--burden_alpha', type=float, default=0.3)
        parser.add_argument('--burden_w_time', type=float, default=0.4)
        parser.add_argument('--burden_w_cpu', type=float, default=0.3)
        parser.add_argument('--burden_w_ram', type=float, default=0.3)
        parser.add_argument('--min_keep_ratio', type=float, default=0.2)
        parser.add_argument('--max_keep_ratio', type=float, default=1.0)
        parser.add_argument('--default_keep_ratio', type=float, default=0.8)
        parser.add_argument('--start_prune_round', type=int, default=10)
        parser.add_argument('--prune_interval', type=int, default=10)
        parser.add_argument('--max_overall_reduction_rate', type=float, default=0.6)
        parser.add_argument('--min_samples_after_global_prune', type=int, default=20)
        parser.add_argument('--num_groups_ccs', type=int, default=100)
        parser.add_argument('--client_id', type=int, required=True, help='Client ID')
        parser.add_argument('--global_model_path', type=str, required=True, help='Path to the global model')
        parser.add_argument('--current_epoch', type=int, required=True, help='Current training epoch')
        parser.add_argument('--data_indices_path', type=str, required=True, help='Path to client data indices')
        parser.add_argument('--output_dir', type=str, required=True, help='Directory for output (model, stats, etc.)')

        return parser

try:
    from models import CNNMnist, CNNFashion_Mnist, CNNCifar, MLP
except ImportError:
    print("ERROR in update_sub.py: Could not import models. Ensure models.py is accessible by adding src to PYTHONPATH or placing it correctly.")

try:
    from utils import get_dataset
except ImportError:
    print("ERROR in update_sub.py: Could not import get_dataset from utils. Ensure utils.py is accessible by adding src to PYTHONPATH or placing it correctly.")

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if isinstance(image, torch.Tensor): img_t = image.clone().detach()
        else: img_t = torch.as_tensor(image)
        if isinstance(label, torch.Tensor): lbl_t = label.clone().detach()
        else: lbl_t = torch.tensor(label)
        return self.idxs[item], img_t, lbl_t

class LocalUpdate(object):
    def __init__(self, args_obj, full_dataset_obj, client_specific_global_idxs_list, logger_obj, client_id_val):
        self.args = args_obj
        self.logger = logger_obj
        self.full_dataset = full_dataset_obj
        self.client_specific_global_idxs = list(client_specific_global_idxs_list)
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            self.full_dataset, self.client_specific_global_idxs
        )
        self.device = 'cpu'
        effective_gpu_arg = getattr(self.args, 'gpu', None)
        if effective_gpu_arg is not None:
            gpu_arg_val = str(effective_gpu_arg)
            if gpu_arg_val.lower() not in ['none', 'cpu','']:
                try:
                    gpu_id_val = int(gpu_arg_val.split(':')[-1])
                    if torch.cuda.is_available() and gpu_id_val < torch.cuda.device_count():
                        self.device = f'cuda:{gpu_id_val}'
                except ValueError: pass
            elif isinstance(effective_gpu_arg, int):
                 if torch.cuda.is_available() and effective_gpu_arg < torch.cuda.device_count():
                    self.device = f'cuda:{effective_gpu_arg}'
        self.criterion = nn.CrossEntropyLoss() # .to(self.device) はモデルやデータ転送時に行う
        self.client_id = client_id_val
        self.process = psutil.Process(os.getpid())

    def train_val_test(self, global_dataset, client_global_idxs_list):
        num_client_data = len(client_global_idxs_list)
        current_local_bs = self.args.local_bs if hasattr(self.args, 'local_bs') and self.args.local_bs > 0 else 10
        if num_client_data == 0 or global_dataset is None or len(global_dataset) == 0 :
            empty_ds_split = DatasetSplit(global_dataset if global_dataset and len(global_dataset)>0 else torch.utils.data.TensorDataset(torch.empty(0,dtype=torch.float),torch.empty(0,dtype=torch.long)), [])
            empty_loader = DataLoader(empty_ds_split, batch_size=current_local_bs)
            return empty_loader, empty_loader, empty_loader
        train_end_idx = int(0.8 * num_client_data)
        val_end_idx = int(0.9 * num_client_data)
        idxs_train = client_global_idxs_list[:train_end_idx]
        idxs_val = client_global_idxs_list[train_end_idx:val_end_idx]
        idxs_test = client_global_idxs_list[val_end_idx:]
        num_workers_val = getattr(self.args, 'num_workers', 0)
        trainloader = DataLoader(DatasetSplit(global_dataset, idxs_train),
                                 batch_size=current_local_bs, shuffle=True, 
                                 drop_last=True if len(idxs_train) >= current_local_bs else False, num_workers=num_workers_val)
        val_bs = max(1, int(len(idxs_val) / 10)) if len(idxs_val) > 0 else 1
        validloader = DataLoader(DatasetSplit(global_dataset, idxs_val), batch_size=val_bs, shuffle=False, 
                                 drop_last=True if len(idxs_val) >= val_bs else False, num_workers=num_workers_val)
        test_bs = max(1, int(len(idxs_test) / 10)) if len(idxs_test) > 0 else 1
        testloader = DataLoader(DatasetSplit(global_dataset, idxs_test), batch_size=test_bs, shuffle=False, 
                                drop_last=True if len(idxs_test) >= test_bs else False, num_workers=num_workers_val)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        model.to(self.device)
        model.train()
        epoch_loss = []
        lr = self.args.lr * (0.5 ** (global_round // 10))
        optimizer_type = getattr(self.args, 'optimizer', 'sgd')
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=getattr(self.args, 'momentum', 0.9), weight_decay=1e-4)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        else: 
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        self.process.cpu_percent(interval=None) 
        t_start_local_train = time.monotonic()
        num_samples_in_current_loader = 0
        if self.trainloader and hasattr(self.trainloader, 'dataset') and self.trainloader.dataset:
             num_samples_in_current_loader = len(self.trainloader.dataset)
        calculated_loss_for_round = float('nan') 

        if num_samples_in_current_loader > 0 and len(self.trainloader) > 0:
            for iter_ep in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, data_batch in enumerate(self.trainloader):
                    try: _, images, labels = data_batch 
                    except ValueError: continue 
                    if not images.size(0): continue 
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad(); probs = model(images); loss_val = self.criterion(probs, labels)
                    loss_val.backward(); optimizer.step()
                    batch_loss.append(loss_val.item())
                if batch_loss: epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if epoch_loss: calculated_loss_for_round = sum(epoch_loss) / len(epoch_loss)
        
        train_time_this_round = time.monotonic() - t_start_local_train
        process_cpu_usage_this_client = self.process.cpu_percent(interval=None)
        client_ram_rss_mb_this_client = self.process.memory_info().rss / (1024 * 1024) 
        performance_metrics = {
            'train_time': train_time_this_round,
            'num_samples': num_samples_in_current_loader,
            'process_cpu_usage': process_cpu_usage_this_client if process_cpu_usage_this_client is not None else 0.0,
            'client_ram_rss_mb': client_ram_rss_mb_this_client
        }
        return model.cpu().state_dict(), calculated_loss_for_round, performance_metrics

    def compute_el2n_scores_4(self, model):
        model.to(self.device)
        model.eval()
        el2n_scores_collected = []
        score_calculation_loader = self.trainloader
        if not (score_calculation_loader and hasattr(score_calculation_loader, 'dataset') and score_calculation_loader.dataset and len(score_calculation_loader.dataset) > 0):
            return torch.tensor([])
        num_classes = getattr(self.args, 'num_classes', 10)
        for _, data, labels in score_calculation_loader: 
            if not data.size(0): continue
            data, labels = data.to(self.device), labels.to(self.device)
            if labels.numel() > 0: 
                if labels.max() >= num_classes or labels.min() < 0: continue 
            elif data.numel() > 0 : continue
            if data.numel() == 0: continue
            try:
                probs = model(data)
                one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
                errors = F.softmax(probs, dim=1) - one_hot_labels
                scores = torch.norm(errors, p=2, dim=1).detach().cpu()
                el2n_scores_collected.append(scores)
            except RuntimeError: continue
        return torch.cat(el2n_scores_collected) if el2n_scores_collected else torch.tensor([])

    def update_dataset(self, new_keep_local_idxs_for_current_trainloader, el2n_arg_dummy):
        current_dataset_split = self.trainloader.dataset 
        if not isinstance(current_dataset_split, DatasetSplit):
            self.client_specific_global_idxs = [] 
            self.trainloader, self.validloader, self.testloader = self.train_val_test(
                self.full_dataset, self.client_specific_global_idxs)
            return
        if not new_keep_local_idxs_for_current_trainloader or len(new_keep_local_idxs_for_current_trainloader) == 0:
            self.client_specific_global_idxs = []
        else:
            try:
                valid_indices = [i for i in new_keep_local_idxs_for_current_trainloader if i < len(current_dataset_split.idxs)]
                self.client_specific_global_idxs = [current_dataset_split.idxs[i] for i in valid_indices]
            except IndexError: self.client_specific_global_idxs = [] 
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            self.full_dataset, self.client_specific_global_idxs)

    def inference(self, model):
        model.to(self.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        current_test_loader = self.testloader
        if not (current_test_loader and hasattr(current_test_loader, 'dataset') and current_test_loader.dataset and len(current_test_loader.dataset) > 0 and len(current_test_loader) > 0):
            return 0.0, 0.0
        num_batches_processed = 0
        for _, images, labels in current_test_loader: 
            if not images.size(0): continue
            num_batches_processed +=1
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss_val = self.criterion(outputs, labels)
            loss += batch_loss_val.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct/total if total > 0 else 0.0
        avg_loss = loss / num_batches_processed if num_batches_processed > 0 else 0.0
        return accuracy, avg_loss
# import sys
# print(f"Executing update_sub.py with arguments: {sys.argv}")
# --- サブプロセスとして実行される際のメイン処理ブロック ---
if __name__ == '__main__':
    # 1. コマンドライン引数パーサーの定義
    parser = argparse.ArgumentParser(description="Client Update Subprocess Script for Federated Learning")
    
    # --- limited_ccs.py から渡される必須引数を定義 ---
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    parser.add_argument('--global_model_path', type=str, required=True, help='Path to the global model file')
    parser.add_argument('--current_epoch', type=int, required=True, help='Current epoch number')
    parser.add_argument('--data_indices_path', type=str, required=True, help='Path to the data indices file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    # --- options.py から引き継ぐ共通の学習パラメータをここで定義 ---
    #     limited_ccs.py の cmd_list 作成ロジックで実際に渡されるものを網羅する
    default_args_source = main_args_parser_for_defaults_setup()

    # limited_ccs.py から渡される可能性のある引数と、LocalUpdateやモデル構築で必要な引数を定義
    # エラーメッセージの "usage:" にリストされていた引数をすべて定義する
    parser.add_argument('--epochs', type=int, default=getattr(default_args_source, 'epochs', 10))
    parser.add_argument('--num_users', type=int, default=getattr(default_args_source, 'num_users', 10)) # サブプロセスでは直接使わないことが多い
    parser.add_argument('--frac', type=float, default=getattr(default_args_source, 'frac', 0.5))     # サブプロセスでは直接使わないことが多い
    parser.add_argument('--local_ep', type=int, default=getattr(default_args_source, 'local_ep', 3))
    parser.add_argument('--local_bs', type=int, default=getattr(default_args_source, 'local_bs', 32))
    parser.add_argument('--lr', type=float, default=getattr(default_args_source, 'lr', 0.03))
    parser.add_argument('--momentum', type=float, default=getattr(default_args_source, 'momentum', 0.9))
    parser.add_argument('--model', type=str, default=getattr(default_args_source, 'model', 'cnn'))
    parser.add_argument('--kernel_num', type=int, default=getattr(default_args_source, 'kernel_num', 9))
    parser.add_argument('--kernel_sizes', type=str, default=getattr(default_args_source, 'kernel_sizes', '3,4,5'))
    parser.add_argument('--num_channels', type=int, default=getattr(default_args_source, 'num_channels', 3))
    parser.add_argument('--norm', type=str, default=getattr(default_args_source, 'norm', 'batch_norm'))
    parser.add_argument('--num_filters', type=int, default=getattr(default_args_source, 'num_filters', 32))
    parser.add_argument('--max_pool', type=str, default=getattr(default_args_source, 'max_pool', 'True'))
    parser.add_argument('--dataset', type=str, default=getattr(default_args_source, 'dataset', 'cifar'))
    parser.add_argument('--num_classes', type=int, default=getattr(default_args_source, 'num_classes', 10))
    parser.add_argument('--gpu', default=getattr(default_args_source, 'gpu', 'cuda:0')) 
    parser.add_argument('--optimizer', type=str, default=getattr(default_args_source, 'optimizer', 'sgd'))
    parser.add_argument('--iid', type=int, default=getattr(default_args_source, 'iid', 1)) 
    parser.add_argument('--unequal', type=int, default=getattr(default_args_source, 'unequal', 0)) 
    parser.add_argument('--stopping_rounds', type=int, default=getattr(default_args_source, 'stopping_rounds', 10)) # サブプロセスでは直接使わない
    parser.add_argument('--verbose', type=int, default=getattr(default_args_source, 'verbose', 0)) # サブプロセスでは0推奨
    parser.add_argument('--seed', type=int, default=getattr(default_args_source, 'seed', 42))
    parser.add_argument('--el2n', type=int, default=getattr(default_args_source, 'el2n', 5))
    parser.add_argument('--threshold', type=float, default=getattr(default_args_source, 'threshold', 0.5))
    parser.add_argument('--percent', type=float, default=getattr(default_args_source, 'percent', 0.5))
    parser.add_argument('--acc_thre1', type=float, default=getattr(default_args_source, 'acc_thre1', 0.7))
    parser.add_argument('--acc_thre2', type=float, default=getattr(default_args_source, 'acc_thre2', 0.75))
    parser.add_argument('--acc_thre3', type=float, default=getattr(default_args_source, 'acc_thre3', 0.8))
    parser.add_argument('--start_accuracy', type=float, default=getattr(default_args_source, 'start_accuracy', 0.6))
    parser.add_argument('--pru_percent', type=float, default=getattr(default_args_source, 'pru_percent', 0.6))
    parser.add_argument('--num_per_client', type=int, default=getattr(default_args_source, 'num_per_client',10000))
    parser.add_argument('--prune_rate', type=float, default=getattr(default_args_source, 'prune_rate', 0.1))
    parser.add_argument('--dim_hidden', type=int, default=getattr(default_args_source, 'dim_hidden', 64)) # MLP用

    # limited_ccs.pyから渡されないが、options.pyには存在する引数 (デフォルト値で定義)
    # これらはサブプロセス側では実際には使われない想定
    parser.add_argument('--client_script_path', type=str, default=getattr(default_args_source, 'client_script_path', 'src/update_sub.py'))
    parser.add_argument('--temp_io_dir', type=str, default=getattr(default_args_source, 'temp_io_dir', './client_io_temp_ccs'))
    parser.add_argument('--client_cpu_limits_str', type=str, default=getattr(default_args_source, 'client_cpu_limits_str', '{}'))
    parser.add_argument('--default_client_cpu_limit', type=int, default=getattr(default_args_source, 'default_client_cpu_limit', 100))
    parser.add_argument('--client_timeout', type=int, default=getattr(default_args_source, 'client_timeout', 600))
    parser.add_argument('--burden_alpha', type=float, default=getattr(default_args_source, 'burden_alpha', 0.3))
    parser.add_argument('--burden_w_time', type=float, default=getattr(default_args_source, 'burden_w_time', 0.4))
    parser.add_argument('--burden_w_cpu', type=float, default=getattr(default_args_source, 'burden_w_cpu', 0.3))
    parser.add_argument('--burden_w_ram', type=float, default=getattr(default_args_source, 'burden_w_ram', 0.3))
    parser.add_argument('--min_keep_ratio', type=float, default=getattr(default_args_source, 'min_keep_ratio', 0.2))
    parser.add_argument('--max_keep_ratio', type=float, default=getattr(default_args_source, 'max_keep_ratio', 1.0))
    parser.add_argument('--default_keep_ratio', type=float, default=getattr(default_args_source, 'default_keep_ratio', 0.8))
    parser.add_argument('--start_prune_round', type=int, default=getattr(default_args_source, 'start_prune_round', 10))
    parser.add_argument('--prune_interval', type=int, default=getattr(default_args_source, 'prune_interval', 10))
    parser.add_argument('--max_overall_reduction_rate', type=float, default=getattr(default_args_source, 'max_overall_reduction_rate', 0.6))
    parser.add_argument('--min_samples_after_global_prune', type=int, default=getattr(default_args_source, 'min_samples_after_global_prune', 20))
    parser.add_argument('--num_groups_ccs', type=int, default=getattr(default_args_source, 'num_groups_ccs', 100))


    args_parsed_in_subprocess = parser.parse_args()

    # ---- これ以降は、クライアントの処理ロジック (前回提示のものをベースに、args_parsed_in_subprocess を使用) ----
    
    final_device_for_client_str = 'cpu'
    if hasattr(args_parsed_in_subprocess, 'gpu') and args_parsed_in_subprocess.gpu is not None:
        gpu_arg_str_val = str(args_parsed_in_subprocess.gpu) 
        if gpu_arg_str_val.lower() not in ['none', 'cpu', '']:
            try:
                gpu_id_str_val = gpu_arg_str_val.split(':')[-1]
                gpu_id_val = int(gpu_id_str_val)
                if torch.cuda.is_available() and gpu_id_val < torch.cuda.device_count():
                    # torch.cuda.set_device(gpu_id_val) # メインプロセスで管理、サブプロセスでは不要
                    final_device_for_client_str = f'cuda:{gpu_id_val}'
            except ValueError: pass
        elif isinstance(args_parsed_in_subprocess.gpu, int):
             if torch.cuda.is_available() and args_parsed_in_subprocess.gpu < torch.cuda.device_count():
                # torch.cuda.set_device(args_parsed_in_subprocess.gpu)
                final_device_for_client_str = f'cuda:{args_parsed_in_subprocess.gpu}'
    final_device_for_client = torch.device(final_device_for_client_str)

    # データロード
    full_train_dataset_sub_loaded = None
    client_data_indices_sub_loaded = None
    try:
        full_train_dataset_sub_loaded, _, _ = get_dataset(args_parsed_in_subprocess) 
        if full_train_dataset_sub_loaded is None: raise ValueError("get_dataset returned None.")
        with open(args_parsed_in_subprocess.data_indices_path, 'rb') as f: client_data_indices_sub_loaded = pickle.load(f)
        
        if not client_data_indices_sub_loaded and isinstance(client_data_indices_sub_loaded, list):
             perf_metrics_empty = {'train_time':0.0, 'num_samples':0, 'process_cpu_usage':0.0, 'client_ram_rss_mb':0.0, 'error': "Empty data indices."}
             output_empty = {'client_id': args_parsed_in_subprocess.client_id, 'weights': {}, 'loss': float('nan'), 'perf_metrics': perf_metrics_empty}
             output_filename_empty = f'client_{args_parsed_in_subprocess.client_id}_round_{args_parsed_in_subprocess.current_epoch}_output.pkl'
             output_path_empty = os.path.join(args_parsed_in_subprocess.output_dir, output_filename_empty)
             os.makedirs(args_parsed_in_subprocess.output_dir, exist_ok=True)
             with open(output_path_empty, 'wb') as f_out_empty: pickle.dump(output_empty, f_out_empty)
             exit(0) 
    except Exception as e:
        print(f"Subprocess Client {args_parsed_in_subprocess.client_id}: FATAL Error during data loading - {e}")
        error_output = {'client_id': args_parsed_in_subprocess.client_id, 'error': f"Data loading error: {e}", 
                        'perf_metrics': {'num_samples': 0, 'train_time':0, 'process_cpu_usage':0, 'client_ram_rss_mb':0}}
        error_filename = f'client_{args_parsed_in_subprocess.client_id}_round_{args_parsed_in_subprocess.current_epoch}_error.pkl'
        error_path = os.path.join(args_parsed_in_subprocess.output_dir, error_filename)
        os.makedirs(args_parsed_in_subprocess.output_dir, exist_ok=True)
        with open(error_path, 'wb') as f_err: pickle.dump(error_output, f_err)
        exit(1)

    # モデル構築とロード
    client_model_for_training = None
    try:
        if args_parsed_in_subprocess.model == 'cnn':
            if args_parsed_in_subprocess.dataset == 'mnist': client_model_for_training = CNNMnist(args=args_parsed_in_subprocess)
            elif args_parsed_in_subprocess.dataset == 'fmnist': client_model_for_training = CNNFashion_Mnist(args=args_parsed_in_subprocess)
            elif args_parsed_in_subprocess.dataset == 'cifar': 
                client_model_for_training = torchvision.models.resnet18(num_classes=args_parsed_in_subprocess.num_classes)
        elif args_parsed_in_subprocess.model == 'mlp':
            if len(full_train_dataset_sub_loaded) == 0: raise ValueError("Full training dataset is empty for MLP.")
            try: sample_img_sub_for_mlp, _ = full_train_dataset_sub_loaded[0]
            except IndexError: raise ValueError("Cannot get sample from empty full_train_dataset_sub for MLP.")
            img_size_sub_for_mlp = sample_img_sub_for_mlp.shape; len_in_sub_for_mlp = 1
            for x_dim_mlp_loop in img_size_sub_for_mlp: len_in_sub_for_mlp *= x_dim_mlp_loop
            client_model_for_training = MLP(dim_in=len_in_sub_for_mlp, 
                                   dim_hidden=getattr(args_parsed_in_subprocess, 'dim_hidden', 64), 
                                   dim_out=args_parsed_in_subprocess.num_classes)
        if client_model_for_training is None: raise ValueError(f"Model creation failed.")
        
        client_model_for_training.load_state_dict(torch.load(args_parsed_in_subprocess.global_model_path, map_location='cpu'))
        client_model_for_training.to(final_device_for_client)
    except Exception as e:
        print(f"Subprocess Client {args_parsed_in_subprocess.client_id}: FATAL Error during model setup - {e}")
        error_output = {'client_id': args_parsed_in_subprocess.client_id, 'error': f"Model setup error: {e}", 
                        'perf_metrics': {'num_samples': 0, 'train_time':0, 'process_cpu_usage':0, 'client_ram_rss_mb':0}}
        # ... (エラー出力ファイルの保存) ...
        exit(1)

    # LocalUpdate インスタンス化
    try:
        local_updater_instance = LocalUpdate(args_obj=args_parsed_in_subprocess, 
                                        full_dataset_obj=full_train_dataset_sub_loaded, 
                                        client_specific_global_idxs_list=client_data_indices_sub_loaded, 
                                        logger_obj=None, 
                                        client_id_val=args_parsed_in_subprocess.client_id)
        local_updater_instance.device = final_device_for_client
    except Exception as e:
        print(f"Subprocess Client {args_parsed_in_subprocess.client_id}: FATAL Error instantiating LocalUpdate - {e}")
        error_output = {'client_id': args_parsed_in_subprocess.client_id, 'error': f"LocalUpdate instantiation error: {e}", 
                        'perf_metrics': {'num_samples': 0, 'train_time':0, 'process_cpu_usage':0, 'client_ram_rss_mb':0}}
        # ... (エラー出力ファイルの保存) ...
        exit(1)

    # ローカル学習実行
    final_updated_weights, final_loss, final_perf_metrics = ({}, float('inf'), 
        {'train_time':0.0, 'num_samples':0, 'process_cpu_usage':0.0, 'client_ram_rss_mb':0.0, 'error': 'Init before train'})
    try:
        final_updated_weights, final_loss, final_perf_metrics = local_updater_instance.update_weights(
            model=client_model_for_training, global_round=args_parsed_in_subprocess.current_epoch
        )
    except Exception as e:
        print(f"Subprocess Client {args_parsed_in_subprocess.client_id}: Error during update_weights - {e}")
        # ... (エラー時のメトリクス設定) ...
        t_error_train_val = time.monotonic() - getattr(local_updater_instance, 't_start_local_train', time.monotonic()) 
        error_cpu_val = local_updater_instance.process.cpu_percent(interval=None) if hasattr(local_updater_instance, 'process') else -1.0
        error_ram_val = local_updater_instance.process.memory_info().rss / (1024*1024) if hasattr(local_updater_instance, 'process') else -1.0
        final_perf_metrics = {
            'train_time': t_error_train_val, 'num_samples': 0, 
            'process_cpu_usage': error_cpu_val if error_cpu_val is not None else -1.0, 
            'client_ram_rss_mb': error_ram_val, 'error': f"update_weights error: {e}"}
        final_updated_weights = client_model_for_training.cpu().state_dict() 
        final_loss = float('inf')

    # 結果保存
    output_data_to_save = {
        'client_id': args_parsed_in_subprocess.client_id,
        'weights': final_updated_weights, 'loss': final_loss, 'perf_metrics': final_perf_metrics,
    }
    os.makedirs(args_parsed_in_subprocess.output_dir, exist_ok=True)
    client_output_filename_final = f'client_{args_parsed_in_subprocess.client_id}_round_{args_parsed_in_subprocess.current_epoch}_output.pkl'
    client_output_path_final = os.path.join(args_parsed_in_subprocess.output_dir, client_output_filename_final)
    try:
        with open(client_output_path_final, 'wb') as f: pickle.dump(output_data_to_save, f)
    except Exception as e:
        print(f"Subprocess Client {args_parsed_in_subprocess.client_id}: CRITICAL Error saving output to {client_output_path_final} - {e}")