#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn.functional as F
import numpy as np
from decimal import Decimal
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import time
import time
import psutil # psutil をインポート

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    # def __getitem__(self, item):
    #     image, label = self.dataset[self.idxs[item]]
    #     return self.idxs[item], torch.tensor(image), torch.tensor(label)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
# --- image 側の処理 ---
        if isinstance(image, torch.Tensor):
            img_t = image.clone().detach()
        else:
            # NumPy 配列や PIL Image の場合
            img_t = torch.as_tensor(image)

        # --- label 側の処理 ---
        if isinstance(label, torch.Tensor):
            lbl_t = label.clone().detach()
        else:
            # Python int/float の場合
            lbl_t = torch.tensor(label)

        return self.idxs[item], img_t, lbl_t        


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, client_id):
        self.args = args
        self.logger = logger
        
        # CHANGED: データセットとインデックスの管理を明確化
        self.full_dataset = dataset  # 親から渡される、分割前のフルデータセット全体
        self.client_specific_global_idxs = list(idxs) # このクライアントが担当する、フルデータセット中のインデックスのリスト

        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            self.full_dataset, self.client_specific_global_idxs)
        
        # CHANGED: device 設定をより安全に
        self.device = 'cuda' if hasattr(self.args, 'gpu') and self.args.gpu is not None and torch.cuda.is_available() else 'cpu'
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.client_id = client_id
        
        # NEW: psutil.Process() を初期化時に取得
        self.process = psutil.Process(os.getpid())
        
        # REMOVED: self.train_dataset = None (未使用だったため)
        # REMOVED: self.idxs = idxs (self.client_specific_global_idxs で管理)
        # REMOVED: self.pruned_idxs = [] (未使用だったため)
        

    def compute_el2n_scores(self, model, el2n_threshold: float):
        """
        Compute el2n scores for the local dataset.

        Args:
        - model: Global model for which el2n scores are computed.

        Returns:
        - el2n_scores: Numpy array containing el2n scores for each data point.
        """
        model.eval()
        el2n_scores = []
        keep_orig_idxs = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            # outputs = F.softmax(model(data), dim=1)
            # errors = outputs - F.one_hot(labels,num_classes=self.args.num_classes).float()
            # log_probs = model(data)
            # probs = torch.exp(log_probs)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            # print("el2n scores = ",scores)
            el2n_scores.append(scores)
            keep_idxs = torch.where(scores > el2n_threshold)[0]
            # print("keep_idxs a = ",keep_idxs) #学習対象の番号を表示 speed upのため削減
            keep_orig_idxs.append(orig_idxs[keep_idxs])
            if len(keep_idxs) == 0:
                keep_orig_idxs.append(orig_idxs[[1]])
            # el2n = torch.norm(errors, p=2, dim=1).detach().cpu().numpy()
            # el2n_scores.extend(el2n)
        # TODO: Sort?
        return torch.cat(keep_orig_idxs)
        # return torch.cat(el2n_scores)
        # return np.array(el2n_scores)

    def compute_el2n_scores_1(self, model, percent):
        """
        Compute el2n scores for the local dataset.

        Args:
        - model: Global model for which el2n scores are computed.

        Returns:
        - el2n_scores: Numpy array containing el2n scores for each data point.
        """
        model.eval()
        el2n_scores = []
        keep_orig_idxs = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            # outputs = F.softmax(model(data), dim=1)
            # errors = outputs - F.one_hot(labels,num_classes=self.args.num_classes).float()
            # log_probs = model(data)
            # probs = torch.exp(log_probs)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            # print("el2n scores = ",scores)
            el2n_scores.append(scores)
            sorted, indices = torch.sort(scores)
            print("sorted = ",sorted)
            print("indices = ",indices)
            memory = int(len(data) * (1 - percent)) #Pruning一回なのでlen(data)でもorg_numでもOK!
            print("memory = ",memory)
            keep_idxs = indices[:memory]
            keep_orig_idxs.append(orig_idxs[keep_idxs])

            # keep_idxs = torch.where(scores > el2n_threshold)[0]
            # keep_orig_idxs.append(orig_idxs[keep_idxs])

            # el2n = torch.norm(errors, p=2, dim=1).detach().cpu().numpy()
            # el2n_scores.extend(el2n)

        # TODO: Sort?
        if not keep_orig_idxs :
            return [0]
        else :
            return torch.cat(keep_orig_idxs)
        
    def compute_el2n_scores_2(self, model, percent, sup):
        """
        Compute el2n scores for the local dataset.

        Args:
        - model: Global model for which el2n scores are computed.

        Returns:
        - el2n_scores: Numpy array containing el2n scores for each data point.
        """
        #6割で25％、７割で50％、8割で75％でプルーニングする？？
        #一次(二次）関数的にプルーニングを行う。始点と終点を確定！
        model.eval()
        el2n_scores = []
        keep_orig_idxs = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            # outputs = F.softmax(model(data), dim=1)
            # errors = outputs - F.one_hot(labels,num_classes=self.args.num_classes).float()
            # log_probs = model(data)
            # probs = torch.exp(log_probs)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            # print("el2n scores = ",scores)
            el2n_scores.append(scores)
            sorted, indices = torch.sort(scores)
            # print("sorted = ",sorted)
            # print("indices = " ,indices)
            print("len(indices) = ",len(indices))
            print("len(data) = " ,len(data))
            print("sup = ",sup)
            memory = int(len(data) * (1 - percent) + sup)
            # memory = int(memory_a * (1 - percent))
            print("memory = ",memory)
            keep_idxs = indices[:memory]
            print("len(keep_idxs) = ",len(keep_idxs))
            keep_orig_idxs.append(orig_idxs[keep_idxs])
            print("len(keep_orig_idxs) = ",len(keep_orig_idxs))
            # keep_idxs = torch.where(scores > el2n_threshold)[0]
            # keep_orig_idxs.append(orig_idxs[keep_idxs])

            # el2n = torch.norm(errors, p=2, dim=1).detach().cpu().numpy()
            # el2n_scores.extend(el2n)
        # TODO: Sort?
        if not keep_orig_idxs :
            return [0]
        else :
            return torch.cat(keep_orig_idxs)

    def compute_el2n_scores_3(self, model, pru_batch):
        """
        Compute el2n scores for the local dataset.

        Args:
        - model: Global model for which el2n scores are computed.

        Returns:
        - el2n_scores: Numpy array containing el2n scores for each data point.
        """
        #6割で25％、７割で50％、8割で75％でプルーニングする？？
        #一次(二次）関数的にプルーニングを行う。始点と終点を確定！
        model.eval()
        el2n_scores = []
        keep_orig_idxs = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            # outputs = F.softmax(model(data), dim=1)
            # errors = outputs - F.one_hot(labels,num_classes=self.args.num_classes).float()
            # log_probs = model(data)
            # probs = torch.exp(log_probs)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            # print("el2n scores = ",scores)
            el2n_scores.append(scores)
            sorted, indices = torch.sort(scores)
            # print("len(indices) = ",len(indices))
            # print("len(data) = " ,len(data))
            # print("sup = ",sup)
            memory = int(len(data) - pru_batch)
            # print("memory = ",memory)
            keep_idxs = indices[:memory]
            # print("len(keep_idxs) = ",len(keep_idxs))
            keep_orig_idxs.append(orig_idxs[keep_idxs])
            # print("len(keep_orig_idxs) = ",len(keep_orig_idxs))
            # keep_idxs = torch.where(scores > el2n_threshold)[0]
            # keep_orig_idxs.append(orig_idxs[keep_idxs])

            # el2n = torch.norm(errors, p=2, dim=1).detach().cpu().numpy()
            # el2n_scores.extend(el2n)
        # TODO: Sort?
        if not keep_orig_idxs :
            return [0]
        else :
            return torch.cat(keep_orig_idxs)
        
    def compute_el2n_scores_4(self, model):
        """
        Compute el2n scores for the local dataset.

        Args:
        - model: Global model for which el2n scores are computed.

        Returns:
        - el2n_scores: 1D tensor containing el2n scores for all data points.
        """
        model.eval()
        el2n_scores_collected = []
        score_calculation_loader = self.trainloader # 現在のtrainloaderで計算

        if not (score_calculation_loader and score_calculation_loader.dataset and len(score_calculation_loader.dataset) > 0):
            return torch.tensor([])
        for _, data, labels in score_calculation_loader: # orig_idxは使わない
            if not data.size(0): continue
            data, labels = data.to(self.device), labels.to(self.device)
            probs = model(data)
            num_classes = getattr(self.args, 'num_classes', 10)
            if labels.max() >= num_classes: continue # エラー回避
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=num_classes).float()
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            el2n_scores_collected.append(scores)
        
        return torch.cat(el2n_scores_collected) if el2n_scores_collected else torch.tensor([])


    def save_el2n_scores(self, model, save_path):
        """
        Compute and save el2n scores for the local dataset.

        Args:
        - model: Global model for which el2n scores are computed.
        - save_path: Path to save the el2n scores.
        """
        os.makedirs(save_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
        el2n_scores = self.compute_el2n_scores_4(model, el2n_threshold=0.0)  # 全データのスコアを取得
        file_path = os.path.join(save_path, f"el2n_scores_client_{self.client_id}.npy")
        np.save(file_path, el2n_scores)
        print(f"el2n scores saved to {file_path}")

    def update_dataset(self, new_keep_local_idxs_for_current_trainloader, el2n):
        # CHANGED: データセット更新ロジックの明確化と堅牢化
        # new_keep_local_idxs_for_current_trainloader は、現在の self.trainloader.dataset 
        # (DatasetSplitインスタンス) の中での0からの連番インデックスのリストと仮定。
        
        current_dataset_split = self.trainloader.dataset
        if not isinstance(current_dataset_split, DatasetSplit):
            # print(f"Client {self.client_id} update_dataset Error: trainloader.dataset is not DatasetSplit.")
            self.client_specific_global_idxs = [] # 安全のため空にする
            self.trainloader, self.validloader, self.testloader = self.train_val_test(
                self.full_dataset, self.client_specific_global_idxs)
            return

        if not new_keep_local_idxs_for_current_trainloader or len(new_keep_local_idxs_for_current_trainloader) == 0:
            # print(f"Client {self.client_id}: No data to keep. Emptying dataset.")
            self.client_specific_global_idxs = []
        else:
            try:
                # 現在のDatasetSplitが持つグローバルインデックスリストから、保持すべきグローバルインデックスを選択
                self.client_specific_global_idxs = [current_dataset_split.idxs[i] for i in new_keep_local_idxs_for_current_trainloader]
            except IndexError:
                # print(f"Client {self.client_id} update_dataset Error: IndexError. Mismatch in keep_idxs and current dataset.")
                self.client_specific_global_idxs = [] # エラー時は安全のため空にする

        # 更新された client_specific_global_idxs を使って全データローダーを再作成
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            self.full_dataset, self.client_specific_global_idxs)
        # print(f"Client {self.client_id}: Dataset updated. New trainloader size: {len(self.trainloader.dataset)}")

    # CHANGED: train_val_test メソッドを明確化
    def train_val_test(self, global_dataset, client_global_idxs_list):
        num_client_data = len(client_global_idxs_list)

        if num_client_data == 0:
            empty_ds_split = DatasetSplit(global_dataset, [])
            empty_loader = DataLoader(empty_ds_split, batch_size=self.args.local_bs)
            return empty_loader, empty_loader, empty_loader

        train_end_idx = int(0.8 * num_client_data)
        val_end_idx = int(0.9 * num_client_data)

        idxs_train = client_global_idxs_list[:train_end_idx]
        idxs_val = client_global_idxs_list[train_end_idx:val_end_idx]
        idxs_test = client_global_idxs_list[val_end_idx:]
        
        trainloader = DataLoader(DatasetSplit(global_dataset, idxs_train),
                                batch_size=self.args.local_bs, shuffle=True, 
                                drop_last=True if len(idxs_train) >= self.args.local_bs else False)

        val_bs = max(1, int(len(idxs_val) / 10)) if len(idxs_val) > 0 else 1
        validloader = DataLoader(DatasetSplit(global_dataset, idxs_val),
                                batch_size=val_bs, shuffle=False, 
                                drop_last=True if len(idxs_val) >= val_bs else False)
        
        test_bs = max(1, int(len(idxs_test) / 10)) if len(idxs_test) > 0 else 1
        testloader = DataLoader(DatasetSplit(global_dataset, idxs_test),
                                batch_size=test_bs, shuffle=False, 
                                drop_last=True if len(idxs_test) >= test_bs else False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        lr = self.args.lr * (0.5 ** (global_round // 10))
        optimizer_type = getattr(self.args, 'optimizer', 'sgd')
        # Set optimizer for the local updates
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=getattr(self.args, 'momentum', 0.9), weight_decay=1e-4)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        else: 
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        print("len(self.trainloader.dataset) = ",len(self.trainloader.dataset))
        # NEW: CPU使用率測定の準備と学習時間計測開始
        self.process.cpu_percent(interval=None) 
        t_start_local_train = time.monotonic()

        
        num_samples_in_current_loader = len(self.trainloader.dataset) if self.trainloader and self.trainloader.dataset else 0

        if num_samples_in_current_loader == 0:
            process_cpu_usage = self.process.cpu_percent(interval=None)
            train_time_this_round = time.monotonic() - t_start_local_train
            performance_metrics = { # NEW: 返り値
                'train_time': train_time_this_round,
                'num_samples': 0,
                'process_cpu_usage': process_cpu_usage if process_cpu_usage is not None else 0.0
            }
            return model.state_dict(), 0.0, performance_metrics # CHANGED


        for iter_ep in range(self.args.local_ep):
            batch_loss = []
            if len(self.trainloader) == 0: continue
            for batch_idx, (_, images, labels) in enumerate(self.trainloader): # orig_idxは使わない
                if not images.size(0): continue 
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                probs = model(images)
                loss = self.criterion(probs, labels)
                loss.backward()
                optimizer.step()
                # (ログ関連は既存のものをベースにclient_idを付与するなど)
                self.logger.add_scalar(f'client_{self.client_id}_train_loss_batch', loss.item())
                batch_loss.append(loss.item())
            if batch_loss:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # NEW: CPU使用率と学習時間の測定終了
        process_cpu_usage = self.process.cpu_percent(interval=None) 
        train_time_this_round = time.monotonic() - t_start_local_train
        print("train_time = ",train_time_this_round)
        performance_metrics = { # NEW: 返り値
            'train_time': train_time_this_round,
            'num_samples': num_samples_in_current_loader,
            'process_cpu_usage': process_cpu_usage if process_cpu_usage is not None else 0.0
        }
        
        final_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        return model.state_dict(), final_loss, performance_metrics # CHANGED

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        current_test_loader = self.testloader
        if not (current_test_loader and current_test_loader.dataset and len(current_test_loader.dataset) > 0):
            return 0.0, 0.0
        
        num_batches_processed = 0
        for _, images, labels in current_test_loader: # orig_idxは使わない
            if not images.size(0): continue
            num_batches_processed +=1
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct/total if total > 0 else 0.0
        avg_loss = loss / num_batches_processed if num_batches_processed > 0 else 0.0
        return accuracy, avg_loss


def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if hasattr(args, 'gpu') and args.gpu is not None and torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    if not test_dataset or len(test_dataset) == 0: return 0.0, 0.0
    testloader = DataLoader(test_dataset, batch_size=getattr(args, 'test_bs', 128), shuffle=False, drop_last=False)
    num_batches_processed = 0
    for images, labels in testloader: # test_datasetは (image, label) を直接返す想定
        if not images.size(0): continue
        num_batches_processed +=1
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct/total if total > 0 else 0.0
    avg_loss = loss / num_batches_processed if num_batches_processed > 0 else 0.0
    return accuracy, avg_loss