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
import psutil  # CPU使用率を取得するためのライブラリ

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # --- image 側の処理 ---
        if isinstance(image, torch.Tensor):
            img_t = image.clone().detach()
        else:
            img_t = torch.as_tensor(image)
        # --- label 側の処理 ---
        if isinstance(label, torch.Tensor):
            lbl_t = label.clone().detach()
        else:
            lbl_t = torch.tensor(label)
        return self.idxs[item], img_t, lbl_t


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, client_id):
        self.args = args
        self.logger = logger
        self.train_dataset = None  # 元のトレーニングデータセット
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.idxs = idxs
        self.logger = logger
        self.full_dataset = dataset  # 元のトレーニングデータセット
        self.pruned_idxs = []  # プルーニングされたデータのインデックス
        self.client_id = client_id  # クライアントIDを追加



    def compute_el2n_scores_4(self, model):
        """Compute el2n scores for the local dataset."""

        model.eval()
        el2n_scores = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            el2n_scores.append(scores)

        return torch.cat(el2n_scores) if el2n_scores else torch.tensor([])

    def save_el2n_scores(self, model, save_path):
        """Compute and save el2n scores for the local dataset."""

        os.makedirs(save_path, exist_ok=True)  # ディレクトリが存在しない場合は作成
        el2n_scores = self.compute_el2n_scores_4(model)  # 全データのスコアを取得
        file_path = os.path.join(save_path, f"el2n_scores_client_{self.client_id}.npy")
        np.save(file_path, el2n_scores)
        print(f"el2n scores saved to {file_path}")

    def update_dataset(self, keep_idxs, el2n):
        """Update the dataset with the given indices."""

        self.trainloader.dataset.idxs = keep_idxs
        print(f"Updated dataset size: {len(self.trainloader.dataset)}")

    def train_val_test(self, dataset, idxs):
        """Returns train, validation and test dataloaders for a given dataset and user indexes."""

        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):] 

        self.idxs_train = idxs_train
        self.idxs_val = idxs_val
        self.idxs_test = idxs_test

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val) / 10), shuffle=False, drop_last=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test) / 10), shuffle=False, drop_last=True)

        return trainloader, validloader, testloader
    def update_weights(self, model, global_round): 
        """Update weights for the local model."""

        model.train()
        epoch_loss = []
        lr = self.args.lr * (0.5 ** (global_round // 10))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) if self.args.optimizer == 'sgd' else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (_, images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                probs = model(images)
                loss = self.criterion(probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_1(self, model, global_round):
        """Update weights for the local model."""

        model.train()
        epoch_loss = []
        lr = self.args.lr * (0.5 ** (global_round // 10))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) if self.args.optimizer == 'sgd' else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # ─── CPU サンプル用の変数を初期化 ───
        proc = psutil.Process()  
        # 最初に一度呼んでおくと、以降 interval=None で「前回からのデルタ」を計測しやすくなる
        _ = proc.cpu_percent(interval=None)
        # メモリの初期値は呼び出すだけでOK（ここで測定し直す間隔は行わず）
        _ = proc.memory_info().rss

        cpu_samples = []  # 各ミニバッチ後の CPU% を溜めておくリスト
        mem_samples = []  # バッチごとのメモリ使用量（RSS のままバイト単位で取得）

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (_, images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                probs = model(images)
                loss = self.criterion(probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                # ── (2) バッチ終了直後に CPU% をサンプリング ──
                # interval=None とすると「直前の呼出し以降の CPU 使用率を瞬間値で返す」
                cur_cpu = proc.cpu_percent(interval=None)
                cpu_samples.append(cur_cpu)

                # ── 同じくメモリ使用量 (RSS) をサンプリング ──
                #    RSS はバイト単位なので、MB に変換しておく
                cur_mem_mb = proc.memory_info().rss / (1024 ** 2)
                mem_samples.append(cur_mem_mb)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if len(cpu_samples) > 0:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
        else:
            avg_cpu = 0.0
        # ── ラウンド全体の平均メモリ(MB) を計算 ──
        if mem_samples:
            avg_mem = sum(mem_samples) / len(mem_samples)
        else:
            avg_mem = 0.0
        # 返り値を「モデル重み, 平均損失, 平均 CPU 使用率, 平均メモリ使用量」のタプルに変更
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), avg_cpu, avg_mem


    def inference(self, model):
        """Returns the inference accuracy and loss."""

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (_, images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(pred_labels.view(-1), labels)).item()
            total += len(labels)
        return correct / total, loss
    def compute_el2n_scores(self, model, el2n_threshold: float):
        """Compute el2n scores for the local dataset."""

        model.eval()
        el2n_scores = []
        keep_orig_idxs = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            el2n_scores.append(scores)
            keep_idxs = torch.where(scores > el2n_threshold)[0]
            keep_orig_idxs.append(orig_idxs[keep_idxs])
            if len(keep_idxs) == 0:
                keep_orig_idxs.append(orig_idxs[[1]])

        return torch.cat(keep_orig_idxs)

    def compute_el2n_scores_1(self, model, percent):
        """Compute el2n scores for the local dataset."""

        model.eval()
        el2n_scores = []
        keep_orig_idxs = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            el2n_scores.append(scores)
            sorted, indices = torch.sort(scores)
            memory = int(len(data) * (1 - percent))  # Pruning一回なのでlen(data)でもorg_numでもOK!
            keep_idxs = indices[:memory]
            keep_orig_idxs.append(orig_idxs[keep_idxs])

        return torch.cat(keep_orig_idxs) if keep_orig_idxs else [0]

    def compute_el2n_scores_2(self, model, percent, sup):
        """Compute el2n scores for the local dataset."""

        model.eval()
        el2n_scores = []
        keep_orig_idxs = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            el2n_scores.append(scores)
            sorted, indices = torch.sort(scores)
            memory = int(len(data) * (1 - percent) + sup)
            keep_idxs = indices[:memory]
            keep_orig_idxs.append(orig_idxs[keep_idxs])

        return torch.cat(keep_orig_idxs) if keep_orig_idxs else [0]

    def compute_el2n_scores_3(self, model, pru_batch):
        """Compute el2n scores for the local dataset."""

        model.eval()
        el2n_scores = []
        keep_orig_idxs = []

        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            el2n_scores.append(scores)
            sorted, indices = torch.sort(scores)
            memory = int(len(data) - pru_batch)
            keep_idxs = indices[:memory]
            keep_orig_idxs.append(orig_idxs[keep_idxs])

        return torch.cat(keep_orig_idxs) if keep_orig_idxs else [0]

def test_inference(args, model, test_dataset):
    """Returns the test accuracy and loss."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True)
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        correct += torch.sum(torch.eq(pred_labels.view(-1), labels)).item()
        total += len(labels)
    return correct / total, loss