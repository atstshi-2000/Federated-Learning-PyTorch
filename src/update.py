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
        self.train_dataset = None  # 元のトレーニングデータセット
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # Change if ResNet18 is used
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.idxs = idxs
        self.logger = logger
        self.full_dataset = dataset  # 元のトレーニングデータセット
        self.pruned_idxs = []  # プルーニングされたデータのインデックス
        self.client_id = client_id  # クライアントIDを追加
        

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
        el2n_scores = []      
        for orig_idxs, data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            probs = model(data)
            errors = F.softmax(probs, dim=1) - F.one_hot(labels, num_classes=self.args.num_classes)
            scores = torch.norm(errors, p=2, dim=1).detach().cpu()
            el2n_scores.append(scores)
        
        # 結合して1次元テンソルとして返す
        return torch.cat(el2n_scores) if el2n_scores else torch.tensor([])

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

    def update_dataset(self, keep_idxs, el2n):
        """
        Compute and save el2n scores for the local dataset.

        Args:
        - new_dataset: The new dataset to replace the existing local training dataset.
        """
        #reduced_idxs_scalar = int(reduced_idxs[0])  # Convert the first element to a scalar
        #self.idxs = [int(idx) - reduced_idxs_scalar for idx in self.idxs]
        #assert len(np.intersect1d(self.idxs_train, keep_idxs)) == len(keep_idxs)
        # if el2n != 5:
        #     self.trainloader = DataLoader(DatasetSplit(self.full_dataset, keep_idxs),
        #                                   batch_size=self.args.local_bs,
        #                                   shuffle=True, drop_last=True)
        # else:
            # Update the indices in DatasetSplit instead of accessing `data` directly
        self.trainloader.dataset.idxs = keep_idxs
        print(f"Updated dataset size: {len(self.trainloader.dataset)}")

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        self.idxs_train = idxs_train
        self.idxs_val = idxs_val
        self.idxs_test = idxs_test

        # print("len of idxs_train = ", len(idxs_train))
        # print("len of idxs_val = ", len(idxs_val))
        # print("len of idxs_test= ", len(idxs_test))

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last = True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False, drop_last = True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False, drop_last = True)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        lr = self.args.lr * (0.5 ** (global_round // 10))
        #lr = self.args.lr

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.9,weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)
        print("len(self.trainloader.dataset) = ",len(self.trainloader.dataset))
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (_, images, labels) in enumerate(self.trainloader):
                # if self.args.local_bs >= self.trainloader[0]:
                #     continue
                images, labels = images.to(self.device), labels.to(self.device)

                # print("images.size() = ",images.size())
                # print("images.shape = ",images.shape)
                model.zero_grad()
                probs = model(images)
                loss = self.criterion(probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (_,images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss  


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.orig
    """
    
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    # Change if ResNet18 is used
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, drop_last = True)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference_idxs
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
