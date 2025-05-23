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
import psutil
import json
from torch.utils.data import DataLoader
from collections import defaultdict
import slackweb

# slack送信メソッド
def slackPost(message):
    slack = slackweb.Slack(url = SLACKURL)
    slack.notify(text = message)

def extract_features(model, dataset, device):
    """
    Extract features (embeddings) for a dataset using the given model.

    Args:
    - model: Trained model.
    - dataset: Dataset for which features are to be extracted.
    - device: Device to run the model on.

    Returns:
    - features: Numpy array of extracted features.
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    features = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())

    return np.vstack(features)
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

###注意！！実行時には169行目以降のコメントアウトと340行目以降のコメント 関数プルーニングの時は116行目以降も！加えてユーザー選択もコメントアウトしているためそこの確認も忘れずに！
    
if __name__ == '__main__':
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    #print(args)
    exp_details(args)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    print("num of train dataset = ", len(train_dataset))
    print("num of test dataset = ", len(test_dataset))
    # print("num of  dataset = ", len(train_dataset))

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
    val_loss_pre, counter = 0, 0
    global_round = 0
    el2n_time = 0.0
    ta,tt,te_all,fin = 0,0,0,0
    local_models = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, client_id=idx) for idx in range(args.num_users)]
    report_lines = []
    # クライアントオブジェクトのリストを作成
    clients = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, client_id=idx) for idx in range(args.num_users)]
    el2n_times = {cid: 0.0 for cid in range(args.num_users)}
    train_times = {cid: 0.0 for cid in range(args.num_users)}
        # ループ前に空のリストを用意
    mem_history  = []   # 各ラウンドの memory_MB を格納
    size_history = []   # 各ラウンドの comm_KB    を格納
    # if args.dataset == 'cifar':
    org_num = 40000 / args.num_users
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        org_num = 48000 / args.num_users
    # pru_one,pru_two,pru_three = 0,0,0
    # percent1, percent2, percent3 = 0.25, 0.5, 0.75
    if args.el2n == 4:
        pru_func_stock = [0 for _ in range(10)]
        #線形Pruning用
        a1 = (args.local_bs * args.pru_percent) / (1 - args.start_accuracy)
        b1 = -1 * a1 * args.start_accuracy
    #二次関数Pruning用
        a2 = (args.local_bs * args.pru_percent) / (1 - (args.start_accuracy) ** 2)
        b2 = -1 * a2 * (args.start_accuracy) ** 2
        #指数関数Pruning用
        a =sp.Symbol('a')
        b = sp.Symbol('b')
        eq1 = a ** args.start_accuracy + b
        eq2 = a + b - args.local_bs * args.pru_percent
        a3,b3 = sp.solve([eq1,eq2],[a,b])[0]
    # print("local_models = ", local_models)
    # print("localmodels  = ", len(local_models[0].trainloader))
    el2n = args.el2n
    percent = args.percent
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        if len(choice_users) <= 4 :
            print("no client")
            break
        idxs_users = np.random.choice(choice_users, m, replace=False)        
        # if global_round == 10 and args.el2n == 1:
        #     idxs_users = np.array(range(args.num_users))
        # elif global_round == 10 and args.el2n == 3:
        #     idxs_users = np.array(range(args.num_users))

        #精度が〇〇％以上のとき〇〇％pruningするときの残骸
        # elif global_round > 3 and args.el2n == 4 and train_accuracy[-1] > args.acc_thre1 and pru_one == 0:
        #     idxs_users = np.array(range(args.num_users))
        # elif global_round > 3 and args.el2n == 4 and train_accuracy[-1] > args.acc_thre2 and pru_two == 0:
        #     idxs_users = np.array(range(args.num_users))
        # elif global_round > 3 and args.el2n == 4 and train_accuracy[-1] > args.acc_thre3 and pru_three == 0:
        #     idxs_users = np.array(range(args.num_users))
        
        # else :
        #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users = np.array(range(args.num_users))
        print("idxs_users = ", idxs_users)
        # print("pru_one = ",pru_one)
        # print("pru_two = ",pru_two)
        # print("pru_three =",pru_three)
        for idx in idxs_users:

            # if args.el2n != 0 and global_round == 1:
            # if args.el2n != 0 and global_round <= 1:

#一度のみPruningを行う場合186行目までのコメントアウトを削除！
            if args.el2n == 1 and global_round == 10 : 
                el2n_start_time = time.monotonic()
                el2n_threshold = args.threshold # これを適切な値に調整してください.毎回間引くときは0.05くらい 一回のときは0.5くらい
                # Calculate and print el2n scores
                local_model = local_models[idx]
                # el2n = local_model.compute_el2n_scores(global_model)
                # print(f'el2n scores: {el2n}')
                # Reduce the dataset based on el2n scores threshold
                # reduced_idxs = np.where(el2n <= el2n_threshold)[0]
                # keep_idxs = np.where(el2n > el2n_threshold)[0]
                keep_idxs = local_model.compute_el2n_scores(global_model, el2n_threshold)
                if len(keep_idxs) <= args.local_bs:
                    choice_users.remove(idx)
                    continue
                print("keep_idxs = ",keep_idxs)
                local_model.update_dataset(keep_idxs, el2n=args.el2n)
                pru_num = org_num -len(keep_idxs)# すでに間引かれたデータセットの総数
                # print(f"User {idx}, keep: {keep_idxs[:10]}")
                print(f'Remaining dataset size after el2n thresholding: {len(keep_idxs)}')
                print("Pruned dataset size after el2n thresholding",pru_num)
                prun_num.append(pru_num)
                pri_pru[idx] = pru_num
                el2n_time = (el2n_end_time - el2n_start_time)
                el2n_times[idx] += el2n_time

            # #各回Pruningを行う場合199行目までのコメントアウトを削除！
            if args.el2n == 2 and global_round >= 10 : # 提案手法１ 各回pruning
                el2n_start_time = time.monotonic()
                el2n_threshold = args.threshold # これを適切な値に調整してください.毎回間引くときは0.05くらい 一回のときは0.5くらい
                local_model = local_models[idx]
                keep_idxs = local_model.compute_el2n_scores(global_model, el2n_threshold)
                if len(keep_idxs) <= args.local_bs:
                    choice_users.remove(idx)
                    continue            #     local_model.update_dataset(keep_idxs, el2n=args.el2n)
                pru_num = org_num - len(keep_idxs)# すでに間引かれたデータセットの総数
                # print(f"User {idx}, keep: {keep_idxs[:10]}")
                print(f'Remaining dataset size after el2n thresholding: {len(keep_idxs)}')
                print("Pruning dataset size after el2n thresholding",pru_num)
                prun_num.append(pru_num)
                pri_pru[idx] = pru_num
                el2n_time = (el2n_end_time - el2n_start_time)
                el2n_times[idx] += el2n_time

            
#既存手法を行う場合213行目までのコメントアウトを削除！
            if args.el2n == 3 and global_round == 10 : # 既存手法
                el2n_start_time = time.monotonic()
                el2n_threshold = args.threshold
                local_model = local_models[idx]
                keep_idxs = local_model.compute_el2n_scores_1(global_model, percent)
                print("keep_idxs = ",keep_idxs)
                if len(keep_idxs) <= args.local_bs:
                    choice_users.remove(idx)
                    continue
                local_model.update_dataset(keep_idxs, el2n=args.el2n)
                pru_num = org_num -len(keep_idxs)# すでに間引かれたデータセットの総数
                # print(f"User {idx}, keep: {keep_idxs[:10]}")
                print(f'Remaining dataset size after el2n thresholding: {len(keep_idxs)}')
                print("Pruning dataset size after el2n thresholding",pru_num)
                prun_num.append(pru_num)
                pri_pru[idx] = pru_num
                el2n_time = (el2n_end_time - el2n_start_time)
                el2n_times[idx] += el2n_time

            
            # if args.el2n == 4 and global_round > 3 and train_accuracy[-1] > args.acc_thre1: # accuracyに応じてプルーニング,プルーニングしたいとき以外は入れたくない。
            #     local_model = local_models[idx]
            #     if pru_one < 10 and train_accuracy[-1] > args.acc_thre1 and train_accuracy[-1] < args.acc_thre2:
            #         print("pru_one:")
            #         percent = 0.25
            #         keep_idxs = local_model.compute_el2n_scores_2(global_model,percent,org_num)
            #         pru_one = pru_one + 1
            #     if pru_two < 10 and train_accuracy[-1] > args.acc_thre2 and train_accuracy[-1] < args.acc_thre3:
            #         print("pru_two:")
            #         percent = 0.5
            #         keep_idxs = local_model.compute_el2n_scores_2(global_model,percent,org_num)
            #         pru_two = pru_two + 1
            #     if pru_three < 10 and train_accuracy[-1] > args.acc_thre3 and train_accuracy[-1] < args.acc_thre3 + 0.5:
            #         print("pru_three:")
            #         percent = 0.75
            #         keep_idxs = local_model.compute_el2n_scores_2(global_model,percent,org_num)
            #         pru_three = pru_three + 1
            #     print("keep_idxs = ",keep_idxs)
            #     print("len(keep_idxs) = ",len(keep_idxs))
            #     local_model.update_dataset(keep_idxs, el2n=args.el2n)
            #     pru_num = org_num - len(keep_idxs)# すでに間引かれたデータセットの総数
            #     # print(f"User {idx}, keep: {keep_idxs[:10]}")
            #     print(f'Remaining dataset size after el2n thresholding: {len(keep_idxs)}')
            #     print("Pruning dataset size after el2n thresholding",pru_num)
            #     prun_num.append(pru_num)
            #     pri_pru[idx] = pru_num

            # if args.el2n == 4 and global_round > 3 and train_accuracy[-1] > args.acc_thre1: # accuracyに応じてプルーニング,プルーニングしたいとき以外は入れたくない。
            #     local_model = local_models[idx]

#多種の関数で間引く数を決める提案手法行うなら275行目までのコメントアウトを削除！
            if args.el2n == 4 and global_round > 3:
                el2n_start_time = time.monotonic()
                pru_func = a1 * stock_accuracy[idx] + b1 #　間引きたい数を表す(線形)
                # pru_func = a2 * (stock_accuracy[idx]) ** 2 + b2 #間引きたい数を表す（二次関数）
            #     # pru_func = 56 * train_accuracy[-1] - 33.6 #　間引きたい数を表す(線形)
            #     # pru_func = 120 * (train_accuracy[-1] - 0.6 ) ** 2 #6割pruning
            #     # pru_func = 140 * (train_accuracy[-1] - 0.6 ) ** 2 #7割pruning
            #     # pru_func = 30 * train_accuracy[-1] ** 2 -10.8  #6割pruning
                # pru_func = a3 ** stock_accuracy[idx]  + b3  #指数関数
                print("pru_func = ",pru_func)
            #     # print("pru_func - pru_func_stock[idx] = ",pru_func - pru_func_stock[idx])
            if args.el2n == 4 and global_round > 3  and (pru_func - pru_func_stock[idx]) > 1 and stock_accuracy[idx] > 0.6 and pru_func > 0:
                #batchからpruningする数が1以上のとき実行。
                local_model = local_models[idx]
                # percent = 0.25
                # bs = args.local_bs
                pru_batch = pru_func - pru_func_stock[idx]
                # print("pru_batch = ",pru_batch)
                keep_idxs = local_model.compute_el2n_scores_3(global_model,pru_batch)
                # print("keep_idxs = ",keep_idxs)
                # print("len(keep_idxs) = ",len(keep_idxs))
                if len(keep_idxs) <= args.local_bs:
                    choice_users.remove(idx)
                    continue
                local_model.update_dataset(keep_idxs, el2n=args.el2n)                
                pru_num = org_num - len(keep_idxs)# すでに間引かれたデータセットの総数
                # print(f"User {idx}, keep: {keep_idxs[:10]}")
                print(f'Remaining dataset size after el2n thresholding: {len(keep_idxs)}')
                print("Pruning dataset size after el2n thresholding",pru_num)
                prun_num.append(pru_num)
                pri_pru[idx] = pru_num
                pru_func_stock[idx] = pru_func
                # print("pru_func_stock = ",pru_func_stock[idx])
                el2n_end_time = time.monotonic()
                el2n_time = (el2n_end_time - el2n_start_time)
                el2n_times[idx] += el2n_time
        #el2nの数値によって決める
            if len(keep_idxs) <= args.local_bs and args.el2n == 1 and global_round == 10: #一回プルーニング
                print("len(keep_idxs) <= local_bs")
                t2 = time.monotonic()
                e = el2n_end_time - el2n_start_time
                print(f"EL2N: {e:.3f}s, Train: {train_time:.3f}s")
                fin = 1
                break

            if len(keep_idxs) <= args.local_bs and args.el2n == 3 and global_round == 10: #既存手法
                print("len(keep_idxs) <= local_bs")
                t2 = time.monotonic()
                e = el2n_end_time - el2n_start_time
                print(f"EL2N: {e:.3f}s, Train: {train_time:.3f}s")
                fin = 1
                break

            if len(keep_idxs) <= args.local_bs and args.el2n == 2 and global_round >= 10: #10epochs以降各回プルーニング
                print("len(keep_idxs) <= local_bs")
                t2 = time.monotonic()
                e = el2n_end_time - el2n_start_time
                print(f"EL2N: {e:.3f}s, Train: {train_time:.3f}s")
                fin = 1
                break

            local_model = local_models[idx]
            t_train_start = time.monotonic()
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            t_train_end = time.monotonic()
            train_time = (t_train_end - t_train_start)
            train_times[idx] += train_time


            # if len(keep_idxs) <= args.local_bs and args.el2n == 4 and global_round >= 10 and pru_one != 0 and pru_two != 0 and pru_three != 0: # accuracyに応じてプルーニング
            #     print("len(keep_idxs) <= local_bs")
            #     t2 = time.monotonic()
            #     te = t1 - t0
            #     tt = t2 - t1
            #     print(f"EL2N: {te:.3f}s, Train: {tt:.3f}s")
            #     ta = tt + ta + te
            #     te_all = te + te_all
            #     fin = 1
            #     break

            # if len(keep_idxs) <= args.local_bs and args.el2n == 4 and global_round >= 10 and pru_one != 0 and pru_two != 0 and pru_three != 0: # accuracyに応じてプルーニング
            #     print("len(keep_idxs) <= local_bs")
            #     t2 = time.monotonic()
            #     te = t1 - t0
            #     tt = t2 - t1
            #     print(f"EL2N: {te:.3f}s, Train: {tt:.3f}s")
            #     ta = tt + ta + te
            #     te_all = te + te_all
            #     fin = 1
            #     break

            print(f"EL2N: {el2n_time:.3f}s, Train: {train_time:.3f}s")
            # ta = train_time + el2n_time
            te_all = el2n_time + te_all

        if fin == 0:
            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger, client_id = idx)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))
            # 関数プルーニング前回との比較用
            if args.el2n == 4 and global_round > 3:
                for idx in idxs_users:
                    stock_accuracy[idx] = sum(list_acc)/len(list_acc)


            # print global training loss after every 'i' rounds
            
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            global_round = global_round + 1

            print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
            # import sys
            # sys.exit(0)
            # 各ラウンド終了後に評価指標を記録
            mem_mb, model_kb = log_metrics(
                round_num=epoch + 1,
                train_time=time.time() - start_time,
                model=global_model,
                accuracies=list_acc,
                log_path="logs")
            # ラウンドごとにリストへ追加
            mem_history.append(mem_mb)
            size_history.append(model_kb)
        else :
            break

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    end_time = time.time()
    all_time = end_time - start_time
    # calculate pruning data
    # total_pru = len(self.trainloader.dataset)
    total_el2n =sum(el2n_times.values())
    total_train = sum(train_times.values())
    print(f' \n Results after {global_round} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    # print("train time =",ta)
    print(f"Train time = : {total_train:.3f}s")
    print(f"el2n time = : {total_el2n:.3f}s")
    print(f"Total time = : {all_time:.3f}s")
    if args.el2n !=0 and global_round >=10:
        for idx in range(args.num_users):
            print("Pruning dataset:",pri_pru[idx])
            total_pru = total_pru + pri_pru[idx]
        print("Total Pruning dataset",total_pru)
    
    
    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_Crient[{}]_iid[{}]_Epoch[{}]_Batch_s[{}]_lr[{}]__el2n[{}]_num_users[{}]_threshold[{}]_percent[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.lr ,args.el2n, args.num_users, args.threshold,args.percent)
    if args.el2n != 0:
        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy, test_acc, keep_idxs, prun_num, args.threshold, pri_pru,args.percent], f)
    else :    
        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy, test_acc, keep_idxs, prun_num, args.threshold], f)

        # 実行コマンド
    report_lines.append(f"```")
    report_lines.append(f"python src/federated_main.py "
                        f"--dataset {args.dataset} "
                        f"--el2n {args.el2n} "
                        f"--epoch {args.epochs} "
                        f"--threshold {args.threshold} "
                        f"--pru_percent {args.pru_percent} "
                        f"--percent {args.percent} "
                        f"--start_accuracy {args.start_accuracy}")
    report_lines.append(f"```")

    # 実行環境と基本結果
    report_lines += [
        "",
        f"training on fugu",
        f"Results after {global_round} global rounds of training:",
        f"|---- Avg Train Accuracy      : {100 * train_accuracy[-1]:.2f}%",
        f"|---- Training Loss           : {np.mean(np.array(train_loss)):.4f}",
        f"|---- Test Accuracy           : {100 * test_acc:.2f}%",
        f"|---- Train time              : {total_train:.3f}s",
        f"|---- el2n time               : {total_el2n:.3f}s",
        f"|---- Total time              : {all_time:.3f}s",
        f"|---- Pruning dataset         : {pri_pru}",
        f"|---- Total Pruning dataset   : {total_pru}",
        f"|---- num_client_dataset      : {args.num_per_client}",
    ]
    # モデルメモリ関連情報
    avg_mem = sum(mem_history) / len(mem_history)
    final_mem = mem_history[-1]
    avg_size = sum(size_history) / len(size_history)
    final_size = size_history[-1]

    report_lines += [
        "",
        f"|---- 平均メモリ使用量         : {avg_mem:.1f} MB",
        f"|---- 最終ラウンドメモリ使用量 : {final_mem:.1f} MB",
        f"|---- 平均モデルサイズ         : {avg_size:.1f} KB",
        f"|---- 最終ラウンドモデルサイズ : {final_size:.1f} KB",
    ]

    # レポート送信
    report = "\n".join(report_lines)
    slackPost(report)

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))


# # 470行目付近に以下のコードを追加
# # featuresを生成する関数

# def generate_features(model, dataset, device):
#     """
#     Generate feature vectors for a given dataset using the provided model.

#     Args:
#     - model: Trained model to extract features.
#     - dataset: Dataset for which features are to be generated.
#     - device: Device to run the model on (e.g., 'cpu' or 'cuda').

#     Returns:
#     - features: A 2D numpy array of shape (num_samples, num_features).
#     """
#     model.eval()
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
#     features = []

#     with torch.no_grad():
#         for images, _ in dataloader:
#             images = images.to(device)
#             outputs = model(images)
#             features.append(outputs.cpu().numpy())

#     return np.vstack(features)
