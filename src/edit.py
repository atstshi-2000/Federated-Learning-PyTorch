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
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, f"round_{round_num}.json"), "w") as f:
        json.dump(log_data, f, indent=4)

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
    local_models = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, client_id=idx) for idx in range(args.num_users)]
    
    # クライアントオブジェクトのリストを作成
    clients = [LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, client_id=idx) for idx in range(args.num_users)]
    
    # if args.dataset == 'cifar':
    org_num = 40000 / args.num_users
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        org_num = 48000 / args.num_users
    el2n = args.el2n
    percent = args.percent

    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        if len(choice_users) <= 4 :
            print("no crient")
            break
        idxs_users = np.random.choice(choice_users, m, replace=False)        
        print("idxs_users = ", idxs_users)
        for idx in idxs_users:
            t0 = time.monotonic()

            local_model = local_models[idx]  # そもそもこのループは各クライアント毎に回っている。
            
            if global_round >= 10 and global_round % 10 == 0:
                # クライアントごとに1次元配列を格納
                scores = local_model.compute_el2n_scores_4(global_model)
                scores_np = scores.detach().cpu().numpy().tolist()
                el2n_scores[idx] = scores_np
           
            t1 = time.monotonic()

            local_model = local_models[idx]
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)  # update_weights内でtrainを実行している
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            t2 = time.monotonic()
            te = t1 - t0
            tt = t2 - t1
            print(f"EL2N: {te:.3f}s, Train: {tt:.3f}s")
            ta = tt + ta + te
            te_all = te + te_all


            if 'el2n_scores' in locals() and global_round >= 10 and global_round % 10 == 0:
                # 全 client のスコアを flatten しつつマッピング
                flattened_scores = []
                global_to_local_map = []
                for uid, scores_list in el2n_scores.items():
                    for local_i, sc in enumerate(scores_list):
                        flattened_scores.append(sc)
                        global_to_local_map.append((uid, local_i))
                flattened_scores = np.array(flattened_scores)
                
                # 正規化（global min-max）
                mn, mx = flattened_scores.min(), flattened_scores.max()
                normalized_scores = (flattened_scores - mn) / (mx - mn + 1e-8)

                # CCS のパラメータ設定
                total_data_points = len(flattened_scores)
                prune_rate = 0.20
                num_to_keep = int(total_data_points * (1 - prune_rate * (epoch // 10)))
                num_groups = 100

                # CCS 実行
                keep_global_indices = coverage_centric_selection(
                    flattened_scores,
                    num_to_keep=num_to_keep,
                    num_groups=num_groups
                )
                # クライアントごとのローカルインデックスに変換
                from collections import defaultdict
                client_local_keep_indices = defaultdict(list)

                for global_idx in keep_global_indices:
                    client_id, local_idx = global_to_local_map[global_idx]
                    client_local_keep_indices[client_id].append(local_idx)



                # 対応するインデックスが存在すれば使う、なければ全データを使用
                if idx in client_local_keep_indices:
                    keep_indices = client_local_keep_indices[idx]
                else:
                    keep_indices = list(range(len(local_model.train_dataset)))  # または空リストなどの処理

                    # データセットの再構成
                local_model.update_dataset(keep_indices, el2n=args.el2n)
                
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

            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            global_round = global_round + 1

            print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
            # import sys
            # sys.exit(0)
            # 各ラウンド終了後に評価指標を記録
            log_metrics(
                round_num=epoch + 1,
                train_time=time.time() - start_time,
                model=global_model,
                accuracies=list_acc,
                log_path="logs"
            )
        
        else :
            break

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {global_round} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    # print("train time =",ta)
    print(f"Train time = : {ta:.3f}s")
    print(f"el2n time = : {te_all:.3f}s")

    
    
    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_Client[{}]_iid[{}]_Epoch[{}]_Batch_s[{}]_lr[{}]__el2n[{}]_num_users[{}]_threshold[{}]_percent[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.lr ,args.el2n, args.num_users, args.threshold,args.percent)
    if args.el2n != 0:
        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy, test_acc, keep_idxs, prun_num, args.threshold, pri_pru,args.percent], f)
    else :    
        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy, test_acc, keep_idxs, prun_num, args.threshold], f)
    

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    
    slackPost(f'python src/federated_main.py --dataset {args.dataset} --el2n {args.el2n} --epoch {args.epochs} --threshold {args.threshold} --pru_percent {args.pru_percent} --percent {args.percent} --start_accuracy {args.start_accuracy} \n training on fugu \n Results after {global_round} global rounds of training: \n |---- Avg Train Accuracy: {100*train_accuracy[-1]:.2f} \n Training Loss : {np.mean(np.array(train_loss))} \n |---- Test Accuracy:{100*test_acc:.2f} \n Train time = : {ta:.3f}s \n el2n time = : {te_all:.3f}s \n 各クライアントデータセット \n {len(client_local_keep_indices[1])} \n {len(client_local_keep_indices[2])} \n {len(client_local_keep_indices[3])} \n {len(client_local_keep_indices[4])} \n {len(client_local_keep_indices[5])} \n {len(client_local_keep_indices[6])} \n {len(client_local_keep_indices[7])} \n {len(client_local_keep_indices[8])} \n {len(client_local_keep_indices[9])} \n {len(client_local_keep_indices[10])}')

    if args.el2n !=0 and global_round >=10:
        for idx in range(args.num_users):
            keep_list = client_local_keep_indices.get(idx, [])
            print("Pruning dataset:", keep_list)
            total_pru += len(keep_list)