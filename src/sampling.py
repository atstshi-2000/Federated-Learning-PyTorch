#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/num_users)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items,
    #                                          replace=False))
    #     all_idxs = list(set(all_idxs) - dict_users[i])
    # return dict_users
    all_idxs = list(range(len(dataset)))
    user_groups = {i: all_idxs.copy() for i in range(num_users)}
    return user_groups
# def mnist_iid(dataset, num_users, num_per_client=None, replace=False, seed=42):
#     """
#     Sample I.I.D. client data from MNIST dataset with configurable samples per client.

#     Args:
#       dataset         : torch Dataset (with __len__ defined)
#       num_users       : number of clients
#       num_per_client  : samples to assign to each client; if None, defaults to len(dataset)//num_users
#       replace         : whether to sample with replacement when num_per_client * num_users > len(dataset)
#       seed            : random seed for reproducibility

#     Returns:
#       dict_users: dict mapping client_id -> list of sample indices
#     """
#     n = len(dataset)
#     rng = np.random.default_rng(seed)

#     # デフォルトは均等分割
#     if num_per_client is None:
#         num_per_client = n // num_users

#     # 重複なしで割り当てたい量が総数を超える場合はエラーか置き換え動作
#     if not replace and num_per_client * num_users > n:
#         raise ValueError(
#             f"Cannot assign {num_per_client} samples to {num_users} clients "
#             f"without replacement (total {num_per_client*num_users} > {n})."
#         )

#     all_indices = np.arange(n)
#     dict_users = {}

#     for uid in range(num_users):
#         # サンプリング
#         chosen = rng.choice(all_indices, size=num_per_client, replace=replace)
#         dict_users[uid] = chosen.tolist()

#     return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # dir(dataset)
    # import sys
    # sys.exit(0)
    # labels = dataset.train_labels.numpy()
    labels = dataset.targets.numpy()
    # labels = dataset.train_list.numpy()


    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = dataset.targets.numpy()
    # labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     :トレーニングデータの数を管理している。
#     """
#     # num_items = int(len(dataset)/num_users)
#     # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     # for i in range(num_users):
#     #     dict_users[i] = set(np.random.choice(all_idxs, num_items,
#     #                                          replace=False))
#     #     all_idxs = list(set(all_idxs) - dict_users[i])
#     all_idxs = list(range(len(dataset)))
#     dict_users = {i: all_idxs.copy() for i in range(num_users)}
#     return dict_users
#     # print("dict users = ", len(dict_users[0]))
#     # exit(0)
#     # return dict_users
def cifar_iid(dataset, num_users, num_per_client=None, replace=True, seed=None):
    """
    Sample I.I.D. client data from CIFAR10 dataset with configurable samples per client.

    Args:
      dataset         : torch Dataset instance (must support len())
      num_users       : number of clients
      num_per_client  : number of samples to assign to each client;
                        if None, defaults to len(dataset)//num_users
      replace         : whether to sample with replacement
      seed            : random seed for reproducibility

    Returns:
      dict_users: dict mapping client_id -> list of sample indices
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)

    # デフォルトは均等分割
    if num_per_client is None:
        num_per_client = n // num_users

    # # 重複なしで総数を超える場合は例外
    # if not replace and num_per_client * num_users > n+1:
    #     raise ValueError(
    #         f"Cannot assign {num_per_client} unique samples to {num_users} clients "
    #         f"(requires {num_per_client*num_users} > {n+1})."
    #     )

    all_indices = np.arange(n)
    dict_users = {}

    for uid in range(num_users):
        chosen = rng.choice(all_indices, size=num_per_client, replace=replace)
        dict_users[uid] = chosen.tolist()

    return dict_users

def cifar_noniid(dataset, num_users, num_per_client=10000, replace=False, seed=42):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    #labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.train_labels)
    labels = np.array(dataset.targets)
    # labels = np.array(dataset.train_list)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # # divide and assign 元ネタ
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard,int(num_shards/num_users), replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     # print("set(idx_shard) ", len(set(idx_shard)))
    #     print("idx_shard = ", len(idx_shard))
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    # 新：任意件数サンプリング
    rng = np.random.default_rng(seed)
    all_idxs = np.arange(len(dataset))
    dict_users = {}
    for uid in range(num_users):
        dict_users[uid] = rng.choice(all_idxs, size=num_per_client,
                                     replace=replace)
    return dict_users
    # print("dict users = ", len(dict_users[0]))
    # exit(0)
    # return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
