#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=50,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=3,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=1,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--el2n', type=int, default=0,help='pruning 0はpruningなし 1はthreで一回 2は各回、3は既存研究、4は精度でのPruning')
    parser.add_argument('--threshold', type=float, default=0.5,help='pruning_threshold')
    parser.add_argument('--percent', type=float, default=0.5,help='pruning_percentage')
    parser.add_argument('--acc_thre1', type=float, default=0.7,help='accuracy pruning first')
    parser.add_argument('--acc_thre2', type=float, default=0.75,help='accuracy pruning second')
    parser.add_argument('--acc_thre3', type=float, default=0.8,help='accuracy pruning third')
    parser.add_argument('--start_accuracy', type=float, default=0.6,help='pruning start accuracy')
    parser.add_argument('--pru_percent', type=float, default=0.6,help='pruning max percent')
    parser.add_argument('--num_per_client', type=int, default=12000,help='number of samples per client ただし、20%のデータが訓練時の精度評価に使われる。')
    parser.add_argument('--prune_rate', type=float, default=0.1,help='CCSのprune_rate')
    
    # NEW: Subprocess and CPU Limit related arguments
    parser.add_argument('--client_script_path', type=str, default='src/update_sub.py', # CHANGED: 正しいパスに
                        help="Path to the client training script (relative to project root)")
    parser.add_argument('--temp_io_dir', type=str, default='./client_io_temp_ccs', 
                        help="Directory for temporary client I/O files")
    parser.add_argument('--client_cpu_limits_str', type=str, default="{}", 
                        help='String representation of a dict for client CPU limits e.g., "{0:20, 1:50, 2:80, 3:70, 4:40, 5:70, 6:100, 7:50, 8:80, 9:90, 10:70}"')
    parser.add_argument('--default_client_cpu_limit', type=int, default=100, 
                        help="Default CPU limit (%) if not specified for a client")
    parser.add_argument('--client_timeout', type=int, default=100000000000, 
                        help="Timeout in seconds for client subprocess execution")

    # NEW: Burden-aware CCS arguments (前回提案と同様)
    parser.add_argument('--burden_alpha', type=float, default=0.3, help="Smoothing factor")
    parser.add_argument('--burden_w_time', type=float, default=0.4, help="Weight for time_per_sample")
    parser.add_argument('--burden_w_cpu', type=float, default=0.3, help="Weight for process_cpu_usage")
    parser.add_argument('--burden_w_ram', type=float, default=0.3, help="Weight for client_ram_rss_mb")
    parser.add_argument('--min_keep_ratio', type=float, default=0.2, help="Min keep_ratio")
    parser.add_argument('--max_keep_ratio', type=float, default=1.0, help="Max keep_ratio")
    parser.add_argument('--default_keep_ratio', type=float, default=0.8, help="Default keep_ratio")

    # NEW: CCS timing and pruning control (前回提案と同様)
    parser.add_argument('--start_prune_round', type=int, default=10, help="Global round to start CCS")
    parser.add_argument('--prune_interval', type=int, default=10, help="CCS interval")
    # args.prune_rate は既存のものを流用
    parser.add_argument('--max_overall_reduction_rate', type=float, default=0.6, help="Max global data reduction") # 元の0.1から変更
    parser.add_argument('--min_samples_after_global_prune', type=int, default=20, help="Min samples kept globally post-CCS")
    parser.add_argument('--num_groups_ccs', type=int, default=100, help="Num clusters for CCS")
    args = parser.parse_args()
    return args
