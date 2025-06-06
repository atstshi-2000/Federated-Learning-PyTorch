# client_task.py
import os
import torch
import time
import psutil
import pickle
import numpy as np
import argparse
import torchvision

# プロジェクト内の他モジュールをインポートするために、必要に応じてsys.pathを設定
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # srcディレクトリをパスに追加する例

from update import LocalUpdate # update.py から LocalUpdate をインポート
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, MLP # models.py からモデル定義をインポート
from utils import get_dataset # utils.py からデータセット取得関数をインポート
from options import args_parser as main_args_parser # options.py のパーサーをデフォルト値参照用にインポート

def client_specific_args_parser():
    parser = argparse.ArgumentParser(description="Client Training Task Script")
    # このスクリプト固有の必須引数
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    parser.add_argument('--global_model_path', type=str, required=True, help='Path to the saved global model state_dict')
    parser.add_argument('--current_epoch', type=int, required=True, help='Current global round/epoch number')
    parser.add_argument('--data_indices_path', type=str, required=True, help='Path to the pickle file containing data indices for this client')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save client outputs')

    # メインスクリプト (new_ccs.py) の options.py で定義された引数を引き継ぐ
    # これにより、lr, local_ep, local_bs などの共通パラメータを client_task.py でも利用できる
    temp_main_args_for_defaults = main_args_parser()
    for action in temp_main_args_for_defaults._actions:
        # 上記で明示的に定義した引数や 'help' は除外
        if action.dest not in ['client_id', 'global_model_path', 'current_epoch', 'data_indices_path', 'output_dir', 'help']:
            parser.add_argument(f'--{action.dest}', type=action.type, default=action.default, help=f"(Inherited) {action.help}")
    return parser

if __name__ == '__main__':
    args = client_specific_args_parser().parse_args()

    # このクライアントプロセス自身の情報を取得
    client_process_info = psutil.Process(os.getpid())

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
    # 2. データセットの準備 (メインスクリプトと同様のロジックでフルデータセットをロード)
    #    get_dataset は args (学習データセットの種類などを含む) を受け取ると仮定
    try:
        full_train_dataset, _, _ = get_dataset(args) # user_groups はここでは不要
    except Exception as e:
        print(f"Client {args.client_id}: Error loading dataset with args '{args.dataset}'. Error: {e}")
        exit(1)
    
    try:
        with open(args.data_indices_path, 'rb') as f:
            client_data_indices = pickle.load(f)
    except FileNotFoundError:
        print(f"Client {args.client_id}: Data indices file not found at {args.data_indices_path}. Exiting.")
        exit(1)
    except Exception as e:
        print(f"Client {args.client_id}: Error loading data indices from {args.data_indices_path}: {e}. Exiting.")
        exit(1)
        
    # print(f"Client {args.client_id}: Loaded {len(client_data_indices)} data indices.")

    # 3. グローバルモデルの構築とロード
    client_model = None
    try:
        if args.model == 'cnn':
            if args.dataset == 'mnist': client_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist': client_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar': client_model = torchvision.models.resnet18(pretrained=True)
        elif args.model == 'mlp':
            # データセットの最初の要素から画像サイズを取得
            if len(full_train_dataset) == 0: raise ValueError("Full training dataset is empty.")
            sample_img, _ = full_train_dataset[0]
            img_size = sample_img.shape
            len_in = 1
            for x_dim in img_size: len_in *= x_dim # 全要素数を計算
            client_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
        
        if client_model is None:
            raise ValueError(f"Model '{args.model}' for dataset '{args.dataset}' could not be created.")
        
        client_model.load_state_dict(torch.load(args.global_model_path, map_location='cpu'))
        client_model.to(device)
    except FileNotFoundError:
        print(f"Client {args.client_id}: Global model file not found at {args.global_model_path}. Exiting.")
        exit(1)
    except Exception as e:
        print(f"Client {args.client_id}: Error setting up model: {e}. Exiting.")
        exit(1)

    # 4. LocalUpdateインスタンスの作成 (logger は None)
    # LocalUpdateの__init__が dataset(フルデータセット)とidxs(このクライアントのグローバルインデックス)を正しく処理する前提
    try:
        local_updater = LocalUpdate(args=args, dataset=full_train_dataset, idxs=client_data_indices, 
                                    logger=None, client_id=args.client_id)
        local_updater.device = device # LocalUpdate内で再度device設定があるかもしれないので、ここで上書き
    except Exception as e:
        print(f"Client {args.client_id}: Error instantiating LocalUpdate: {e}. Exiting.")
        exit(1)

    # 5. ローカル学習の実行
    # update_weights は (weights_dict, loss, performance_metrics_dict) を返すように
    # update.py の LocalUpdate クラスが修正されている必要がある。
    try:
        updated_weights_dict, loss, perf_metrics = local_updater.update_weights(
            model=client_model, # サブプロセスなのでdeepcopyは不要
            global_round=args.current_epoch
        )
    except Exception as e:
        print(f"Client {args.client_id}: Error during local training (update_weights): {e}. Exiting.")
        # エラー時も、それまでの情報を保存する試み（オプション）
        perf_metrics = {
            'train_time': -1, 'num_samples': 0, 
            'process_cpu_usage': -1, 'client_ram_rss_mb': -1, 'error': str(e)
        }
        updated_weights_dict = client_model.state_dict() # 学習失敗時は元の重みを返す
        loss = float('inf')


    # 6. 結果の保存
    output_data = {
        'client_id': args.client_id,
        'weights': updated_weights_dict,
        'loss': loss,
        'perf_metrics': perf_metrics, # これに 'train_time', 'num_samples', 'process_cpu_usage', 'client_ram_rss_mb' が含まれる
        'data_indices_used_count': len(client_data_indices)
    }
    
    os.makedirs(args.output_dir, exist_ok=True) # output_dirの存在確認と作成
    client_output_filename = f'client_{args.client_id}_round_{args.current_epoch}_output.pkl'
    client_output_path = os.path.join(args.output_dir, client_output_filename)
    try:
        with open(client_output_path, 'wb') as f:
            pickle.dump(output_data, f)
        # print(f"Client {args.client_id}: Output successfully saved to {client_output_path}")
    except Exception as e:
        print(f"Client {args.client_id}: Error saving output to {client_output_path}: {e}")

    print(f"Client {args.client_id} [PID:{os.getpid()}] completed round {args.current_epoch}.")