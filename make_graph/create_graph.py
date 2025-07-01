import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

def parse_experiment_log(filepath="CCS 実験記録 - 1万～5万.csv"):
    """
    ログ形式のCSVファイルを解析し、実験結果のリストを返します。
    """
    experiments = []
    current_experiment = {}
    
    # Slackログからコマンドラインを抽出するためのパターン
    command_pattern = re.compile(r'python src/federated_main.py (.*)')
    
    # 各項目を抽出するための正規表現パターン
    patterns = {
        'Test_Accuracy': re.compile(r'Test Accuracy:\s*([0-9.]+)'),
        'Total_Time': re.compile(r'total time\s*:\s*([0-9.]+)'),
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # コマンドラインからハイパーパラメータを抽出
                cmd_match = command_pattern.search(line)
                if cmd_match:
                    # 進行中の実験があればリストに追加
                    if current_experiment:
                        experiments.append(current_experiment)
                    # 新しい実験のために辞書をリセット
                    current_experiment = {}
                    args_str = cmd_match.group(1)
                    # 各引数を抽出
                    current_experiment['local_ep'] = float(re.search(r'--local_ep\s+(\d+)', args_str).group(1)) if re.search(r'--local_ep\s+(\d+)', args_str) else 3 # デフォルト値
                    current_experiment['prune_rate'] = float(re.search(r'--prune_rate\s+([0-9.]+)', args_str).group(1)) if re.search(r'--prune_rate\s+([0-9.]+)', args_str) else 0.0 # デフォルト値
                    current_experiment['el2n'] = float(re.search(r'--el2n\s+(\d+)', args_str).group(1)) if re.search(r'--el2n\s+(\d+)', args_str) else 0 # デフォルト値

                # 数値結果を抽出
                for key, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        current_experiment[key] = float(match.group(1))

            if current_experiment: # ファイル末尾の最後の実験データをリストに追加
                experiments.append(current_experiment)
        
        if not experiments:
            print("エラー: ログファイルから有効な実験データを抽出できませんでした。")
            return None

        df = pd.DataFrame(experiments)
        # el2n=0 の場合、prune_rateを0に設定
        df.loc[df['el2n'] == 0, 'prune_rate'] = 0.0
        
        print(f"{len(df)}件の実験データを正常に抽出しました。")
        return df

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
        return None
    except Exception as e:
        print(f"ファイルの解析中にエラーが発生しました: {e}")
        return None

def create_comparison_graph(df):
    """
    指定されたDataFrameから、テスト精度と学習時間の比較グラフを生成します。
    """
    if df is None or df.empty:
        print("データが空のため、グラフを作成できません。")
        return

    # --- データのフィルタリングと準備 ---
    # local_epが3のデータのみを対象
    df_plot = df[df['local_ep'] == 3].copy()
    if df_plot.empty:
        print("local_ep=3 の実験データが見つかりませんでした。")
        return
        
    # プルーニング率でソート
    df_plot = df_plot.sort_values(by='prune_rate').reset_index(drop=True)

    # グラフ描画用にデータを抽出
    t = df_plot['prune_rate'].tolist()
    y1 = df_plot['Test_Accuracy'].tolist() # test_acc
    y2 = df_plot['Total_Time'].tolist()    # train time

    # --- Matplotlibの日本語フォント設定 ---
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Noto Sans CJK JP']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 日本語フォントの設定に失敗しました。グラフの日本語が文字化けする可能性があります。")

    # --- グラフ描画 (ご提示のコードスタイルを再現) ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    c1, c2 = "dodgerblue", "tomato"
    l1, l2 = "テスト精度 (Test Accuracy)", "総学習時間 (Total Time)"

    # 1つ目のY軸 (テスト精度) - 折れ線グラフ
    ax1.plot(t, y1, color=c1, label=l1, marker='o', linestyle='-')
    ax1.set_xlabel('プルーニング率 (Pruning Rate)', fontsize=12)
    ax1.set_ylabel(l1, fontsize=12, color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2つ目のY軸 (学習時間) - 棒グラフ
    ax2 = ax1.twinx()
    bar_width = 0.01 # 棒グラフの幅を調整
    ax2.bar(t, y2, color=c2, label=l2, alpha=0.6, width=bar_width)
    ax2.set_ylabel(l2, fontsize=12, color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)

    # グラフの体裁を整える
    plt.title('プルーニング率と精度・学習時間の関係 (local_ep=3)', fontsize=16, pad=20)
    ax1.set_xticks(t) # x軸の目盛りをプルーニング率に合わせる
    ax1.set_xticklabels([f'{x:.2f}' for x in t]) # x軸のラベル表示形式
    
    # 凡例を一つにまとめる
    h1, lbl1 = ax1.get_legend_handles_labels()
    h2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, lbl1 + lbl2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.savefig('comparison_graph_ep3.png', bbox_inches='tight')
    print("グラフ 'comparison_graph_ep3.png' を保存しました。")
    plt.close()

# --- メイン処理 ---
if __name__ == '__main__':
    # 1. ログファイルからデータを抽出
    experiment_df = parse_experiment_log()
    
    # 2. グラフを生成
    create_comparison_graph(experiment_df)
