import numpy as np
from sklearn.cluster import MiniBatchKMeans
from numpy.random import default_rng
from collections import defaultdict

class IncrementalCCS:
    def __init__(self, num_groups=100, seed=42, batch_size=10000):
        self.num_groups = num_groups
        self.rng = default_rng(seed)   # ← 乱数ジェネレータを定義
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_groups,
            batch_size=batch_size,
            random_state=seed,
            init='k-means++',
            n_init=1,            # インクリメンタルなので毎回完全再初期化しない
            max_no_improvement=None
        )
        self.fitted = False
    def update_and_select(self, scores: np.ndarray, num_to_keep: int,
                            global_to_local: list, keep_ratio_per_client: dict):
            """
            Args:
                scores: np.ndarray of shape (n_samples,) - EL2Nスコア
                num_to_keep: int - 全体で保持するデータ数
                global_to_local: List of (client_id, local_idx)
                keep_ratio_per_client: dict[int, float] - 各クライアントの保持比率（0〜1）

            Returns:
                keep_indices: dict[int, list[int]] - client_idごとの保持local indexリスト
            """
            normalized = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
            X = normalized.reshape(-1, 1)

            if not self.fitted:
                self.kmeans.fit(X)
                self.fitted = True
            else:
                self.kmeans.partial_fit(X)

            labels = self.kmeans.predict(X)
            n = len(scores)

            cluster_sizes = np.bincount(labels, minlength=self.num_groups)
            ideal = cluster_sizes * (num_to_keep / n)
            base = np.floor(ideal).astype(int)
            remainders = ideal - base
            deficit = num_to_keep - base.sum()
            if deficit > 0:
                order = np.argsort(-remainders)
                for i in order[:deficit]:
                    base[i] += 1

            keep_global_indices = []
            for cid, cnt in enumerate(base):
                if cnt <= 0:
                    continue
                idxs = np.where(labels == cid)[0]
                cnt = min(cnt, len(idxs))
                keep_global_indices.extend(self.rng.choice(idxs, size=cnt, replace=False))

            # クライアントごとの保持割当を適用
            per_client_indices = defaultdict(list)
            tmp = defaultdict(list)
            for gidx in keep_global_indices:
                client_id, local_idx = global_to_local[gidx]
                tmp[client_id].append(local_idx)

            for client_id, local_idxs in tmp.items():
                ratio = keep_ratio_per_client.get(client_id, 1.0)
                num = int(len(local_idxs) * ratio)
                num = min(num, len(local_idxs))
                per_client_indices[client_id] = list(self.rng.choice(local_idxs, size=num, replace=False))

            return per_client_indices

    #インクリメンタルCCSのみの場合
    # def update_and_select(self, scores: np.ndarray, num_to_keep: int):
    #     # 1) 0–1 正規化
    #     normalized = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
    #     X = normalized.reshape(-1, 1)  # ← (n_samples, n_features)

    #     # 2) 初回は fit、以降は partial_fit
    #     if not self.fitted:
    #         self.kmeans.fit(X)
    #         self.fitted = True
    #     else:
    #         self.kmeans.partial_fit(X)

    #     # 3) 各点を最寄りのクラスタ中心に割り当て
    #     labels = self.kmeans.predict(X)
    #     n = len(scores)

    #     # 4) 各クラスタの理想割当数を計算
    #     cluster_sizes = np.bincount(labels, minlength=self.num_groups)
    #     ideal = cluster_sizes * (num_to_keep / n)
    #     base = np.floor(ideal).astype(int)
    #     remainders = ideal - base
    #     deficit = num_to_keep - base.sum()
    #     if deficit > 0:
    #         order = np.argsort(-remainders)
    #         for i in order[:deficit]:
    #             base[i] += 1

    #     # 5) 各クラスタからサンプリング
    #     keep = []
    #     for cid, cnt in enumerate(base):
    #         if cnt <= 0:
    #             continue
    #         idxs = np.where(labels == cid)[0]
    #         cnt = min(cnt, len(idxs))
    #         keep.extend(self.rng.choice(idxs, size=cnt, replace=False))

    #     return np.array(keep, dtype=int)

# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.metrics import pairwise_distances_argmin_min
# from numpy.random import default_rng

# def coverage_centric_selection(scores, num_to_keep, num_groups=100, seed=42, use_greedy=False):
#     from numpy.random import default_rng
#     rng = default_rng(seed)

#     n = len(scores)
#     normalized = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
#     labels = MiniBatchKMeans(n_clusters=num_groups,
#                              batch_size=10000,
#                              random_state=seed).fit_predict(normalized.reshape(-1,1))

#     # 1) 各クラスタの要素数
#     cluster_sizes = np.bincount(labels, minlength=num_groups)

#     # 2) 各クラスタの理想的な割当（浮動小数点）
#     ideal = cluster_sizes * (num_to_keep / n)

#     # 3) 整数部分と小数部分に分離
#     base = np.floor(ideal).astype(int)
#     remainders = ideal - base

#     # 4) 切り捨て分の合計と残差を計算
#     deficit = num_to_keep - base.sum()

#     # 5) 残差が大きいクラスタ上位から +1 を割り当て
#     #    deficit 分だけ追加
#     if deficit > 0:
#         # 小数部分が大きいクラスタのインデックスをソート
#         order = np.argsort(-remainders)
#         for i in order[:deficit]:
#             base[i] += 1

#     # 6) 各クラスタから実際にサンプリング
#     keep = []
#     for cid, cnt in enumerate(base):
#         idxs = np.where(labels == cid)[0]
#         if len(idxs) == 0 or cnt == 0:
#             continue
#         # cnt がクラスタサイズを上回らないように
#         cnt = min(cnt, len(idxs))
#         keep.extend(rng.choice(idxs, size=cnt, replace=False))

#     return np.array(keep, dtype=int)

# def coverage_centric_selection(scores, num_to_keep, num_groups=100, seed=42, use_greedy=False):
#     # """
#     # Perform coverage-centric selection using KMeans clustering on scores.

#     # Args:
#     # - scores: 1D numpy array of scores.
#     # - num_to_keep: Total number of data points to retain.
#     # - num_groups: Number of clusters to form.

#     # Returns:
#     # - keep_indices: Indices of the data points to retain.
#     # """
#     # scores = scores.reshape(-1, 1)  # KMeans expects 2D array

#     # # クラスタリング
#     # kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init='auto')
#     # cluster_labels = kmeans.fit_predict(scores)

#     # keep_indices = []

#     # # 各クラスタから均等にデータを選ぶ
#     # for cluster_id in range(num_groups):
#     #     cluster_indices = np.where(cluster_labels == cluster_id)[0]
#     #     if len(cluster_indices) == 0:
#     #         continue  # 空クラスタに注意
#     #     num_to_select = max(1, len(cluster_indices) * num_to_keep // len(scores))
#     #     selected = np.random.choice(cluster_indices, size=min(num_to_select, len(cluster_indices)), replace=False)
#     #     keep_indices.extend(selected)
#     rng = default_rng(seed)
#     n = len(scores)
#     # 重み付きスコア（例：正規化＋EL2N混合）
#     range_ = np.ptp(scores)
#     normalized = (scores - scores.min()) / (range_ + 1e-8) # NumPy 2.0 以降はこの関数を使う
#     # クラスタリング
#     kmeans = MiniBatchKMeans(n_clusters=num_groups, batch_size=10000, random_state=seed)
#     labels = kmeans.fit_predict(normalized.reshape(-1,1))
#     keep = []

#     if use_greedy:
#         # Greedy k-center
#         centers = [rng.integers(n)]
#         dists = np.linalg.norm(normalized - normalized[centers[0]], axis=0)
#         for _ in range(num_to_keep-1):
#             nxt = np.argmax(dists)
#             keep.append(nxt)
#             new_d, _ = pairwise_distances_argmin_min(normalized.reshape(-1,1), normalized[nxt].reshape(1,-1))
#             dists = np.minimum(dists, new_d)
#     else:
#         # 各クラスタから重み付き選抜
#         counts = np.maximum(1, (np.bincount(labels, minlength=num_groups) * num_to_keep) // n)
#         for cid, cnt in enumerate(counts):
#             idxs = np.where(labels==cid)[0]
#             if len(idxs)==0: continue
#             keep.extend(rng.choice(idxs, size=min(cnt,len(idxs)), replace=False))

#     return np.array(keep,dtype=int)