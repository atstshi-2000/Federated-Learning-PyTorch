import numpy as np
from collections import defaultdict
from numpy.random import default_rng
from sklearn.cluster import MiniBatchKMeans

class IncrementalCCS:
    def __init__(self, num_groups=100, seed=42, batch_size=10000):
        self.num_groups = num_groups
        self.rng = default_rng(seed)
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_groups,
            batch_size=batch_size,
            random_state=seed,
            init='k-means++',
            n_init=1,
            max_no_improvement=None
        )
        self.fitted = False

    def update_and_select(
        self,
        scores: np.ndarray,
        num_to_keep: int,
        global_to_local: list,
        keep_ratio_per_client: dict
    ):
        """
        スコアのクラスタリングによる多様性確保 ＋ 
        keep_ratio_per_client に基づく直接的な割合分配を行う。
        """
        # 1) スコアを正規化してクラスタリング
        normalized = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
        X = normalized.reshape(-1, 1)

        if not self.fitted:
            self.kmeans.fit(X)
            self.fitted = True
        else:
            self.kmeans.partial_fit(X)

        labels = self.kmeans.predict(X)
        N_total = len(scores)

        # 2) 各クラスタから "おおよそ num_to_keep × (クラスタサイズ／N_total)" を選ぶ
        cluster_sizes = np.bincount(labels, minlength=self.num_groups)
        ideal = cluster_sizes * (num_to_keep / N_total)
        base = np.floor(ideal).astype(int)
        remainders = ideal - base
        deficit = num_to_keep - base.sum()
        if deficit > 0:
            order = np.argsort(-remainders)
            for i in order[:deficit]:
                base[i] += 1

        keep_global = []
        for cl in range(self.num_groups):
            cnt = base[cl]
            if cnt <= 0:
                continue
            idxs = np.where(labels == cl)[0]
            cnt = min(cnt, len(idxs))
            keep_global.extend(self.rng.choice(idxs, size=cnt, replace=False))

        # 3) global_to_local を使ってクライアントごとの「元データ数」をカウント（必要に応じて使うが、今回は参照のみ）
        original_count = defaultdict(int)
        for (cid, _) in global_to_local:
            original_count[cid] += 1

        # 4) ここで「各クライアントの最終保持目標数」を直接計算
        #    keep_ratio_per_client の合計は 1.0 である想定
        target_keep = {}
        for cid, orig_n in original_count.items():
            # num_to_keep をそのまま比率で分配
            k = int(np.floor(num_to_keep * keep_ratio_per_client.get(cid, 0.0)))
            # 最大でも orig_n を超えないように
            k = min(k, orig_n)
            target_keep[cid] = k

        # 5) 「クラスタリングで選ばれた keep_global の中から、クライアントごとに target_keep[cid] 件だけ残す」
        kept_count = {cid: 0 for cid in original_count}
        final_keep_global = []
        self.rng.shuffle(keep_global)  # シャッフルしてランダムに取り出す

        for gidx in keep_global:
            cid, local_idx = global_to_local[gidx]
            if kept_count[cid] < target_keep.get(cid, 0):
                final_keep_global.append(gidx)
                kept_count[cid] += 1
            # すでに目標数に達していればスキップ

        # 6) グローバルインデックスを「クライアント → ローカル idx リスト」に変換して返却
        per_client_indices = defaultdict(list)
        for gidx in final_keep_global:
            cid, local_idx = global_to_local[gidx]
            per_client_indices[cid].append(local_idx)

        return per_client_indices
    # def update_and_select(
    #         self,
    #         scores: np.ndarray,
    #         num_to_keep: int,
    #         global_to_local: list,
    #         keep_ratio_per_client: dict
    #     ):
    #         """
    #         旧版クラスタリングロジックを使いつつ、各クライアントごとに
    #         「元データ数 × keep_ratio_per_client[cid]」に達するまで保持し、
    #         それ以降は候補から除外するフローを実装。

    #         Args:
    #             scores: ndarray (N_total,) - 全サンプルの EL2N スコア
    #             num_to_keep: int           - 全体でおおよそ残したいサンプル数
    #             global_to_local: list of (client_id, local_idx) of length N_total
    #                 → インデックス gidx (0 <= gidx < N_total) がどのクライアントのどのローカル idx
    #                 かを示すマッピング
    #             keep_ratio_per_client: dict { cid: float (0~1) }
    #                 → 「各クライアント c は、元データ数の何割を最終的に残すか」を示す

    #         Returns:
    #             per_client_indices: dict { cid: [local_idx, ...] }
    #                 → “最終的に各クライアントが保持すべきローカルインデックス” のリスト
    #         """

    #         # ─── (1) 全体スコアを 0–1 正規化してクラスタリング ───
    #         normalized = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
    #         X = normalized.reshape(-1, 1)  # (N_total, 1)

    #         if not self.fitted:
    #             self.kmeans.fit(X)
    #             self.fitted = True
    #         else:
    #             self.kmeans.partial_fit(X)

    #         labels = self.kmeans.predict(X)
    #         N_total = len(scores)

    #         # ─── (2) 各クラスタに「おおよそ num_to_keep × (cluster_size / N_total)」件を割り当て ───
    #         cluster_sizes = np.bincount(labels, minlength=self.num_groups)
    #         ideal   = cluster_sizes * (num_to_keep / N_total)
    #         base    = np.floor(ideal).astype(int)
    #         remainders = ideal - base
    #         deficit = num_to_keep - base.sum()
    #         if deficit > 0:
    #             # 端数が大きい順に +1 して、合計が num_to_keep になるように調整
    #             order = np.argsort(-remainders)
    #             for i in order[:deficit]:
    #                 base[i] += 1

    #         # ─── (3) 各クラスタから base[cl] 件ずつランダムに sampling して keep_global を得る ───
    #         keep_global = []  # グローバルインデックスのリスト
    #         for cl in range(self.num_groups):
    #             cnt = base[cl]
    #             if cnt <= 0: 
    #                 continue
    #             idxs = np.where(labels == cl)[0]
    #             cnt = min(cnt, len(idxs))
    #             keep_global.extend(self.rng.choice(idxs, size=cnt, replace=False))

    #         # ─── (4) global_to_local を使って「元データ数」を各クライアントごとに数える ───
    #         #      global_to_local は長さ N_total のリスト (gidx→(cid, local_idx)) なので、
    #         #      ここで “元々各クライアントが持っていたサンプル数” を自動で計算
    #         original_count = defaultdict(int)  # cid -> 元々のサンプル数
    #         for (cid, _) in global_to_local:
    #             original_count[cid] += 1

    #         # ─── (5) 各クライアントごとに「最終的に保持すべき件数 target_keep[cid]」を計算 ───
    #         target_keep = {}
    #         for cid, orig_n in original_count.items():
    #             ratio = keep_ratio_per_client.get(cid, 1.0)
    #             # floor(orig_n * ratio) で整数化
    #             k = int(np.floor(orig_n * ratio))
    #             # もちろん元の件数を超えないように
    #             k = min(k, orig_n)
    #             target_keep[cid] = k

    #         # ─── (6) keep_global をシャッフルし、先着順で各クライアントの target_keep[cid] を満たすまで keep ───
    #         kept_count = {cid: 0 for cid in original_count}  # 保持済みカウント
    #         final_keep_global = []

    #         # 初期候補 keep_global を再度ランダムにシャッフルしておく（よりランダム性を高める）
    #         self.rng.shuffle(keep_global)

    #         for gidx in keep_global:
    #             cid, local_idx = global_to_local[gidx]
    #             # まだ kept_count[cid] < target_keep[cid] なら、このサンプルを keep
    #             if kept_count[cid] < target_keep[cid]:
    #                 final_keep_global.append(gidx)
    #                 kept_count[cid] += 1
    #             # すでに kept_count[cid] >= target_keep[cid] のときはスキップ（＝プルーニング）

    #         # ─── (7) 最終的な global_idx 列 final_keep_global を「各クライアントごとに local_idx リスト」に変換 ───
    #         per_client_indices = defaultdict(list)
    #         for gidx in final_keep_global:
    #             cid, local_idx = global_to_local[gidx]
    #             per_client_indices[cid].append(local_idx)

    #         return per_client_indices
    # # def update_and_select(
    #     self,
    #     scores: np.ndarray,
    #     num_to_keep: int,
    #     global_to_local: list,
    #     keep_ratio_per_client: dict
    # ):
    #     """
    #     簡易版：
    #       1) 全スコアを k-means でクラスタリング→ラベルを取得
    #       2) 各クラスタに「およそ num_to_keep * (cluster_size/total)」件をランダムに選ぶ
    #       3) その候補をクライアントごとに分け、各クライアントは floor(候補件数 * keep_ratio) 件をランダムに選ぶ
    #       （最終的に全体合計は num_to_keep 付近で止まるが厳密には合わせない。±5％程度のズレを許容）
    #     """
    #     # 1) 正規化＆クラスタリング
    #     normalized = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
    #     X = normalized.reshape(-1, 1)

    #     if not self.fitted:
    #         self.kmeans.fit(X)
    #         self.fitted = True
    #     else:
    #         self.kmeans.partial_fit(X)

    #     labels = self.kmeans.predict(X)
    #     n = len(scores)

    #     # 2) 各クラスタにおよそ振り分ける件数（端数まるめは floor だけ）
    #     cluster_sizes = np.bincount(labels, minlength=self.num_groups)
    #     # ideal[i] = cluster_sizes[i] * (num_to_keep / n)
    #     ideal = cluster_sizes * (num_to_keep / n)
    #     base = np.floor(ideal).astype(int)

    #     # 2-1) 各クラスタからランダムに base[i] 件だけピックアップ
    #     keep_global_indices = []
    #     for cl in range(self.num_groups):
    #         cnt = base[cl]
    #         if cnt <= 0:
    #             continue
    #         idxs = np.where(labels == cl)[0]
    #         # 実際のクラスタサイズよりも base[cl] が大きい場合は、クラスタサイズ分だけに制限
    #         cnt = min(cnt, len(idxs))
    #         # コストを抑えるため、np.random.choice の代わりに rng.choice で一発
    #         keep_global_indices.extend(self.rng.choice(idxs, size=cnt, replace=False))

    #     # 3) クライアントごとに「候補リスト tmp[cid]」を作成
    #     tmp = defaultdict(list)
    #     for gidx in keep_global_indices:
    #         cid, local_idx = global_to_local[gidx]
    #         tmp[cid].append(local_idx)

    #     # 4) 最終的に各クライアントは floor(len(tmp[cid]) * keep_ratio) 件を残す
    #     per_client_indices = {}
    #     for cid, local_idxs in tmp.items():
    #         ratio = keep_ratio_per_client.get(cid, 1.0)
    #         k = int(len(local_idxs) * ratio)
    #         if k <= 0:
    #             per_client_indices[cid] = []
    #         else:
    #             # k > len(local_idxs) にならないよう min
    #             k = min(k, len(local_idxs))
    #             per_client_indices[cid] = list(self.rng.choice(local_idxs, size=k, replace=False))

    #     # ※この時点で sum(len(v) for v in per_client_indices.values())
    #     #   は「およそ num_to_keep * (average keep_ratio)」程度になる。
    #     return per_client_indices

    # def update_and_select(self, scores: np.ndarray, num_to_keep: int,
    #                         global_to_local: list, keep_ratio_per_client: dict):
    #         """
    #         Args:
    #             scores: np.ndarray of shape (n_samples,) - EL2Nスコア
    #             num_to_keep: int - 全体で保持するデータ数
    #             global_to_local: List of (client_id, local_idx)
    #             keep_ratio_per_client: dict[int, float] - 各クライアントの保持比率（0〜1）

    #         Returns:
    #             keep_indices: dict[int, list[int]] - client_idごとの保持local indexリスト
    #         """
    #         normalized = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
    #         X = normalized.reshape(-1, 1)

    #         if not self.fitted:
    #             self.kmeans.fit(X)
    #             self.fitted = True
    #         else:
    #             self.kmeans.partial_fit(X)

    #         labels = self.kmeans.predict(X)
    #         n = len(scores)

    #         cluster_sizes = np.bincount(labels, minlength=self.num_groups)
    #         ideal = cluster_sizes * (num_to_keep / n)
    #         base = np.floor(ideal).astype(int)
    #         remainders = ideal - base
    #         deficit = num_to_keep - base.sum()
    #         if deficit > 0:
    #             order = np.argsort(-remainders)
    #             for i in order[:deficit]:
    #                 base[i] += 1

    #         keep_global_indices = []
    #         for cid, cnt in enumerate(base):
    #             if cnt <= 0:
    #                 continue
    #             idxs = np.where(labels == cid)[0]
    #             cnt = min(cnt, len(idxs))
    #             keep_global_indices.extend(self.rng.choice(idxs, size=cnt, replace=False))

    #         # クライアントごとの保持割当を適用
    #         per_client_indices = defaultdict(list)
    #         tmp = defaultdict(list)
    #         for gidx in keep_global_indices:
    #             client_id, local_idx = global_to_local[gidx]
    #             tmp[client_id].append(local_idx)

    #         for client_id, local_idxs in tmp.items():
    #             ratio = keep_ratio_per_client.get(client_id, 1.0)
    #             num = int(len(local_idxs) * ratio)
    #             num = min(num, len(local_idxs))
    #             per_client_indices[client_id] = list(self.rng.choice(local_idxs, size=num, replace=False))

    #         return per_client_indices

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