from sklearn.cluster import KMeans
import numpy as np

def coverage_centric_selection(scores, num_to_keep, num_groups=100):
    """
    Perform coverage-centric selection using KMeans clustering on scores.

    Args:
    - scores: 1D numpy array of scores.
    - num_to_keep: Total number of data points to retain.
    - num_groups: Number of clusters to form.

    Returns:
    - keep_indices: Indices of the data points to retain.
    """
    scores = scores.reshape(-1, 1)  # KMeans expects 2D array

    # クラスタリング
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(scores)

    keep_indices = []

    # 各クラスタから均等にデータを選ぶ
    for cluster_id in range(num_groups):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue  # 空クラスタに注意
        num_to_select = max(1, len(cluster_indices) * num_to_keep // len(scores))
        selected = np.random.choice(cluster_indices, size=min(num_to_select, len(cluster_indices)), replace=False)
        keep_indices.extend(selected)

    return np.array(keep_indices)