import numpy as np
from sklearn.cluster import DBSCAN


def MSC(offline_metrics, online_metrics, new_radius=0.1, cluster_method=DBSCAN):
    offline_metrics = np.array(offline_metrics)
    online_metrics = np.array(online_metrics)
    clustering = cluster_method().fit(offline_metrics)
    labels = clustering.labels_

    # use idx as cluster items
    clusters = [[index for index, label in enumerate(labels) if label == unique_label] for unique_label in set(labels)]
    cluster_centers = []
    cluster_sizes = []
    cluster_radius = []

    for cluster in clusters:
        cluster_centers.append(np.mean(offline_metrics[cluster], axis=0))
        cluster_sizes.append(len(cluster))
        dists = np.linalg.norm(offline_metrics[cluster] - cluster_centers[-1], axis=1)
        cluster_radius.append(np.max(dists))

    for idx, metrics_stream in enumerate(online_metrics):
        dists = np.linalg.norm(np.array(cluster_centers) - metrics_stream, axis=1)
        nearest_pattern = np.argmin(dists)
        nearest_dist = dists[nearest_pattern]

        if cluster_radius[nearest_pattern] < nearest_dist:
            # Feedback needed
            clusters.append([idx + len(offline_metrics)])
            cluster_centers.append(metrics_stream)
            cluster_sizes.append(1)
            cluster_radius.append(new_radius)

        else:
            # Combine to existing cluster
            cluster_center, cluster_size = cluster_centers[nearest_pattern], cluster_sizes[nearest_pattern]
            updated_center = (cluster_center * cluster_size + metrics_stream) / (cluster_size + 1)
            subseq_dist = np.linalg.norm(updated_center - metrics_stream)
            updated_graph_dist = max(subseq_dist, cluster_radius[nearest_pattern])
            max_dist = updated_graph_dist

            clusters[nearest_pattern].append(idx + len(offline_metrics))
            cluster_centers[nearest_pattern] = updated_center
            cluster_sizes[nearest_pattern] += 1
            cluster_radius[nearest_pattern] = max_dist

    return clusters
