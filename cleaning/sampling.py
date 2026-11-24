from collections import defaultdict, deque

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def diverse_sample(vectors, groups, risks, budget, seed, cluster_limit):
    """Allocate the global budget round-robin across groups and cluster roles."""
    buckets = defaultdict(list)
    for index, key in enumerate(groups):
        buckets[key].append(index)
    queues = []
    for key in sorted(buckets):
        members = buckets[key]
        members.sort(key=lambda index: (-risks[index], index))
        # Stratified order statistics retain both low- and high-risk members.
        if len(members) > 512:
            positions = np.linspace(0, len(members) - 1, 512, dtype=int)
            members = [members[position] for position in positions]
        x = StandardScaler().fit_transform(np.asarray([vectors[index] for index in members]))
        clusters = min(cluster_limit, max(1, len(members) // 12), len(np.unique(x, axis=0)))
        if clusters == 1:
            labels, centers = np.zeros(len(x), dtype=int), [x.mean(axis=0)]
        else:
            km = KMeans(n_clusters=clusters, random_state=seed, n_init=3).fit(x)
            labels, centers = km.labels_, km.cluster_centers_
        roles = [[], [], []]
        for cluster, center in enumerate(centers):
            indices = np.flatnonzero(labels == cluster)
            if len(indices) == 0:
                continue
            distances = np.linalg.norm(x[indices] - center, axis=1)
            roles[0].append(members[indices[int(distances.argmin())]])
            roles[1].append(min((members[index] for index in indices),
                                key=lambda index: (abs(risks[index] - 0.5), index)))
            roles[2].append(members[indices[int(distances.argmax())]])
        queues.append(deque(dict.fromkeys([index for role in roles for index in role] + members)))
    selected = []
    queues = deque(queues)
    while queues and len(selected) < budget:
        queue = queues.popleft()
        selected.append(queue.popleft())
        if queue:
            queues.append(queue)
    return selected
