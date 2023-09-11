import numpy as np
from sklearn.cluster import DBSCAN, OPTICS


## list contains 16 dictionaries
## First layer of dict contains the cluster
## Second layer of dict contains the mu, var, pi, parent of cluster

def clustering(mu, var, pi, eps, min_samples):  ## mu LxMx3  var LxMx3  pi LxM
    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='auto')
    group_info = [{} for _ in range(mu.shape[0])]
    for i in range(mu.shape[0]):
        if i == 0:
            mask = pi[i, :] > (1 / mu.shape[1])
            groupings = cluster.fit_predict(mu[i][mask])
            for idx in range(max(groupings) + 1):
                center = np.sum(mu[i][mask][groupings == idx] * np.expand_dims(pi[i][mask][groupings == idx], axis=1),
                                axis=0)
                # radius = np.sum(((mu[i][mask][groupings == idx] - center) ** 2 + var[i][mask][groupings == idx])
                #                 * np.expand_dims(pi[i][mask][groupings == idx], axis=1), axis=0)
                radius = np.sum((mu[i][mask][groupings == idx] ** 2 + var[i][mask][groupings == idx])
                                * np.expand_dims(pi[i][mask][groupings == idx], axis=1), axis=0) - center ** 2
                weight = np.sum(pi[i][mask][groupings == idx])

                group_info[i][str(idx)] = {
                    'center': center,
                    'radius': radius,
                    'weight': weight

                }
            noise_index = np.where(groupings == -1)

            for j in range(len(noise_index[0])):
                group_info[i][str(max(groupings) + j + 1)] = {
                    'center': mu[i][mask][noise_index[0][j]],
                    'radius': var[i][mask][noise_index[0][j]],
                    'weight': pi[i][mask][noise_index[0][j]]
                }
        else:
            mask = pi[i, :] > (1 / mu.shape[1])
            groupings = cluster.fit_predict(mu[i][mask])
            for idx in range(max(groupings) + 1):
                center = np.sum(mu[i][mask][groupings == idx] * np.expand_dims(pi[i][mask][groupings == idx], axis=1),
                                axis=0)
                # radius = np.sum(((mu[i][mask][groupings == idx] - center) ** 2 + var[i][mask][groupings == idx])
                #                 * np.expand_dims(pi[i][mask][groupings == idx], axis=1), axis=0)
                radius = np.sum((mu[i][mask][groupings == idx] ** 2 + var[i][mask][groupings == idx])
                                * np.expand_dims(pi[i][mask][groupings == idx], axis=1), axis=0) - center ** 2
                weight = np.sum(pi[i][mask][groupings == idx])
                group_info[i][str(idx)] = {
                    'center': center,
                    'radius': radius,
                    'weight': weight
                }
            noise_index = np.where(groupings == -1)

            for j in range(len(noise_index[0])):
                group_info[i][str(max(groupings) + j + 1)] = {
                    'center': mu[i][mask][noise_index[0][j]],
                    'radius': var[i][mask][noise_index[0][j]],
                    'weight': pi[i][mask][noise_index[0][j]]
                }
            ## matching last centroid
            paired_idx = []
            for current_idx in range(len(group_info[i])):
                min_dist = float('inf')
                id = 0
                current_center = group_info[i][str(current_idx)]['center']
                for last_idx in range(len(group_info[i - 1])):
                    last_center = group_info[i - 1][str(last_idx)]['center']
                    dist = np.sqrt(np.sum((current_center - last_center) ** 2))
                    if dist < min_dist:
                        id = last_idx
                        min_dist = dist
                group_info[i][str(current_idx)]['parent'] = id
                paired_idx.append(id)

            ## removing unpaired parent
            unpaired_idx = set(range(len(group_info[i - 1]))) - set(paired_idx)
            for idx in unpaired_idx:
                group_info[i - 1].pop(str(idx))

    ## From the last to the first step:
    cluster_path = [{'mu': np.zeros([mu.shape[0], 3]),
                     'var': np.zeros([mu.shape[0], 3]),
                     'weight': np.zeros([mu.shape[0]])} for _ in range(len(group_info[-1]))]

    for idx in range(len(group_info[-1])):
        parent_idx = idx
        for i in reversed(range(len(group_info))):
            cluster_path[idx]['mu'][i, :] = group_info[i][str(parent_idx)]['center']
            cluster_path[idx]['var'][i, :] = group_info[i][str(parent_idx)]['radius']
            cluster_path[idx]['weight'][i] = group_info[i][str(parent_idx)]['weight']
            if i > 0:
                parent_idx = group_info[i][str(parent_idx)]['parent']

    return cluster_path
