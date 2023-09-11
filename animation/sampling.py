import numpy as np


def sampling(cluster_path, sample_num):
    mu_list = []
    var_list = []
    weight_list = []

    for i in range(len(cluster_path)):
        mu_list.append(cluster_path[i]['mu'])
        var_list.append(cluster_path[i]['var'])
        weight_list.append(cluster_path[i]['weight'])

    mu_list = np.stack(mu_list, axis=-1)
    var_list = np.stack(var_list, axis=-1)
    weight_list = np.stack(weight_list, axis=-1)

    output_len = mu_list.shape[0]

    sample_list = [{} for _ in range(output_len)]
    candidate = np.zeros([output_len, sample_num, 3])

    for i in range(output_len):
        mu, indices = np.unique(mu_list[i], axis=1, return_index=True)
        var = var_list[i][:, indices]
        weight = weight_list[i][indices]
        weight = np.exp(weight) / np.exp(weight).sum()

        for j in range(sample_num):
            index = np.random.choice(np.arange(weight.shape[0]), p=weight)

            mu_select = mu[:, index]
            var_select = var[:, index]

            sample = np.random.normal(loc=mu_select, scale=np.sqrt(var_select))
            candidate[i, j, :] = sample

            if i == 0:
                sample_list[i][str(j)] = {'sample': sample,
                                          'parent': -1
                                          }

            else:
                distance = np.sum((candidate[i-1, :, :] - sample) ** 2, axis=-1)
                sample_list[i][str(j)] = {'sample': sample,
                                          'parent': np.argmin(distance)
                                          }

    sample_traj = np.zeros([sample_num, output_len, 3])

    for i in range(sample_num):
        j = -1
        sample = sample_list[j][str(i)]['sample']
        parent = sample_list[j][str(i)]['parent']
        sample_traj[i, j, :] = sample

        while parent != -1:
            j -= 1
            sample = sample_list[j][str(parent)]['sample']
            parent = sample_list[j][str(parent)]['parent']
            sample_traj[i, j, :] = sample

    return sample_traj
