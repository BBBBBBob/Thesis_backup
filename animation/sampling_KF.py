import numpy as np


def sampling_kf(prediction, covariance, sample_num):

    output_len = prediction.shape[0]

    sample_list = [{} for _ in range(output_len)]
    candidate = np.zeros([output_len, sample_num, 3])

    for i in range(output_len):
        for j in range(sample_num):
            mu = prediction[i, :]
            var = covariance[i, :, :]
            sample = np.random.multivariate_normal(mean=mu, cov=var)[:3]
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
