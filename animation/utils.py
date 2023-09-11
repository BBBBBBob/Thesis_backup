import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2


def matching(left_detection, right_detection, left_box, right_box):
    cost_matrix = np.zeros([len(left_detection), len(right_detection)])
    for i, left_pos in enumerate(left_detection):
        for j, right_pos in enumerate(right_detection):
            dist = np.sqrt(np.sum((left_pos - right_pos) ** 2))
            cost_matrix[i, j] = dist

    row_index, col_index = linear_sum_assignment(cost_matrix)

    match_pos_list = []
    match_box_list = []
    for i, j in zip(row_index, col_index):
        match_pos = np.concatenate((left_detection[i], right_detection[j]), axis=1).T
        match_box = np.concatenate((left_box[i], right_box[j]), axis=1).T
        match_pos_list.append(match_pos)
        match_box_list.append(match_box)

    match_list = {'pos': match_pos_list, 'box': match_box_list}
    return match_list


def triangulation(pos_img_left, pos_img_right, left_cam_pos, intrinsic, baseline):
    intrinsic_inv = np.linalg.inv(intrinsic)
    pos_img_left_inv = intrinsic_inv @ pos_img_left
    pos_img_right_inv = intrinsic_inv @ pos_img_right
    A = np.concatenate((pos_img_left_inv, -pos_img_right_inv), axis=1)
    A_pinv = np.linalg.pinv(A.T @ A) @ A.T
    lambda_var = A_pinv @ baseline
    pos_world = np.linalg.inv(np.diag([-1, -1, 1])) @ (
            lambda_var[0] * (intrinsic_inv @ pos_img_left).squeeze() - left_cam_pos)

    return pos_world


def undistort(points_img, optical_center, focal_length, K):
    points_img_rect = np.zeros(points_img.shape)
    for m in range(points_img.shape[1]):
        point_img = points_img[:, m]
        point_centered = point_img - optical_center
        theta_d = np.linalg.norm(point_centered / focal_length, ord=2) + 1e-5
        theta = theta_d

        for _ in range(20):
            theta2 = theta ** 2
            theta4 = theta2 ** 2
            theta6 = theta4 * theta2
            theta8 = theta4 ** 2

            theta = theta_d / (1.0 + K[0] * theta2 + K[1] * theta4 + K[2] * theta6 + K[3] * theta8)

        scale = np.tan(theta) / theta_d

        point_img_rect = scale * point_centered + optical_center
        points_img_rect[:, m] = point_img_rect

    return points_img_rect


def distort(points_img, optical_center, focal_length, K):
    points_img_dist = np.zeros(points_img.shape)
    for m in range(points_img.shape[1]):
        point_img = points_img[:, m]
        point_centered = point_img - optical_center
        r = np.linalg.norm(point_centered / focal_length, ord=2) + 1e-5
        theta = np.arctan(r)

        theta2 = theta ** 2
        theta4 = theta2 ** 2
        theta6 = theta4 * theta2
        theta8 = theta4 ** 2

        theta_d = theta * (1.0 + K[0] * theta2 + K[1] * theta4 + K[2] * theta6 + K[3] * theta8)

        scale = theta_d / r

        point_img_dist = scale * point_centered + optical_center
        points_img_dist[:, m] = point_img_dist

    return points_img_dist


def distort_th(points_img, optical_center_th, focal_length_th, K):  ## Nx2xL
    optical_center_b = optical_center_th.repeat(points_img.shape[0], 1, points_img.shape[2])
    points_centered = points_img - optical_center_b
    r = torch.norm(points_centered / focal_length_th, p=2, dim=1, keepdim=True) + 1e-5
    theta = torch.arctan(r)
    theta2 = theta ** 2
    theta4 = theta2 ** 2
    theta6 = theta4 * theta2
    theta8 = theta4 ** 2

    theta_d = theta * (1.0 + K[0] * theta2 + K[1] * theta4 + K[2] * theta6 + K[3] * theta8)
    scale = theta_d / r
    points_img_dist = scale * points_centered + optical_center_th

    return points_img_dist


def projection_norm_th(pred_pos_w, extrinsic_th, intrinsic_th, normalize_th, optical_center_th,
                       focal_length_th):  ## N x 4 x L
    extrinsic_b = extrinsic_th.repeat(pred_pos_w.shape[0], 1, 1)
    intrinsic_b = intrinsic_th.repeat(pred_pos_w.shape[0], 1, 1)
    normalize_b = normalize_th.repeat(pred_pos_w.shape[0], 1, 1)
    pred_pos_cam = torch.bmm(extrinsic_b, pred_pos_w)
    pred_pos_img = torch.bmm(intrinsic_b, pred_pos_cam)
    pred_pos_img = pred_pos_img[:, :2, :] / pred_pos_img[:, 2:, :]  ## N x 2 x L
    pred_pos_img_dist = distort_th(pred_pos_img, optical_center_th, focal_length_th)
    pred_pos_img_dist_norm = pred_pos_img_dist / normalize_b * 2 - 1
    # pred_pos_img_rect_norm = pred_pos_img_rect / normalize_b * 2 - 1

    return pred_pos_img_dist_norm.permute(0, 2, 1)


def get_median_frame(left_frames, right_frames):
    left_frame_list = []
    right_frame_list = []

    for i in range(60):
        left_frame = cv2.imread(left_frames[i])
        right_frame = cv2.imread(right_frames[i])

        left_frame_list.append(left_frame)
        right_frame_list.append(right_frame)

    left_frame_median = np.median(left_frame_list, axis=0).astype(dtype=np.uint8)
    left_gray_frame_median = cv2.cvtColor(left_frame_median, cv2.COLOR_BGR2GRAY)

    right_frame_median = np.median(right_frame_list, axis=0).astype(dtype=np.uint8)
    right_gray_frame_median = cv2.cvtColor(right_frame_median, cv2.COLOR_BGR2GRAY)

    return left_gray_frame_median, right_gray_frame_median
