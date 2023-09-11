import torch
def distort(points_img, optical_center, focal_length, batch_size, K):  ## Nx2xL
    optical_center_b = optical_center.repeat(batch_size, 1, points_img.shape[2])
    points_centered = points_img - optical_center_b
    r = torch.norm(points_centered / focal_length, p=2, dim=1, keepdim=True) + 1e-5
    theta = torch.arctan(r)
    theta2 = theta ** 2
    theta4 = theta2 ** 2
    theta6 = theta4 * theta2
    theta8 = theta4 ** 2

    theta_d = theta * (1.0 + K[0] * theta2 + K[1] * theta4 + K[2] * theta6 + K[3] * theta8)
    scale = theta_d / r
    points_img_dist = scale * points_centered + optical_center

    return points_img_dist


def undistort(points_img, optical_center, focal_length, batch_size, K):  ## Nx2xL L is predict length
    optical_center_b = optical_center.repeat(batch_size, 1, points_img.shape[2])
    points_centered = points_img - optical_center_b
    theta_d = torch.norm(points_centered / focal_length, p=2, dim=1, keepdim=True) + 1e-5
    theta = theta_d

    for _ in range(20):
        theta2 = theta ** 2
        theta4 = theta2 ** 2
        theta6 = theta4 * theta2
        theta8 = theta4 ** 2

        theta = theta_d / (1.0 + K[0] * theta2 + K[1] * theta4 + K[2] * theta6 + K[3] * theta8)

    scale = torch.tan(theta) / theta_d
    points_img_rect = scale * points_centered + optical_center_b

    return points_img_rect


def projection(pred_pos_w, extrinsic, intrinsic, normalize, optical_center, focal_length, batch_size, K):  ## N x 4 x L
    extrinsic_b = extrinsic.repeat(batch_size, 1, 1)
    intrinsic_b = intrinsic.repeat(batch_size, 1, 1)
    normalize_b = normalize.repeat(batch_size, 1, 1)
    pred_pos_cam = torch.bmm(extrinsic_b, pred_pos_w)
    pred_pos_img = torch.bmm(intrinsic_b, pred_pos_cam)
    pred_pos_img = pred_pos_img[:, :2, :] / pred_pos_img[:, 2:, :]  ## N x 2 x L
    pred_pos_img_dist = distort(pred_pos_img, optical_center, focal_length, batch_size, K)

    return pred_pos_img_dist.permute(0, 2, 1)


def projection_norm(pred_pos_w, extrinsic, intrinsic, normalize, optical_center, focal_length,
                    batch_size, K):  ## N x 4 x L
    extrinsic_b = extrinsic.repeat(batch_size, 1, 1)
    intrinsic_b = intrinsic.repeat(batch_size, 1, 1)
    normalize_b = normalize.repeat(batch_size, 1, 1)
    pred_pos_cam = torch.bmm(extrinsic_b, pred_pos_w)
    pred_pos_img = torch.bmm(intrinsic_b, pred_pos_cam)
    pred_pos_img = pred_pos_img[:, :2, :] / pred_pos_img[:, 2:, :]  ## N x 2 x L
    pred_pos_img_dist = distort(pred_pos_img, optical_center, focal_length, batch_size, K)
    pred_pos_img_dist_norm = pred_pos_img_dist / normalize_b * 2 - 1

    return pred_pos_img_dist_norm.permute(0, 2, 1)


def projection_direct(pred_pos_w, extrinsic, intrinsic, normalize, optical_center, focal_length,
                      batch_size):  ## N x 4 x L
    extrinsic_b = extrinsic.repeat(batch_size, 1, 1)
    intrinsic_b = intrinsic.repeat(batch_size, 1, 1)
    pred_pos_cam = torch.bmm(extrinsic_b, pred_pos_w)
    pred_pos_img = torch.bmm(intrinsic_b, pred_pos_cam)
    pred_pos_img = pred_pos_img[:, :2, :] / pred_pos_img[:, 2:, :]  ## N x 2 x L

    return pred_pos_img.permute(0, 2, 1)


def calculate_angle(pred_pos_world, gt_pos_world, left_cam_pos, right_cam_pos):  ## NxLx3
    left_cam_pos_b = left_cam_pos.repeat(pred_pos_world.shape[0], pred_pos_world.shape[1], 1)
    right_cam_pos_b = right_cam_pos.repeat(pred_pos_world.shape[0], pred_pos_world.shape[1], 1)
    pred_pos_world_left = pred_pos_world - left_cam_pos_b
    gt_pos_world_left = gt_pos_world - left_cam_pos_b
    pred_pos_world_right = pred_pos_world - right_cam_pos_b
    gt_pos_world_right = gt_pos_world - right_cam_pos_b

    inner_left = torch.nn.functional.cosine_similarity(pred_pos_world_left, gt_pos_world_left, dim=2)
    inner_right = torch.nn.functional.cosine_similarity(pred_pos_world_right, gt_pos_world_right, dim=2)

    angle_left = torch.acos(torch.clamp(inner_left, -1 + 1e-6, 1 - 1e-6))
    angle_right = torch.acos(torch.clamp(inner_right, -1 + 1e-6, 1 - 1e-6))

    return angle_left, angle_right
