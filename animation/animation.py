import pandas as pd
from train.net_ori import GRU_ATT
import glob
import os
import re
import skimage
from utils import *
from parameters import *
from kf_predictor import predictor_kf
from cluster import clustering
from sampling import sampling
from sampling_KF import sampling_kf
from tracker import KFDistTracker
import time

np.random.seed(11)
model_path = '../weight/best_model_evolve_enc.pt'
net = GRU_ATT(hidden_size=hidden_size, input_size=input_size,
              output_size=output_size, output_len=output_len,
              num_layers=num_layers, mode=8, temp=1, dropout=0).to(device=device)

net_state_dict = torch.load(model_path)
net.load_state_dict(net_state_dict, strict=True)

left_frame_path = r'/media/jiacheng/TOSHIBA EXT/render/track_12_night_left_frames/'
right_frame_path = r'/media/jiacheng/TOSHIBA EXT/render/track_12_night_left_frames/'

left_frames = sorted(glob.glob(left_frame_path + '*.png'),
                     key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))[100:200]
right_frames = sorted(glob.glob(right_frame_path + '*.png'),
                      key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))[100:200]

begin = int(re.findall(r'\d+', os.path.basename(left_frames[0]))[0])
end = int(re.findall(r'\d+', os.path.basename(left_frames[-1]))[0])

# gt_left_csv_path = r'/media/jiacheng/TOSHIBA EXT//render/new_test_pix_align_1/traj_12.csv'
# gt_right_csv_path = r'/media/jiacheng/TOSHIBA EXT//render/new_test_pix_align_2/traj_12.csv'
#
# gt_left = np.asarray(pd.read_csv(gt_left_csv_path, index_col=0))[begin + 2:, :]
# gt_right = np.asarray(pd.read_csv(gt_right_csv_path, index_col=0))[begin + 2:, :]
#
# gt_CoG_left = (gt_left[:, 6:8] + 1) / 2 * np.array([848, 800])
# gt_CoG_right = (gt_right[:, 6:8] + 1) / 2 * np.array([848, 800])

assert len(left_frames) == len(right_frames)

left_gray_frame_median, right_gray_frame_median = get_median_frame(left_frames, right_frames)

history_pos_pix_np = np.zeros([8, 4])

count = 0
step_time = np.tile(np.arange(1, 17) * 0.02, (3, 1)).transpose()

prediction_sum_left = 0
prediction_det_sum_left = 0
baseline_sum_left = 0
prediction_sum_right = 0
prediction_det_sum_right = 0
baseline_sum_right = 0

Tracker = KFDistTracker(dt)

frame_len = len(left_frames)

for i in range(len(left_frames)):
    start_time = time.time()

    left_frame = cv2.imread(left_frames[i])
    right_frame = cv2.imread(right_frames[i])

    left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    left_gray_frame = skimage.exposure.match_histograms(left_gray_frame, left_gray_frame_median)
    right_gray_frame = skimage.exposure.match_histograms(right_gray_frame, right_gray_frame_median)

    left_gray_frame = left_gray_frame.astype(np.uint8)
    right_gray_frame = right_gray_frame.astype(np.uint8)

    left_dframe = cv2.absdiff(left_gray_frame, left_gray_frame_median)
    right_dframe = cv2.absdiff(right_gray_frame, right_gray_frame_median)

    left_blur_frame = cv2.GaussianBlur(left_dframe, (5, 5), 0)
    right_blur_frame = cv2.GaussianBlur(right_dframe, (5, 5), 0)

    _, left_threshold_frame = cv2.threshold(left_blur_frame, 50, 255, cv2.THRESH_BINARY)
    _, right_threshold_frame = cv2.threshold(right_blur_frame, 50, 255, cv2.THRESH_BINARY)

    left_contours, _ = cv2.findContours(left_threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_contours, _ = cv2.findContours(right_threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_frame_detections = []
    right_frame_detections = []
    left_frame_box = []
    right_frame_box = []

    for j in left_contours:
        x, y, width, height = cv2.boundingRect(j)
        pos = np.array([[x + width / 2, y + height / 2]])
        left_frame_detections.append(pos)
        left_frame_box.append(np.array([[width, height]]))

    for j in right_contours:
        x, y, width, height = cv2.boundingRect(j)
        pos = np.array([[x + width / 2, y + height / 2]])
        right_frame_detections.append(pos)
        right_frame_box.append(np.array([[width, height]]))

    match_list = matching(left_frame_detections, right_frame_detections, left_frame_box, right_frame_box)
    track_list = Tracker.associate(match_list)

    for track in track_list:
        pos = track.detection_pos
        box = track.detection_box
        pos_left = pos[:2]
        pos_right = pos[2:]
        box_left = box[:2]
        box_right = box[2:]

        cv2.rectangle(left_frame, (int(pos_left[0] - box_left[0] / 2), int(pos_left[1] - box_left[1] / 2)),
                      (int(pos_left[0] + box_left[0] / 2), int(pos_left[1] + box_left[1] / 2)), (255, 130, 0), 2)
        cv2.rectangle(right_frame, (int(pos_right[0] - box_right[0] / 2), int(pos_right[1] - box_right[1] / 2)),
                      (int(pos_right[0] + box_right[0] / 2), int(pos_right[1] + box_right[1] / 2)), (255, 130, 0), 2)

        pos_norm = pos.T / np.array([848, 800, 848, 800]) * 2 - 1

        if count < 8:
            history_pos_pix_np[count:count + 1, :] = pos_norm
        else:
            history_pos_pix_np = np.concatenate((history_pos_pix_np[1:, :], pos_norm), axis=0)

    if count >= 7:
        history_pos_pix = torch.tensor(history_pos_pix_np,
                                       device=device, dtype=torch.float32).unsqueeze(0)

        kf_prediction = []
        kf_covariance = []

        kf_predictor = predictor_kf(dt)

        # for j in range(input_len):
        #     if j == 0:
        #         kf_predictor.initialize(gt_left[i+j-input_len+1:i+j-input_len+2, :3].T)
        #         kf_predictor.predict()
        #     else:
        #         kf_predictor.update(gt_left[i+j-input_len+1:i+j-input_len+2, :3].T)
        #         kf_predictor.predict()
        #
        # for _ in range(output_len):
        #     kf_prediction.append(kf_predictor.kf.x)
        #     kf_covariance.append(kf_predictor.kf.P)
        #     kf_predictor.predict()
        #
        # kf_prediction = np.stack(kf_prediction, axis=0).squeeze()
        # kf_covariance = np.stack(kf_covariance, axis=0)
        #
        # world_pos_future_baseline_kf = kf_prediction[:, :3]
        #
        # baseline_kf_w = np.concatenate(
        #     (world_pos_future_baseline_kf.transpose(), np.ones([1, world_pos_future_baseline_kf.shape[0]])), axis=0)
        #
        # baseline_kf_cam_1 = extrinsic_1 @ baseline_kf_w
        # baseline_kf_img_1 = intrinsic @ baseline_kf_cam_1
        # baseline_kf_img_1 = baseline_kf_img_1[:2, :] / baseline_kf_img_1[2, :]
        # baseline_kf_img_dist_1 = np.round(distort(baseline_kf_img_1, optical_center, focal_length, K)).T
        #
        # baseline_kf_cam_2 = extrinsic_2 @ baseline_kf_w
        # baseline_kf_img_2 = intrinsic @ baseline_kf_cam_2
        # baseline_kf_img_2 = baseline_kf_img_2[:2, :] / baseline_kf_img_2[2, :]
        # baseline_kf_img_dist_2 = np.round(distort(baseline_kf_img_2, optical_center, focal_length, K)).T
        #
        # cv2.polylines(left_frame, np.int32([baseline_kf_img_dist_1]), isClosed=False, color=[0, 255, 0], thickness=2)
        # cv2.polylines(right_frame, np.int32([baseline_kf_img_dist_2]), isClosed=False, color=[0, 255, 0], thickness=2)

        last_pos_pix = history_pos_pix[:, -1:, :]

        start_time = time.time()
        pred_last_pos_world_mean, pred_last_pos_world_var, pi, mu, var = net(history_pos_pix, device)
        pi = torch.nn.functional.softmax(pi, dim=-1)

        pred_last_pos_world_mean = torch.tile(pred_last_pos_world_mean.squeeze(2), dims=(1, 1, 8, 1))
        pred_last_pos_world_var = torch.tile(pred_last_pos_world_var.squeeze(2), dims=(1, 1, 8, 1))

        pred_pos_world_mean = pred_last_pos_world_mean + torch.cumsum(mu, dim=1)
        pred_pos_world_var = pred_last_pos_world_var + torch.cumsum(var, dim=1)

        pi = pi.detach().cpu().numpy().squeeze()
        pred_pos_world_mean = pred_pos_world_mean.detach().cpu().numpy().squeeze()
        pred_pos_world_var = pred_pos_world_var.detach().cpu().numpy().squeeze()

        # gt_pos = gt_left[i + 1:i + 17, :3]

        cluster_path = clustering(pred_pos_world_mean, pred_pos_world_var, pi, eps=0.2, min_samples=2)

        for m in range(len(cluster_path)):
            mu = cluster_path[m]['mu']
            pred_w = np.concatenate(
                    (mu.transpose(), np.ones([1, mu.shape[0]])),
                    axis=0)

            pred_cam_1 = extrinsic_1 @ pred_w
            pred_img_1 = intrinsic @ pred_cam_1
            pred_img_1 = pred_img_1[:2, :] / pred_img_1[2, :]
            pred_img_dist_1 = np.round(distort(pred_img_1, optical_center, focal_length, K)).T

            pred_cam_2 = extrinsic_2 @ pred_w
            pred_img_2 = intrinsic @ pred_cam_2
            pred_img_2 = pred_img_2[:2, :] / pred_img_2[2, :]
            pred_img_dist_2 = np.round(distort(pred_img_2, optical_center, focal_length, K)).T

            cv2.polylines(left_frame, np.int32([pred_img_dist_1]), isClosed=False, color=[0, 0, 255], thickness=2)
            cv2.polylines(right_frame, np.int32([pred_img_dist_2]), isClosed=False, color=[0, 0, 255], thickness=2)

        # sampled_traj = sampling(cluster_path, 50)
        # sampled_traj_kf = sampling_kf(kf_prediction, kf_covariance, 50)
        #
        # gt_img_dist_1 = np.round(gt_CoG_left[i + 1:i + 17, :])
        # gt_img_dist_2 = np.round(gt_CoG_right[i + 1:i + 17, :])

        # for j in range(sampled_traj.shape[0]):
        #     traj = sampled_traj[j, :, :]
        #     pred_w = np.concatenate(
        #             (traj.transpose(), np.ones([1, traj.shape[0]])),
        #             axis=0)
        #
        #     pred_cam_1 = extrinsic_1 @ pred_w
        #     pred_img_1 = intrinsic @ pred_cam_1
        #     pred_img_1 = pred_img_1[:2, :] / pred_img_1[2, :]
        #     pred_img_dist_1 = np.round(distort(pred_img_1, optical_center, focal_length, K)).T
        #
        #     pred_cam_2 = extrinsic_2 @ pred_w
        #     pred_img_2 = intrinsic @ pred_cam_2
        #     pred_img_2 = pred_img_2[:2, :] / pred_img_2[2, :]
        #     pred_img_dist_2 = np.round(distort(pred_img_2, optical_center, focal_length, K)).T
        #
        #     cv2.polylines(left_frame, np.int32([pred_img_dist_1]), isClosed=False, color=[0, 0, 255], thickness=2)
        #     cv2.polylines(right_frame, np.int32([pred_img_dist_2]), isClosed=False, color=[0, 0, 255], thickness=2)
        #
        # for j in range(sampled_traj_kf.shape[0]):
        #     traj = sampled_traj_kf[j, :, :]
        #     pred_w = np.concatenate(
        #         (traj.transpose(), np.ones([1, traj.shape[0]])),
        #         axis=0)
        #
        #     pred_cam_1 = extrinsic_1 @ pred_w
        #     pred_img_1 = intrinsic @ pred_cam_1
        #     pred_img_1 = pred_img_1[:2, :] / pred_img_1[2, :]
        #     pred_img_dist_1 = np.round(distort(pred_img_1, optical_center, focal_length, K)).T
        #
        #     pred_cam_2 = extrinsic_2 @ pred_w
        #     pred_img_2 = intrinsic @ pred_cam_2
        #     pred_img_2 = pred_img_2[:2, :] / pred_img_2[2, :]
        #     pred_img_dist_2 = np.round(distort(pred_img_2, optical_center, focal_length, K)).T
        #
        #     cv2.polylines(left_frame, np.int32([pred_img_dist_1]), isClosed=False, color=[0, 165, 255], thickness=2)
        #     cv2.polylines(right_frame, np.int32([pred_img_dist_2]), isClosed=False, color=[0, 165, 255], thickness=2)

        # cv2.polylines(left_frame, np.int32([gt_img_dist_1]), isClosed=False, color=[255, 0, 0], thickness=2)
        # cv2.polylines(right_frame, np.int32([gt_img_dist_2]), isClosed=False, color=[255, 0, 0], thickness=2)
        cv2.imshow('Animation', np.concatenate((left_frame, right_frame), axis=1))
        key = cv2.waitKey(20)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    count += 1

cv2.destroyAllWindows()

