import cv2

from train.net import GRU_ATT
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

left_frame_path = r'/media/jiacheng/TOSHIBA EXT/jiacheng_bag/2023-08-23-17-55-28/img/left/'
right_frame_path = r'/media/jiacheng/TOSHIBA EXT/jiacheng_bag/2023-08-23-17-55-28/img/right/'

left_frames = sorted(glob.glob(left_frame_path + '*.png'),
                     key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))[-800:]
right_frames = sorted(glob.glob(right_frame_path + '*.png'),
                      key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))[-800:]

assert len(left_frames) == len(right_frames)

left_gray_frame_median, right_gray_frame_median = get_median_frame(left_frames, right_frames)

history_pos_pix_np = np.zeros([8, 4])

count = 0

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

    # heatmapshow_left = None
    # heatmapshow_left = cv2.normalize(left_dframe, heatmapshow_left, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmapshow_left = cv2.applyColorMap(heatmapshow_left, cv2.COLORMAP_JET)
    #
    # heatmapshow_right = None
    # heatmapshow_right = cv2.normalize(left_dframe, heatmapshow_right, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmapshow_right = cv2.applyColorMap(heatmapshow_right, cv2.COLORMAP_JET)

    left_blur_frame = cv2.GaussianBlur(left_dframe, (9, 9), 0)
    right_blur_frame = cv2.GaussianBlur(right_dframe, (9, 9), 0)

    kernel_ero = np.ones((5, 5), np.uint8)
    kernel_dia = np.ones((9, 9), np.uint8)

    left_blur_frame = cv2.erode(left_blur_frame, kernel_ero, iterations=1)
    right_blur_frame = cv2.erode(right_blur_frame, kernel_ero, iterations=1)

    left_blur_frame = cv2.dilate(left_blur_frame, kernel_dia, iterations=1)
    right_blur_frame = cv2.dilate(right_blur_frame, kernel_dia, iterations=1)

    _, left_threshold_frame = cv2.threshold(left_blur_frame, 7, 255, cv2.THRESH_BINARY)
    _, right_threshold_frame = cv2.threshold(right_blur_frame, 7, 255, cv2.THRESH_BINARY)

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
    # #
    # #     pos_norm = pos.T / np.array([848, 800, 848, 800]) * 2 - 1
    # all_box = []
    # for track in track_list:
    #     pos = track.detection_pos
    #     box = track.detection_box
    #     pos_left = pos[:2]
    #     pos_right = pos[2:]
    #     box_left = box[:2]
    #     box_right = box[2:]
    #     all_box.append(box_left[0] * box_left[1] + box_right[0] * box_right[1])
    #     # cv2.rectangle(left_frame, (int(pos_left[0] - box_left[0] / 2), int(pos_left[1] - box_left[1] / 2)),
    #     #               (int(pos_left[0] + box_left[0] / 2), int(pos_left[1] + box_left[1] / 2)), (255, 130, 0), 2)
    #     # cv2.rectangle(right_frame, (int(pos_right[0] - box_right[0] / 2), int(pos_right[1] - box_right[1] / 2)),
    #     #               (int(pos_right[0] + box_right[0] / 2), int(pos_right[1] + box_right[1] / 2)), (255, 130, 0), 2)
    #     # cv2.putText(left_frame, str(track.id), (int(pos_left[0]), int(pos_left[1])), cv2.FONT_HERSHEY_SIMPLEX,
    #     #            1, (255, 255, 255), 1, 2)
    # index = all_box.index(max(all_box))
    # track = track_list[index]
    # pos = track.detection_pos
    # box = track.detection_box
    # pos_left = pos[:2]
    # pos_right = pos[2:]
    # box_left = box[:2]
    # box_right = box[2:]
    # cv2.rectangle(left_frame, (int(pos_left[0] - box_left[0] / 2), int(pos_left[1] - box_left[1] / 2)),
    #               (int(pos_left[0] + box_left[0] / 2), int(pos_left[1] + box_left[1] / 2)), (255, 130, 0), 2)
    # cv2.rectangle(right_frame, (int(pos_right[0] - box_right[0] / 2), int(pos_right[1] - box_right[1] / 2)),
    #               (int(pos_right[0] + box_right[0] / 2), int(pos_right[1] + box_right[1] / 2)), (255, 130, 0), 2)
    #
    # pos_norm = pos.T / np.array([848, 800, 848, 800]) * 2 - 1
    #
    cv2.imshow('Animation', np.concatenate((left_frame, right_frame), axis=1))
    # cv2.imshow('test', heatmapshow_left)
    key = cv2.waitKey(20)
    if key == 27:
        break
    count += 1

cv2.destroyAllWindows()
