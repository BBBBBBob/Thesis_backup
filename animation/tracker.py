import numpy as np
from scipy.optimize import linear_sum_assignment
from track import Track

class KFDistTracker:
    def __init__(self, dt):
        self.dt = dt
        self.tracks = []
        self.id = 0

    def associate(self, detections):
        detection_pos = detections['pos']
        detection_box = detections['box']

        for track in self.tracks:
            track.predict()

        if len(detections) > 0:
            track_num = len(self.tracks)
            detection_num = len(detection_pos)
            cost_matrix = np.zeros([track_num, detection_num])
            for i, track in enumerate(self.tracks):
                for j, detection in enumerate(detection_pos):
                    dist = np.sqrt(np.sum((track.prediction[:2] - detection[:2])**2)) \
                           + np.sqrt(np.sum((track.prediction[2:4] - detection[2:4])**2))
                    cost_matrix[i, j] = dist

            row_index, col_index = linear_sum_assignment(cost_matrix)

            unmatch_tracks = set(range(track_num)) - set(row_index)
            unmatch_detections = set(range(detection_num)) - set(col_index)

            for i, j in zip(row_index, col_index):
                self.tracks[i].update(detection_pos[j], detection_box[j])
            for i in unmatch_tracks:
                self.tracks[i].not_detected_count += 1
            for i in unmatch_detections:
                add_track = Track(self.dt, detection_pos[i], detection_box[i])
                self.tracks.append(add_track)

        self.tracks = [track for track in self.tracks if not track.is_dead()]

        return self.tracks
