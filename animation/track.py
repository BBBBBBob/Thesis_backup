import numpy as np
from filterpy.kalman import KalmanFilter
class Track:
    count = 0

    def __init__(self, dt, detection_pos, detection_box, lifetime=5):
        self.id = Track.count
        Track.count += 1

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                              [0, 1, 0, 0, 0, dt, 0, 0],
                              [0, 0, 1, 0, 0, 0, dt, 0],
                              [0, 0, 0, 1, 0, 0, 0, dt],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.P = np.diag([1, 1, 1, 1, 100, 100, 100, 100])
        self.kf.Q *= 1
        self.kf.R *= 1
        self.kf.x[:4] = detection_pos

        self.prediction = detection_pos
        self.detection_pos = detection_pos
        self.detection_box = detection_box

        self.lifetime = lifetime
        self.not_detected_count = 0

    def predict(self):
        self.kf.predict()
        self.prediction = self.kf.x

    def update(self, detection_pos, detection_box):
        self.kf.update(detection_pos)
        self.detection_pos = detection_pos
        self.detection_box = detection_box
        self.not_detected_count = 0

    def is_dead(self):
        return self.not_detected_count > self.lifetime
