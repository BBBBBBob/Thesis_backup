import numpy as np
from filterpy.kalman import KalmanFilter

class predictor_kf:
    def __init__(self, dt):
        self.kf = KalmanFilter(dim_x=9, dim_z=3)
        self.kf.F = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, dt, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, dt, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, dt, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, dt, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, dt],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0]])

        self.kf.P = np.diag([1, 1, 1, 100, 100, 100, 100, 100, 100])
        self.kf.Q *= 0.1
        self.kf.R *= 0.1

    def initialize(self, state):
        self.kf.x[:3] = state

    def predict(self):
        self.kf.predict()

    def update(self, state=None):
        self.kf.update(state)
