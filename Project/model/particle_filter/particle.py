import numpy as np


class Particle:
    def __init__(self, x, y, s, a, dir_deg, o_deg, weight: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.s = float(s)
        self.a = float(a)
        self.dir = float(dir_deg)
        self.o = float(o_deg)
        self.weight = float(weight)

    def moveParticle(self, speed, accel, dir_deg, dt: float = 0.1, pos_noise_std: float = 0.05, model=None, model_input=None):

        if model is not None:
            pred = model.predict_next_state(model_input)

            x_pred, y_pred, s_pred, a_pred, dir_pred, o_pred = pred

            self.x = float(x_pred) + np.random.normal(0.0, pos_noise_std)
            self.y = float(y_pred) + np.random.normal(0.0, pos_noise_std)

            self.s = float(s_pred)
            self.a = float(a_pred)
            self.dir = float(dir_pred)
            self.o = float(o_pred)

        else:
            theta = np.deg2rad(dir_deg)
            disp = (speed * dt) + (0.5 * accel * dt * dt)

            self.x += disp * np.cos(theta) + np.random.normal(0.0, pos_noise_std)
            self.y += disp * np.sin(theta) + np.random.normal(0.0, pos_noise_std)

            self.s = float(speed + accel * dt)
            self.a = float(accel)
            self.dir = float(dir_deg)
