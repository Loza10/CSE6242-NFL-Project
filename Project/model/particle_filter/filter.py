import numpy as np
import statistics
from model.particle_filter.particle import Particle


def _wrap_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


class Filter:
    def __init__(self, df, motion_model=None):
        self.df = df
        self.particles = []
        self.motion_model = motion_model

    def set_motion_model(self, motion_model):
        self.motion_model = motion_model

    def placeParticles(self, num, play_id, player_name):
        player_df = self.df[(self.df["play_id"] == play_id) & (self.df["player_name"] == player_name)].sort_values("frame_id")

        if player_df.empty:
            print(f"No data found for {player_name} in play {play_id}")
            return

        row0 = player_df.iloc[0]

        x0 = float(row0["x"])
        y0 = float(row0["y"])
        s0 = float(row0.get("s", 0.0))
        a0 = float(row0.get("a", 0.0))
        dir0 = float(row0.get("dir", 0.0))
        o0 = float(row0.get("o", 0.0))

        if len(player_df) > 1:
            try:
                x_std = statistics.stdev(player_df["x"])
                y_std = statistics.stdev(player_df["y"])
            except statistics.StatisticsError:
                x_std = 0.5
                y_std = 0.5
        else:
            x_std = 0.5
            y_std = 0.5

        x_offsets = np.random.uniform(-x_std, x_std, num)
        y_offsets = np.random.uniform(-y_std, y_std, num)

        s_offsets = np.random.normal(0.0, 0.2, num)
        a_offsets = np.random.normal(0.0, 0.2, num)
        dir_offsets = np.random.normal(0.0, 5.0, num)
        o_offsets = np.random.normal(0.0, 5.0, num)

        self.particles = [
            Particle(x0 + dx, y0 + dy, s0 + ds, a0 + da, dir0 + ddir, o0 + do)
            for dx, dy, ds, da, ddir, do in zip(
                x_offsets, y_offsets, s_offsets, a_offsets, dir_offsets, o_offsets
            )
        ]

    def predict(self, speed, dir_deg, accel, model_input=None):
        for p in self.particles:
            p.moveParticle(speed, accel, dir_deg, model=self.motion_model, model_input=model_input)

    def update(self, real_x, real_y, real_s = 0.0, real_a = 0.0, real_dir = 0.0, real_o = 0.0):
        scale_xy = 5.0
        scale_s = 2.0
        scale_a = 2.0
        scale_dir = 10.0
        scale_o = 10.0

        total = 0.0

        for p in self.particles:
            dist_xy = np.sqrt((p.x - real_x) ** 2 + (p.y - real_y) ** 2)

            ds = p.s - real_s
            da = p.a - real_a
            ddir = _wrap_angle_deg(p.dir - real_dir)
            do = _wrap_angle_deg(p.o - real_o)

            dist = (dist_xy / scale_xy + abs(ds) / scale_s + abs(da) / scale_a + abs(ddir) / scale_dir + abs(do) / scale_o)

            p.weight = np.exp(-dist)
            total += p.weight

        if total == 0.0 and len(self.particles) > 0:
            uniform = 1.0 / len(self.particles)
            for p in self.particles:
                p.weight = uniform
        elif total > 0.0:
            for p in self.particles:
                p.weight /= total

    def resample(self):
        N = len(self.particles)
        if N == 0:
            return

        weights = np.array([p.weight for p in self.particles], dtype=np.float64)
        weights /= weights.sum()

        positions = (np.arange(N) + np.random.uniform(0.0, 1.0)) / N
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        indexes = np.searchsorted(cumulative_sum, positions)

        new_particles = []
        for idx in indexes:
            p = self.particles[idx]

            new_x = p.x + np.random.normal(0.0, 0.1)
            new_y = p.y + np.random.normal(0.0, 0.1)
            new_s = p.s + np.random.normal(0.0, 0.05)
            new_a = p.a + np.random.normal(0.0, 0.05)
            new_dir = p.dir + np.random.normal(0.0, 1.0)
            new_o = p.o + np.random.normal(0.0, 1.0)

            new_particles.append(
                Particle(
                    new_x,
                    new_y,
                    new_s,
                    new_a,
                    new_dir,
                    new_o,
                    weight=1.0 / N,
                )
            )

        self.particles = new_particles

    def estimate(self):
        if not self.particles:
            return None

        xs = np.array([p.x for p in self.particles], dtype=np.float64)
        ys = np.array([p.y for p in self.particles], dtype=np.float64)
        ss = np.array([p.s for p in self.particles], dtype=np.float64)
        aa = np.array([p.a for p in self.particles], dtype=np.float64)
        dirs = np.array([p.dir for p in self.particles], dtype=np.float64)
        os = np.array([p.o for p in self.particles], dtype=np.float64)
        ws = np.array([p.weight for p in self.particles], dtype=np.float64)

        ws_sum = ws.sum()
        if ws_sum == 0.0:
            ws = np.ones_like(ws) / len(ws)

        return (
            float(np.average(xs, weights=ws)),
            float(np.average(ys, weights=ws)),
            float(np.average(ss, weights=ws)),
            float(np.average(aa, weights=ws)),
            float(np.average(dirs, weights=ws)),
            float(np.average(os, weights=ws)),
        )
