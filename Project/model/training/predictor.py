import numpy as np
import pandas as pd
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryPredictor:
    def __init__(self, pipeline, model=None, device=None):
        self.pipeline = pipeline
        self.model = model or pipeline.model
        self.device = device or pipeline.DEVICE
        self.model.eval()

    def row_to_model_inputs(self, row):
        num_feats = np.array(
            [[row[col] for col in self.pipeline.INPUT_NUMERIC_FEATURES]],
            dtype=np.float32,
        )

        cat_feats = []
        for col in self.pipeline.INPUT_CATEGORICAL_FEATURES:
            idx_col = f"{col}_idx"
            cat_feats.append(row[idx_col])
        cat_feats = np.array([[cat_feats]], dtype=np.int64)

        H = self.pipeline.HORIZON
        num_dim = num_feats.shape[-1]
        cat_dim = cat_feats.shape[-1]

        num_seq = np.zeros((1, H, num_dim), dtype=np.float32)
        cat_seq = np.zeros((1, H, cat_dim), dtype=np.int64)
        num_seq[0, -1, :] = num_feats
        cat_seq[0, -1, :] = cat_feats

        return (
            torch.tensor(num_seq, dtype=torch.float32, device=self.device),
            torch.tensor(cat_seq, dtype=torch.long, device=self.device),
        )

    def predict_future(self, row):
        numeric, categorical = self.row_to_model_inputs(row)
        with torch.no_grad():
            preds = self.model(numeric, categorical)
        preds = preds.detach().cpu().numpy()[0]
        return preds

    def predict(self, row):
        return self.predict_future(row)

    def predict_next_position(self, row, step: int = 0):
        future = self.predict_future(row)
        H = future.shape[0]
        step = max(0, min(step, H - 1))
        x_next, y_next = future[step]
        return float(x_next), float(y_next)

    def predict_next_state(self, row, step: int = 0):
        x_next, y_next = self.predict_next_position(row, step=step)

        s = float(row.get("s", 0.0))
        a = float(row.get("a", 0.0))
        dir_deg = float(row.get("dir", 0.0))
        o = float(row.get("o", 0.0))

        return [x_next, y_next, s, a, dir_deg, o]

class TrajectoryTransformerPredictor:
    def __init__(self, model, norm_stats, device=None):
        self.model = model
        self.model.eval()
        self.device = device or DEVICE
        self.means, self.stds = norm_stats

    def _denormalize_future(self, future_norm: np.ndarray) -> np.ndarray:
        stds_safe = self.stds.replace(0, 1.0)

        cols = ["x", "y", "s", "a", "dir", "o"]
        future = future_norm.copy()

        for i, col in enumerate(cols):
            mu = self.means[col]
            sigma = stds_safe[col]
            future[:, i] = future[:, i] * sigma + mu

        return future

    def predict(self, model_input):
        src = model_input["src"]
        pos_idx = int(model_input["pos_idx"])
        role_idx = int(model_input["role_idx"])
        side_idx = int(model_input["side_idx"])

        src_tensor = torch.tensor(src, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_tensor = torch.tensor([pos_idx], dtype=torch.long, device=self.device)
        role_tensor = torch.tensor([role_idx], dtype=torch.long, device=self.device)
        side_tensor = torch.tensor([side_idx], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(
                src_tensor,
                tgt=None,
                pos_idx=pos_tensor,
                role_idx=role_tensor,
                side_idx=side_tensor,
                teacher_forcing=False,
            )

        future_norm = out.squeeze(0).detach().cpu().numpy()
        future = self._denormalize_future(future_norm)
        return future

    def predict_next_state(self, model_input, step: int | None = 0):
        future = self.predict(model_input)
        T = future.shape[0]

        if step is None:
            step = 0

        idx = max(0, min(step, T - 1))
        return future[idx].tolist()
