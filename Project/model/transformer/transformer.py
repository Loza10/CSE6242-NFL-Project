import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_height(h):
    if isinstance(h, str) and '-' in h:
        ft, inch = h.split('-')
        try:
            return int(ft) * 12 + int(inch)
        except Exception:
            return 0
    return 0

def feature_engineer(df):
    df = df.copy()
    if 'player_height' in df.columns:
        df['player_height_in'] = df['player_height'].apply(parse_height)
    else:
        df['player_height_in'] = 0

    if 'player_weight' in df.columns:
        df['player_weight_lbs'] = df['player_weight']
    else:
        df['player_weight_lbs'] = 0

    return df

def align_play_direction(df):
    df = df.copy()
    if 'play_direction' in df.columns:
        mask = df['play_direction'] == 'left'
        if mask.any():
            df.loc[mask, 'x'] = 120.0 - df.loc[mask, 'x']
            if 'dir' in df.columns:
                df.loc[mask, 'dir'] = (df.loc[mask, 'dir'] + 180) % 360
            if 'o' in df.columns:
                df.loc[mask, 'o'] = (df.loc[mask, 'o'] + 180) % 360
    return df

def wrap_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0

def add_engineered_features(df):
    df = df.copy()

    for c in ["dir", "o", "s", "a", "x", "y"]:
        if c not in df.columns:
            df[c] = 0.0

    df["dir_rad"] = np.deg2rad(df["dir"])
    df["o_rad"] = np.deg2rad(df["o"])

    df["vx"] = df["s"] * np.cos(df["dir_rad"])
    df["vy"] = df["s"] * np.sin(df["dir_rad"])

    df["ax_comp"] = df["a"] * np.cos(df["dir_rad"])
    df["ay_comp"] = df["a"] * np.sin(df["dir_rad"])

    df["sin_dir"] = np.sin(df["dir_rad"])
    df["cos_dir"] = np.cos(df["dir_rad"])
    df["sin_o"] = np.sin(df["o_rad"])
    df["cos_o"] = np.cos(df["o_rad"])

    df["orientation_signed_diff"] = wrap_angle_deg(df["o"] - df["dir"])
    df["orientation_diff_abs"] = np.abs(df["orientation_signed_diff"])

    if "nfl_id" in df.columns:
        df["angular_vel_dir"] = (
            df.groupby("nfl_id")["dir"].diff().fillna(0).apply(wrap_angle_deg)
        )
        df["angular_vel_o"] = (
            df.groupby("nfl_id")["o"].diff().fillna(0).apply(wrap_angle_deg)
        )
        df["ds_dt"] = df.groupby("nfl_id")["s"].diff().fillna(0)
        df["d_dir_dt"] = (
            df.groupby("nfl_id")["dir"].diff().fillna(0).apply(wrap_angle_deg)
        )
    else:
        df["angular_vel_dir"] = 0.0
        df["angular_vel_o"] = 0.0
        df["ds_dt"] = 0.0
        df["d_dir_dt"] = 0.0

    FIELD_X_MIN = 0.0
    FIELD_X_MAX = 120.0
    FIELD_Y_MAX = 53.3

    df["dist_left_sideline"] = df["x"] - FIELD_X_MIN
    df["dist_right_sideline"] = FIELD_X_MAX - df["x"]
    df["dist_endzone"] = FIELD_Y_MAX - df["y"]

    field_half_width = (FIELD_X_MAX - FIELD_X_MIN) / 2.0
    df["sideline_proximity"] = (
        np.minimum(df["dist_left_sideline"], df["dist_right_sideline"])
        / max(field_half_width, 1e-6)
    )

    if "player_weight" in df.columns:
        df["momentum_x"] = df["vx"] * df["player_weight"]
        df["momentum_y"] = df["vy"] * df["player_weight"]
        df["kinetic_energy"] = 0.5 * df["player_weight"] * (df["s"] ** 2)
    else:
        df["momentum_x"] = df["vx"]
        df["momentum_y"] = df["vy"]
        df["kinetic_energy"] = df["s"] ** 2

    if "nearest_defender_dist" in df.columns:
        df["defender_dist"] = df["nearest_defender_dist"]
        df["defender_angle_diff"] = wrap_angle_deg(
            df["dir"]
            - df.get("nearest_defender_dir", df["dir"])
        )
    else:
        df["defender_dist"] = 99.0
        df["defender_angle_diff"] = 0.0

    df["player_height_in"] = df["player_height"].apply(parse_height) if "player_height" in df else 0.0
    df["player_weight_lbs"] = df["player_weight"].astype(float) if "player_weight" in df else 0.0

    for c in ["num_frames_output", "ball_land_x", "ball_land_y"]:
        if c not in df.columns:
            df[c] = 0.0

    return df


TRANSFORMER_FEATURE_COLS = [
    "x", "y", "s", "a", "dir", "o",
    "player_height_in", "player_weight_lbs",
    "num_frames_output", "ball_land_x", "ball_land_y",
    "dir_rad", "o_rad", "vx", "vy", "ax_comp", "ay_comp",
    "sin_dir", "cos_dir", "sin_o", "cos_o",
    "orientation_signed_diff", "orientation_diff_abs",
    "angular_vel_dir", "angular_vel_o", "ds_dt", "d_dir_dt",
    "dist_left_sideline", "dist_right_sideline", "dist_endzone",
    "sideline_proximity", "momentum_x", "momentum_y", "kinetic_energy",
    "defender_dist", "defender_angle_diff",
]

def transformer_feature_engineer(df, fit_stats=None, feature_cols=None):
    df = df.copy()

    if feature_cols is None:
        feature_cols = TRANSFORMER_FEATURE_COLS

    # ensure all columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    feature_cols = [c for c in feature_cols if c in df.columns]

    if fit_stats is not None:
        means, stds = fit_stats
        stds_safe = stds.replace(0, 1.0)

        means_f = means[feature_cols]
        stds_f = stds_safe[feature_cols]

        df[feature_cols] = (df[feature_cols] - means_f) / stds_f

    return df

def compute_transformer_feature_stats(df):
    df_eng = add_engineered_features(df)
    df_aligned = align_play_direction(df_eng)
    df_fe = transformer_feature_engineer(df_aligned, fit_stats=None, feature_cols=TRANSFORMER_FEATURE_COLS)

    means = df_fe[TRANSFORMER_FEATURE_COLS].mean()
    stds = df_fe[TRANSFORMER_FEATURE_COLS].std()
    return means, stds


position_matchups = {
    "CB": ["WR"],
    "FS": ["WR", "TE"],
    "SS": ["TE", "RB"],
    "S": ["WR", "TE"],
    "OLB": ["RB", "TE"],
    "ILB": ["RB", "FB", "TE"],
    "MLB": ["RB", "FB", "TE"],
    "DE": ["QB", "RB", "FB"],
    "DT": ["QB", "RB", "FB"],
    "NT": ["QB", "RB", "FB"],
    "QB": ["DE", "DT", "NT", "OLB", "ILB"],
    "RB": ["ILB", "MLB", "OLB", "SS"],
    "FB": ["ILB", "MLB", "DT", "NT"],
    "TE": ["OLB", "ILB", "SS", "FS"],
    "WR": ["CB", "FS", "S"],
}

opposite_role_map = {
    "Passer": ["Defensive Coverage"],
    "Targeted Receiver": ["Defensive Coverage"],
    "Other Route Runner": ["Defensive Coverage"],
    "Defensive Coverage": ["Passer", "Targeted Receiver", "Other Route Runner"],
}

side_map = {
    "Offense": "Defense",
    "Defense": "Offense",
}

pos_vocab = sorted(
    set(position_matchups.keys())
    | {v for vals in position_matchups.values() for v in vals}
)
role_vocab = sorted(
    set(opposite_role_map.keys())
    | {v for vals in opposite_role_map.values() for v in vals}
)
side_vocab = sorted(side_map.keys())

pos2idx = {p: i for i, p in enumerate(pos_vocab)}
role2idx = {r: i for i, r in enumerate(role_vocab)}
side2idx = {s: i for i, s in enumerate(side_vocab)}

class TrajectoryDataset(Dataset):
    def __init__(self, input_df, output_df, past_frames=10, future_frames=10, norm_stats=None):
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.norm_stats = norm_stats

        input_eng = add_engineered_features(input_df)
        output_eng = add_engineered_features(output_df)

        input_aligned = align_play_direction(input_eng)
        output_aligned = align_play_direction(output_eng)

        self.input = transformer_feature_engineer(
            input_aligned,
            fit_stats=norm_stats,
            feature_cols=TRANSFORMER_FEATURE_COLS,
        )
        self.output = transformer_feature_engineer(
            output_aligned,
            fit_stats=norm_stats,
            feature_cols=["x", "y", "s", "a", "dir", "o"],
        )

        self.feature_cols = TRANSFORMER_FEATURE_COLS

        grouped_input = self.input.groupby(["game_id", "play_id", "nfl_id"])
        grouped_output = self.output.groupby(["game_id", "play_id", "nfl_id"])

        self.keys = []
        for key in grouped_input.groups.keys():
            if key not in grouped_output.groups:
                continue
            if (
                len(grouped_input.get_group(key)) >= self.past_frames
                and len(grouped_output.get_group(key)) >= self.future_frames
            ):
                self.keys.append(key)

        if len(self.keys) == 0:
            raise ValueError("No valid examples found in TrajectoryDataset.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        game_id, play_id, nfl_id = self.keys[idx]

        inp = self.input[
            (self.input.game_id == game_id)
            & (self.input.play_id == play_id)
            & (self.input.nfl_id == nfl_id)
        ].sort_values("frame_id")

        tgt = self.output[
            (self.output.game_id == game_id)
            & (self.output.play_id == play_id)
            & (self.output.nfl_id == nfl_id)
        ].sort_values("frame_id")

        src_seq = inp[self.feature_cols].values[-self.past_frames:]
        if src_seq.shape[0] < self.past_frames:
            pad = np.zeros((self.past_frames - src_seq.shape[0], src_seq.shape[1]), dtype=np.float32)
            src_seq = np.vstack([pad, src_seq])

        tgt_seq = tgt[["x", "y", "s", "a", "dir", "o"]].values[: self.future_frames]
        if tgt_seq.shape[0] < self.future_frames:
            pad = np.zeros((self.future_frames - tgt_seq.shape[0], 6), dtype=np.float32)
            tgt_seq = np.vstack([tgt_seq, pad])

        raw_input = inp.sort_values("frame_id").iloc[-1]
        pos_idx_val = pos2idx.get(raw_input.get("player_position", None), 0)
        role_idx_val = role2idx.get(raw_input.get("player_role", None), 0)
        side_idx_val = side2idx.get(raw_input.get("player_side", None), 0)

        return (
            torch.tensor(src_seq, dtype=torch.float32),
            torch.tensor(tgt_seq, dtype=torch.float32),
            torch.tensor(pos_idx_val, dtype=torch.long),
            torch.tensor(role_idx_val, dtype=torch.long),
            torch.tensor(side_idx_val, dtype=torch.long),
        )

def collate_fn(batch):
    src = torch.stack([b[0] for b in batch]).to(DEVICE)
    tgt = torch.stack([b[1] for b in batch]).to(DEVICE)
    pos_idx = torch.stack([b[2] for b in batch]).to(DEVICE)
    role_idx = torch.stack([b[3] for b in batch]).to(DEVICE)
    side_idx = torch.stack([b[4] for b in batch]).to(DEVICE)
    return src, tgt, pos_idx, role_idx, side_idx


class TrajectoryTransformerWithCats(nn.Module):
    def __init__(
        self,
        input_feat_dim,
        n_pos, 
        n_role, 
        n_side,
        pos_emb_dim=4,
        role_emb_dim=2,
        side_emb_dim=1,
        hidden_dim=128,
        nhead=4,
        num_layers=2,
        future_frames=10,
        max_pos=1000,
    ):
        super().__init__()
        self.future_frames = future_frames

        self.pos_emb = nn.Embedding(n_pos, pos_emb_dim)
        self.role_emb = nn.Embedding(n_role, role_emb_dim)
        self.side_emb = nn.Embedding(n_side, side_emb_dim)

        self.cat_total_dim = pos_emb_dim + role_emb_dim + side_emb_dim

        proj_input_dim = input_feat_dim + self.cat_total_dim
        self.src_proj = nn.Linear(proj_input_dim, hidden_dim)

        self.tgt_proj = nn.Linear(6 + self.cat_total_dim, hidden_dim)

        self.pos_enc = nn.Parameter(torch.randn(max_pos, hidden_dim))

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_dim, 6)

    def forward(
        self,
        src,
        tgt=None,
        pos_idx=None,
        role_idx=None,
        side_idx=None,
        teacher_forcing=True,
        scheduled_sampling_prob=0.0,
    ):
        B, past_len, _ = src.shape
        device = src.device

        pos_v = self.pos_emb(pos_idx)   # (B, pos_emb_dim)
        role_v = self.role_emb(role_idx)
        side_v = self.side_emb(side_idx)
        cat_v = torch.cat([pos_v, role_v, side_v], dim=1)

        cat_enc = cat_v.unsqueeze(1).expand(-1, past_len, -1)
        src_cat = torch.cat([src, cat_enc], dim=2)
        src_emb = self.src_proj(src_cat) + self.pos_enc[:past_len, :].unsqueeze(0).to(device)
        memory = self.transformer.encoder(src_emb)

        last_state = src[:, -1:, :6]

        if tgt is not None:
            outputs = []
            decoder_inputs = last_state.clone()
            cat_dec = cat_v.unsqueeze(1)

            for t in range(self.future_frames):
                cat_rep = cat_dec.expand(-1, decoder_inputs.size(1), -1)
                dec_cat = torch.cat([decoder_inputs, cat_rep], dim=2)
                dec_emb = (
                    self.tgt_proj(dec_cat)
                    + self.pos_enc[: dec_cat.size(1), :].unsqueeze(0).to(device)
                )
                dec_out = self.transformer.decoder(dec_emb, memory)
                pred = self.output_proj(dec_out[:, -1:, :])
                outputs.append(pred)

                if teacher_forcing:
                    use_pred = torch.rand(1).item() < scheduled_sampling_prob
                    next_in = pred.detach() if use_pred else tgt[:, t : t + 1, :].to(device)
                else:
                    next_in = pred.detach()

                decoder_inputs = torch.cat([decoder_inputs, next_in], dim=1)

            return torch.cat(outputs, dim=1)

        outputs = []
        decoder_inputs = last_state.clone()
        cat_dec = cat_v.unsqueeze(1)

        for t in range(self.future_frames):
            cat_rep = cat_dec.expand(-1, decoder_inputs.size(1), -1)
            dec_cat = torch.cat([decoder_inputs, cat_rep], dim=2)
            dec_emb = (
                self.tgt_proj(dec_cat)
                + self.pos_enc[: dec_cat.size(1), :].unsqueeze(0).to(device)
            )
            dec_out = self.transformer.decoder(dec_emb, memory)
            pred = self.output_proj(dec_out[:, -1:, :])
            outputs.append(pred)
            decoder_inputs = torch.cat([decoder_inputs, pred], dim=1)

        return torch.cat(outputs, dim=1)


TrajectoryTransformer = TrajectoryTransformerWithCats
