import glob
import numpy as np
import pandas as pd
import torch

from model.training.trainer import TrajectoryTrainer, TrajectoryTransformerTrainer
from model.training.predictor import (
    TrajectoryPredictor,
    TrajectoryTransformerPredictor,
)
from model.transformer.transformer import (
    TrajectoryTransformerWithCats,
    feature_engineer,
    align_play_direction,
    add_engineered_features,
    transformer_feature_engineer,
    compute_transformer_feature_stats,
    TRANSFORMER_FEATURE_COLS,
    pos2idx,
    role2idx,
    side2idx,
)
from model.neural_network.neural_net import TrajectoryPipeline
from model.particle_filter.filter import Filter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FullModel:
    def __init__(self, model_type="neural_net", transformer_epochs=10, nn_epochs=20, lr=1e-3, num_particles=200, past_frames=10):
        if model_type is None or str(model_type).lower() == "none":
            self.model_type = None
        else:
            self.model_type = model_type

        self.transformer_epochs = transformer_epochs
        self.nn_epochs = nn_epochs
        self.lr = lr
        self.num_particles = num_particles
        self.past_frames = past_frames

        self.models = {}
        self.pipelines = {}
        self.transformer_feature_cols = {}

        self.transformer_model = None
        self.transformer_norm_stats = None
        self.transformer_feature_cols_global = TRANSFORMER_FEATURE_COLS
        self.pos2idx = pos2idx
        self.role2idx = role2idx
        self.side2idx = side2idx

        self.df_in_raw = None
        self.df_in_simple = None

    def load_all_weeks(self):
        input_files = sorted(glob.glob("Project/data/train/input_2023_w*.csv"))
        output_files = sorted(glob.glob("Project/data/train/output_2023_w*.csv"))

        if len(input_files) == 0 or len(output_files) == 0:
            raise FileNotFoundError("No weekly CSVs found in Project/data/train/")

        df_in = pd.concat([pd.read_csv(f) for f in input_files], ignore_index=True)
        df_out = pd.concat([pd.read_csv(f) for f in output_files], ignore_index=True)

        print(f"Loaded {len(input_files)} input files | {len(output_files)} output files")
        return df_in, df_out

    def load_pretrained_transformer(self):
        if self.transformer_model is not None:
            return

        input_feat_dim = len(self.transformer_feature_cols_global)
        n_pos = len(self.pos2idx)
        n_role = len(self.role2idx)
        n_side = len(self.side2idx)

        model = TrajectoryTransformerWithCats(
            input_feat_dim=input_feat_dim,
            n_pos=n_pos,
            n_role=n_role,
            n_side=n_side,
            pos_emb_dim=4,
            role_emb_dim=2,
            side_emb_dim=1,
            hidden_dim=128,
            nhead=4,
            num_layers=2,
            future_frames=self.past_frames,
        ).to(DEVICE)

        ckpt_path = "Project/model/transformer/model_weights.pth"
        print(f"Loading pretrained transformer weights from: {ckpt_path}")
        chk = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(chk)
        model.eval()

        self.transformer_model = model

    def train_transformer_for_position(self, pos_df_in, pos_df_out):
        return self.transformer_model, self.transformer_feature_cols_global

    def train_neural_net_for_position(self, pos_df_in, pos_df_out):
        test_subset = pos_df_in.sample(min(50, len(pos_df_in))).copy()
        pipeline = TrajectoryPipeline(pos_df_in, pos_df_out, test_subset)

        trainer = TrajectoryTrainer(pipeline, lr=self.lr, epochs=self.nn_epochs)
        trained = trainer.train()

        return trained, pipeline

    def build_nn_input(self, row, pipeline):
        r = row.copy()

        for raw_col, flag_col in pipeline.BINARY_TO_NUMERIC_COLUMNS.items():
            if raw_col in r:
                mapping = pipeline.df_in[[raw_col, flag_col]].drop_duplicates()
                raw_val = r[raw_col]
                mapped = mapping[mapping[raw_col] == raw_val][flag_col]
                r[flag_col] = float(mapped.iloc[0]) if len(mapped) > 0 else float(
                    pipeline.df_in[flag_col].min()
                )
            else:
                r[flag_col] = 0.0

        for col in pipeline.INPUT_CATEGORICAL_FEATURES:
            idx_col = f"{col}_idx"
            token = str(r[col]) if col in r else "<unk>"
            r[idx_col] = pipeline.CATEGORY_METADATA[col]["mapping"].get(token, 0)

        return r

    def build_transformer_model_input(self, row):
        if self.df_in_raw is None:
            raise RuntimeError("df_in_raw is not set. Make sure run() was called.")

        df = self.df_in_raw

        game_id, play_id, nfl_id = row["game_id"], row["play_id"], row["nfl_id"]

        past_df = df[
            (df.game_id == game_id)
            & (df.play_id == play_id)
            & (df.nfl_id == nfl_id)
        ].sort_values("frame_id")

        past_df = past_df.tail(self.past_frames).copy()

        past_eng = add_engineered_features(past_df)
        past_aligned = align_play_direction(past_eng)
        past_norm = transformer_feature_engineer(
            past_aligned,
            fit_stats=self.transformer_norm_stats,
            feature_cols=self.transformer_feature_cols_global,
        )

        src_seq = past_norm[self.transformer_feature_cols_global].values.astype(
            np.float32
        )
        if src_seq.shape[0] < self.past_frames:
            pad = np.zeros(
                (self.past_frames - src_seq.shape[0], src_seq.shape[1]),
                dtype=np.float32,
            )
            src_seq = np.vstack([pad, src_seq])

        pos_name = row.get("player_position", None)
        role_name = row.get("player_role", None)
        side_name = row.get("player_side", None)

        pos_idx_val = self.pos2idx.get(pos_name, 0)
        role_idx_val = self.role2idx.get(role_name, 0)
        side_idx_val = self.side2idx.get(side_name, 0)

        return {
            "src": src_seq,
            "pos_idx": pos_idx_val,
            "role_idx": role_idx_val,
            "side_idx": side_idx_val,
        }

    def evaluate_particle_filter(
        self,
        row,
        df_in,
        df_out,
        motion_model,
        feat_cols=None,
        pipeline=None,
    ):
        game_id, play_id, nfl_id = row["game_id"], row["play_id"], row["nfl_id"]
        player_name = row["player_name"]

        # Fix: model_type None means physics only
        if self.model_type is None:
            model_input = None
        elif self.model_type == "neural_net":
            model_input = self.build_nn_input(row, pipeline)
        elif self.model_type == "transformer":
            model_input = self.build_transformer_model_input(row)
        else:
            model_input = None

        PF = Filter(df_in, motion_model=motion_model)
        PF.placeParticles(self.num_particles, play_id, player_name)

        PF.predict(
            speed=row["s"],
            dir_deg=row["dir"],
            accel=row["a"],
            model_input=model_input,
        )

        gt = df_out[
            (df_out.game_id == game_id)
            & (df_out.play_id == play_id)
            & (df_out.nfl_id == nfl_id)
        ]

        if gt.empty:
            return None

        real_row = gt.sort_values("frame_id").iloc[0]
        real_x = float(real_row["x"])
        real_y = float(real_row["y"])
        real_s = float(real_row.get("s", 0.0))
        real_a = float(real_row.get("a", 0.0))
        real_dir = float(real_row.get("dir", 0.0))
        real_o = float(real_row.get("o", 0.0))

        PF.update(real_x, real_y, real_s, real_a, real_dir, real_o)
        PF.resample()

        est_state = PF.estimate()
        if est_state is None:
            return None

        est_x, est_y, est_s, est_a, est_dir, est_o = est_state
        return real_x, real_y, est_x, est_y, est_s, est_a, est_dir, est_o

    def train_all_positions(self, df_in, df_out):
        positions = df_in["player_position"].unique()

        for pos in positions:
            print(f"\n=== INITIALIZING MODEL FOR POSITION: {pos} ===")

            pos_df_in = df_in[
                (df_in.player_position == pos)
                & (df_in.player_to_predict == True)
            ]

            if len(pos_df_in) < 2:
                print("Not enough samples.")
                continue

            g_ids = pos_df_in["game_id"].unique()
            p_ids = pos_df_in["play_id"].unique()
            n_ids = pos_df_in["nfl_id"].unique()

            pos_df_out = df_out[
                (df_out.game_id.isin(g_ids))
                & (df_out.play_id.isin(p_ids))
                & (df_out.nfl_id.isin(n_ids))
            ]

            if pos_df_out.empty:
                print("No matching outputs.")
                continue

            if self.model_type == "transformer":
                model, feat_cols = self.train_transformer_for_position(
                    pos_df_in, pos_df_out
                )
                if model is None:
                    continue
                self.models[pos] = model
                self.transformer_feature_cols[pos] = feat_cols

            elif self.model_type == "neural_net":
                model, pipeline = self.train_neural_net_for_position(
                    pos_df_in, pos_df_out
                )
                self.models[pos] = model
                self.pipelines[pos] = pipeline

            elif self.model_type is None:
                # Physics-only; nothing to train
                self.models[pos] = None

            print(f"Stored model for {pos}")

    def evaluate_all_positions(self, df_in, df_out):
        total_dist = 0.0
        count = 0

        print("\n============== PARTICLE FILTER EVALUATION ==============\n")

        for pos, model in self.models.items():
            print(f"\n--- Evaluating PF for position {pos} ---")

            pos_rows = df_in[
                (df_in.player_position == pos)
                & (df_in.player_to_predict == True)
            ]
            unique_players = pos_rows.groupby(["game_id", "play_id", "nfl_id"])

            for (game_id, play_id, nfl_id), group in unique_players:
                row = group.sort_values("frame_id").iloc[0]
                player_name = row["player_name"]

                if self.model_type == "transformer":
                    motion_model = TrajectoryTransformerPredictor(
                        self.transformer_model,
                        norm_stats=self.transformer_norm_stats,
                        device=DEVICE,
                    )
                    feat_cols = self.transformer_feature_cols.get(
                        pos, self.transformer_feature_cols_global
                    )
                    pipeline = None

                elif self.model_type == "neural_net":
                    pipeline = self.pipelines[pos]
                    motion_model = TrajectoryPredictor(
                        pipeline=pipeline,
                        model=model,
                        device=DEVICE,
                    )
                    feat_cols = None

                elif self.model_type is None:
                    motion_model = None
                    feat_cols = None
                    pipeline = None

                result = self.evaluate_particle_filter(
                    row,
                    df_in,
                    df_out,
                    motion_model=motion_model,
                    feat_cols=feat_cols,
                    pipeline=pipeline,
                )

                if result is None:
                    continue

                (
                    real_x,
                    real_y,
                    est_x,
                    est_y,
                    est_s,
                    est_a,
                    est_dir,
                    est_o,
                ) = result

                final_gt_row = df_out[
                    (df_out.game_id == game_id)
                    & (df_out.play_id == play_id)
                    & (df_out.nfl_id == nfl_id)
                ].sort_values("frame_id").iloc[-1]

                final_gt_x = float(final_gt_row["x"])
                final_gt_y = float(final_gt_row["y"])

                final_dist = np.sqrt(
                    (final_gt_x - est_x) ** 2 + (final_gt_y - est_y) ** 2
                )

                print(
                    f"Player: {player_name} | Play {play_id} "
                    f"\n| GT Final Position: ({final_gt_x:.2f}, {final_gt_y:.2f}) "
                    f"\n| PF Final Estimate: ({est_x:.2f}, {est_y:.2f}) "
                    f"\n| Final Distance: {final_dist:.2f}"
                )

                total_dist += final_dist
                count += 1

        if count > 0:
            print(
                f"\nMean PF final distance across all predictions: {total_dist / count:.2f}\n"
            )
        else:
            print("\nNo predictions evaluated.\n")

    def run(self):
        df_in_raw, df_out = self.load_all_weeks()
        self.df_in_raw = df_in_raw.copy()

        print("Computing transformer feature normalization stats...")
        self.transformer_norm_stats = compute_transformer_feature_stats(
            self.df_in_raw
        )

        df_in = feature_engineer(align_play_direction(df_in_raw))
        self.df_in_simple = df_in

        if self.model_type == "transformer":
            self.load_pretrained_transformer()

        self.train_all_positions(df_in, df_out)
        self.evaluate_all_positions(df_in, df_out)
