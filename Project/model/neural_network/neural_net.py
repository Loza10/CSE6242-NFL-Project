import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm


class TrajectoryPipeline:
    def __init__(self, df_in, df_out, test_in, DEVICE="cuda" if torch.cuda.is_available() else "cpu"):
        self.df_in = df_in.copy().reset_index(drop=True)
        self.df_out = df_out.copy().reset_index(drop=True)
        self.test_in = test_in.copy().reset_index(drop=True)
        self.DEVICE = DEVICE

        self.HORIZON = 10
        self.INPUT_NUMERIC_FEATURES = [
            "x",
            "y",
            "s",
            "a",
            "dir",
            "o",
            "player_weight",
            "play_direction_flag",
            "player_side_flag",
        ]
        self.INPUT_CATEGORICAL_FEATURES = [
            "player_position",
            "player_role",
        ]
        self.TARGET_FEATURES = ["x", "y"]
        self.KEY_COLUMNS = ["game_id", "play_id", "nfl_id"]

        self.BINARY_TO_NUMERIC_COLUMNS = {
            "play_direction": "play_direction_flag",
            "player_side": "player_side_flag",
        }

        for raw_col, numeric_col in self.BINARY_TO_NUMERIC_COLUMNS.items():
            combined = pd.concat([self.df_in[raw_col], self.test_in[raw_col]], axis=0)
            unique_values = sorted(combined.dropna().unique().tolist())
            if len(unique_values) == 0:
                raise ValueError(f"Column '{raw_col}' has no values after filtering for this subset — cannot create flag (combined series empty).")
            value_to_float = {value: float(idx) for idx, value in enumerate(unique_values)}
            self.df_in[numeric_col] = self.df_in[raw_col].map(value_to_float).astype(np.float32)
            self.test_in[numeric_col] = self.test_in[raw_col].map(value_to_float).astype(np.float32)

        self.CATEGORY_METADATA = {}
        self.CAT_IDX_COLUMNS = []
        for column in self.INPUT_CATEGORICAL_FEATURES:
            cleaned_train = self.df_in[column].fillna("<unk>").astype(str)
            vocab = sorted(cleaned_train.unique().tolist())
            mapping = {token: idx + 1 for idx, token in enumerate(vocab)}
            self.df_in[f"{column}_idx"] = cleaned_train.map(mapping).fillna(0).astype(np.int64)
            self.test_in[f"{column}_idx"] = (
                self.test_in[column]
                .fillna("<unk>")
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype(np.int64)
            )
            self.CATEGORY_METADATA[column] = {"mapping": mapping, "vocab_size": len(mapping) + 1}
            self.CAT_IDX_COLUMNS.append(f"{column}_idx")

        train_sequences = self.build_sequences(self.df_in, self.df_out)
        if train_sequences is None:
            raise RuntimeError("Training sequences could not be constructed. Check the input data.")
        (
            self.all_numeric_sequences,
            self.all_categorical_sequences,
            self.all_target_sequences,
            self.sequence_metadata,
        ) = train_sequences

        self.CATEGORY_VOCAB_SIZES = {
            col: meta["vocab_size"] for col, meta in self.CATEGORY_METADATA.items()
        }
        self.BATCH_SIZE = 256

        (
            (train_num, train_cat, train_targets, train_meta),
            (val_num, val_cat, val_targets, val_meta),
            (test_num, test_cat, test_targets, test_meta),
        ) = self.train_val_test_split(
            self.all_numeric_sequences,
            self.all_categorical_sequences,
            self.all_target_sequences,
            self.sequence_metadata,
        )

        self.train_dataset = self.TrajectoryDataset(train_num, train_cat, train_targets)
        self.val_dataset = self.TrajectoryDataset(val_num, val_cat, val_targets)
        self.test_dataset = self.TrajectoryDataset(test_num, test_cat, test_targets)

        if len(self.train_dataset) == 0:
            raise RuntimeError("Train dataset is empty after splitting — cannot create training DataLoader.")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE) if len(self.val_dataset) > 0 else None
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE) if len(self.test_dataset) > 0 else None

        torch.manual_seed(42)
        self.model = self.TrajectoryCNN(categorical_vocab_sizes=self.CATEGORY_VOCAB_SIZES).to(self.DEVICE)
        print(self.model)

    def build_sequences(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        future_lookup = {
            key: group.sort_values("frame_id")
            for key, group in output_df.groupby(self.KEY_COLUMNS)
        }

        numeric_sequences = []
        categorical_sequences = []
        target_sequences = []
        metadata_records = []

        grouped_inputs = input_df.groupby(self.KEY_COLUMNS)
        grouped_iter = tqdm(
            grouped_inputs,
            total=getattr(grouped_inputs, "ngroups", None),
            desc="Building sequences",
        )
        for idx, (key, hist) in enumerate(grouped_iter):
            future = future_lookup.get(key)
            if future is None:
                continue

            hist_sorted = hist.sort_values("frame_id")
            hist_num = hist_sorted[self.INPUT_NUMERIC_FEATURES].to_numpy(dtype=np.float32)
            hist_cat = hist_sorted[self.CAT_IDX_COLUMNS].to_numpy(dtype=np.int64)

            hist_slice = hist_num[-self.HORIZON:]
            hist_padded = np.zeros((self.HORIZON, len(self.INPUT_NUMERIC_FEATURES)), dtype=np.float32)
            hist_padded[-len(hist_slice):] = hist_slice

            cat_slice = hist_cat[-self.HORIZON:]
            cat_padded = np.zeros((self.HORIZON, len(self.CAT_IDX_COLUMNS)), dtype=np.int64)
            cat_padded[-len(cat_slice):] = cat_slice

            future_sorted = future
            future_vals = future_sorted[self.TARGET_FEATURES].to_numpy(dtype=np.float32)
            future_slice = future_vals[:self.HORIZON]
            future_frames = future_sorted["frame_id"].to_numpy(dtype=np.int64)[:self.HORIZON]
            future_padded = np.zeros((self.HORIZON, len(self.TARGET_FEATURES)), dtype=np.float32)
            future_padded[: len(future_slice)] = future_slice

            numeric_sequences.append(hist_padded)
            categorical_sequences.append(cat_padded)
            target_sequences.append(future_padded)
            metadata_records.append(
                {
                    "game_id": int(key[0]),
                    "play_id": int(key[1]),
                    "nfl_id": int(key[2]),
                    "player_role": (
                        hist_sorted["player_role"].iloc[0]
                        if "player_role" in hist_sorted.columns
                        else None
                    ),
                    "frame_ids": future_frames.tolist(),
                }
            )

        if not numeric_sequences:
            print(f"No overlapping sequences were built.")
            return None

        numeric_arr = np.stack(numeric_sequences)
        categorical_arr = np.stack(categorical_sequences)
        target_arr = np.stack(target_sequences)
        print(
            f"Prepared {numeric_arr.shape[0]} samples → numeric {numeric_arr.shape}, categorical {categorical_arr.shape}"
        )
        return numeric_arr, categorical_arr, target_arr, metadata_records

    class TrajectoryDataset(Dataset):
        def __init__(self, numeric: np.ndarray, categorical: np.ndarray, targets: np.ndarray):
            if numeric is None or numeric.size == 0:
                numeric = np.zeros((0, 10, len(TrajectoryPipeline().INPUT_NUMERIC_FEATURES))) if False else numeric
            self.numeric = torch.from_numpy(numeric) if numeric is not None and numeric.size > 0 else torch.empty((0,))
            self.categorical = torch.from_numpy(categorical) if categorical is not None and categorical.size > 0 else torch.empty((0,))
            self.targets = torch.from_numpy(targets) if targets is not None and targets.size > 0 else torch.empty((0,))

        def __len__(self):
            try:
                return self.numeric.shape[0]
            except Exception:
                return 0

        def __getitem__(self, idx):
            return self.numeric[idx], self.categorical[idx], self.targets[idx]

    def train_val_test_split(self, numeric, categorical, targets, metadata, ratios=(0.7, 0.15, 0.15)):
        assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
        n = len(numeric)
        if n == 0:
            raise RuntimeError("No samples available to split")

        indices = np.random.permutation(n)

        if n == 1:
            train_idx = indices[:1]
            val_idx = np.array([], dtype=int)
            test_idx = np.array([], dtype=int)
        else:
            train_end = max(1, int(n * ratios[0]))
            val_count = int(n * ratios[1])
            val_end = train_end + val_count
            if val_end >= n:
                val_end = n - 1
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]

        metadata_array = np.array(metadata, dtype=object)

        def select(arr, idxs):
            if isinstance(arr, np.ndarray):
                if len(idxs) == 0:
                    return arr[:0]
                return arr[idxs]
            else:
                return np.array(arr, dtype=object)[idxs].tolist()

        return (
            (select(numeric, train_idx), select(categorical, train_idx), select(targets, train_idx), metadata_array[train_idx].tolist() if len(train_idx)>0 else []),
            (select(numeric, val_idx), select(categorical, val_idx), select(targets, val_idx), metadata_array[val_idx].tolist() if len(val_idx)>0 else []),
            (select(numeric, test_idx), select(categorical, test_idx), select(targets, test_idx), metadata_array[test_idx].tolist() if len(test_idx)>0 else []),
        )

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)
            self.proj = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x):
            residual = self.proj(x)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout(x)
            x = self.bn2(self.conv2(x))
            x = x + residual
            return F.relu(x)

    class TrajectoryCNN(nn.Module):
        def __init__(
            self,
            *,
            numeric_dim: int = 9,
            categorical_vocab_sizes: dict[str, int],
            embed_dim: int = 12,
            horizon: int = 10,
            target_dim: int = 2,
        ):
            super().__init__()
            self.horizon = horizon
            self.target_dim = target_dim
            self.cat_cols = list(categorical_vocab_sizes.keys())

            self.embeddings = nn.ModuleDict(
                {col: nn.Embedding(vocab_size, embed_dim) for col, vocab_size in categorical_vocab_sizes.items()}
            )
            fused_dim = numeric_dim + len(self.cat_cols) * embed_dim

            self.backbone = nn.Sequential(
                TrajectoryPipeline.ConvBlock(fused_dim, 128, dropout=0.1),
                TrajectoryPipeline.ConvBlock(128, 256, dropout=0.15),
                TrajectoryPipeline.ConvBlock(256, 256, dropout=0.15),
                TrajectoryPipeline.ConvBlock(256, 256, dropout=0.2),
            )
            self.head = nn.Sequential(
                nn.Linear(256 * horizon, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, horizon * target_dim),
            )

        def forward(self, numeric, categorical):
            embeds = []
            for idx, col in enumerate(self.cat_cols):
                embeds.append(self.embeddings[col](categorical[:, :, idx]))
            cat_embed = torch.cat(embeds, dim=-1) if embeds else None
            fused = torch.cat([numeric, cat_embed], dim=-1) if cat_embed is not None else numeric
            fused = fused.transpose(1, 2)
            hidden = self.backbone(fused).flatten(start_dim=1)
            out = self.head(hidden)
            return out.view(-1, self.horizon, self.target_dim)
