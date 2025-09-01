import os
import random
import math
import glob
import numpy as np
import torch
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Optional, Tuple, Iterable


class ParquetVectorIterable(IterableDataset):
    """
    Stream parquet rows as (oid, vector[tensor]) pairs.
    Intended for very large datasets.

    Parameters
    ----------
    data_dirs : list[str]
        Directories that contain parquet shards.
    column_vector : str
        Column name of the vector (default: "parsed_vector").
    column_id : str
        Column name of the id (default: "md5_oaid").
    expected_dim : int
        Expected vector dim (default: 5120). Rows with mismatched dim are skipped.
    shuffle_files : bool
        Shuffle file order once per epoch.
    shuffle_rows : bool
        Shuffle row order within each file (uses pandas; may add overhead).
    seed : int
        RNG seed for shuffling.
    normalize : None | "l2" | "layernorm"
        Optional per-row normalization.
    dtype : str
        "float32" or "float16".
    """
    def __init__(
        self,
        data_dirs: List[str],
        column_vector: str = "parsed_vector",
        column_id: str = "md5_oaid",
        expected_dim: int = 5120,
        shuffle_files: bool = True,
        shuffle_rows: bool = False,
        seed: int = 42,
        normalize: Optional[str] = None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.data_dirs = list(data_dirs)
        self.column_vector = column_vector
        self.column_id = column_id
        self.expected_dim = expected_dim
        self.shuffle_files = shuffle_files
        self.shuffle_rows = shuffle_rows
        self.seed = seed
        self.normalize = normalize
        self.dtype = dtype

        self.files = []
        for d in self.data_dirs:
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith(".parquet"):
                    self.files.append(os.path.join(d, fn))
        if self.shuffle_files:
            random.Random(seed).shuffle(self.files)

        print(f"找到 {len(self.files)} 个 parquet 文件")

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.normalize is None:
            return x
        if self.normalize == "l2":
            denom = np.linalg.norm(x, ord=2) + 1e-8
            return x / denom
        if self.normalize == "layernorm":
            mu = x.mean()
            sigma = x.std() + 1e-5
            return (x - mu) / sigma
        return x

    def _iter_file(self, path) -> Iterable[Tuple[str, np.ndarray]]:
        table = pq.read_table(path, columns=[self.column_id, self.column_vector])
        df = table.to_pandas()

        idxs = list(range(len(df)))
        if self.shuffle_rows:
            random.shuffle(idxs)

        for i in idxs:
            oid = df.iloc[i][self.column_id]
            vec = np.asarray(df.iloc[i][self.column_vector], dtype=self.dtype)
            if vec.shape[0] != self.expected_dim:
                continue
            vec = np.nan_to_num(vec, nan=0.0, posinf=10.0, neginf=-10.0)
            vec = self._norm(vec)
            yield oid, vec

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            files = self.files
        else:
            files = self.files[worker.id::worker.num_workers]

        for fp in files:
            for oid, vec in self._iter_file(fp):
                yield oid, torch.from_numpy(vec)

