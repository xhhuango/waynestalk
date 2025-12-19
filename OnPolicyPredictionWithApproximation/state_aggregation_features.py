import numpy as np


class StateAggregationFeatures:
    def __init__(self, n_rows=4, n_cols=12, col_bin=2):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.col_bin = col_bin
        self.n_col_bins = n_cols // col_bin
        self.d = n_rows * self.n_col_bins

    def __call__(self, s: int) -> np.ndarray:
        row, col = divmod(s, self.n_cols)
        col_group = col // self.col_bin
        idx = row * self.n_col_bins + col_group
        x = np.zeros(self.d, dtype=float)
        x[idx] = 1.0
        return x
