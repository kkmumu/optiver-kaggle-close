import time
import gc
import optuna
import joblib
import os

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from pathlib import Path
from contextlib import contextmanager
from typing import List, Tuple, Optional
from itertools import combinations


Y_MIN, Y_MAX = -30, 30
NAN_VAL = -90000000000
SINGLE_MODEL_TRAIN_SIZE = 0.8
NUM_BOOST_ROUND = 300
EARLY_STOPPING_ROUNDS = 100
RANDOM_STATE = 0
ID_COLS = ["date_id", "stock_id", "seconds_in_bucket"]

DATA_DIR = Path("../optiver")
OUT_DIR = Path("../optiver/working")
MODEL_DIR = Path("../optiver/models")


def create_features(data: pd.DataFrame):
    id_cols = ["date_id", "stock_id", "seconds_in_bucket"]
    px_cols = [
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
        "mid_price",
    ]
    size_cols = [
        "imbalance_size",
        "matched_size",
        "bid_size",
        "ask_size",
        "auc_ask_size",
        "auc_bid_size",
    ]
    other_features = ["side", "stock_weight", "spread", "minute"]
    assert all(col in data.columns for col in (id_cols + px_cols[:6] + size_cols[:4] + ["imbalance_buy_sell_flag", "target"]))

    df = pl.from_pandas(data)
    df = create_price_related_features(df)
    df = create_volume_related_features(df, size_cols)
    df = create_price_volume_imbalance_features(df)
    df = create_rolling_features1(df)
    df = create_cross_sectional_features(df)
    df = create_rolling_features2(df)
    df = create_rolling_features3(df)
    features_list = id_cols + px_cols + size_cols + other_features + [c for c in df.columns if c.startswith("f_")]
    return df.to_pandas(), features_list


def create_features_with_revealed_targets(data: pd.DataFrame, revealed_targets: pd.DataFrame, curr_date_id: int, curr_second: int):
    id_cols = ["date_id", "stock_id", "seconds_in_bucket"]
    px_cols = [
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
        "mid_price",
    ]
    size_cols = [
        "imbalance_size",
        "matched_size",
        "bid_size",
        "ask_size",
        "auc_ask_size",
        "auc_bid_size",
    ]
    other_features = ["side", "stock_weight", "spread", "minute"]
    assert all(col in data.columns for col in (id_cols + px_cols[:6] + size_cols[:4] + ["imbalance_buy_sell_flag"]))
    assert (id_cols + ["target"]) == revealed_targets.columns.to_list()

    df = pl.from_pandas(data)
    df = create_price_related_features(df)
    df = create_volume_related_features(df, size_cols)
    df = create_price_volume_imbalance_features(df)
    df = create_rolling_features1(df)
    df = create_cross_sectional_features(df)
    df = create_rolling_features2(df)
    df = df.join(pl.from_pandas(revealed_targets).sort(id_cols), how="full", on=id_cols, coalesce=True)
    df = create_rolling_features3(df)
    df = df.filter((pl.col("date_id") == curr_date_id) & (pl.col("seconds_in_bucket") == curr_second))
    features_list = id_cols + px_cols + size_cols + other_features + [c for c in df.columns if c.startswith("f_")]
    return df.to_pandas(), features_list


def impute_missing_values(data: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    df = pl.from_pandas(data)
    for feat in selected_features:
        df = df.with_columns(pl.col(feat).fill_null(strategy="forward").over("stock_id", "date_id"))
        df = df.with_columns(pl.col(feat).fill_null(pl.col(feat).median()).over("stock_id", "date_id"))
    return df.to_pandas()


def create_price_related_features(df: pl.DataFrame) -> pl.DataFrame:
    pass


def create_volume_related_features(df: pl.DataFrame, size_cols: List) -> pl.DataFrame:
    pass


def create_price_volume_imbalance_features(df: pl.DataFrame) -> pl.DataFrame:
    pass


def create_rolling_features1(df: pl.DataFrame) -> pl.DataFrame:
    pass


def create_cross_sectional_features(df):
    pass


def create_rolling_features2(df: pl.DataFrame) -> pl.DataFrame:
    pass


def create_rolling_features3(df: pl.DataFrame) -> pl.DataFrame:
    pass


def make_ensemble(X_test: pd.DataFrame, models: List):
    try:
        y_pred = np.nan_to_num(
            np.nanmean(
                np.nan_to_num(
                    [model.predict(X_test).clip(min=Y_MIN, max=Y_MAX) for model in models],
                    copy=True,
                    nan=np.nan,
                    posinf=Y_MAX,
                    neginf=Y_MIN,
                ),
                axis=0,
            ),
            copy=True,
            nan=0,
            posinf=Y_MAX,
            neginf=Y_MIN,
        ).astype(np.float64)
    except:
        return np.zeros(X_test.shape[0], dtype=np.float64)
    return y_pred


class BlockingTimeSeriesSplit:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.folds = []

    def get_n_splits(self):
        return self.folds

    def split(self, X, train_frac, test_frac, gap: int = 0):
        n_samples = len(X)
        train_size = int(n_samples * train_frac)
        test_size = int(n_samples * test_frac)
        k_fold_size = train_size + gap + test_size
        indices = np.arange(n_samples)

        start_step = (n_samples - k_fold_size) // (self.n_splits - 1)
        for i in range(self.n_splits):
            start = i * start_step
            mid = start + train_size
            stop = start + k_fold_size
            self.folds.append((indices[start:mid], indices[mid + gap : stop]))


def make_cv_folds(X, n_folds: int = 5, mode="random"):
    dates_id = X["date_id"].unique()

    if mode == "purged":
        tscv = BlockingTimeSeriesSplit(n_splits=n_folds)
        tscv.split(dates_id, train_frac=SINGLE_MODEL_TRAIN_SIZE, test_frac=SINGLE_MODEL_TRAIN_SIZE / 4, gap=0)
        splits = tscv.get_n_splits()

    elif mode == "random":
        train_size = int(len(dates_id) * SINGLE_MODEL_TRAIN_SIZE)
        test_size = int(len(dates_id) * SINGLE_MODEL_TRAIN_SIZE / 4)
        selected = []
        for i in range(n_folds):
            np.random.seed(RANDOM_STATE + 10 * i)
            selected.append(np.random.choice(dates_id, size=train_size + test_size, replace=False))
        splits = [(samples[:train_size], samples[train_size:]) for samples in selected]

    folds = []
    for i, (train_dates, valid_dates) in enumerate(splits):
        if mode == "purged":
            print(f"folds{i}: train dates={train_dates[0]}~{train_dates[-1]}, valid dates={valid_dates[0]}~{valid_dates[-1]}")
        idx_train = X[X["date_id"].isin(train_dates)].index
        idx_valid = X[X["date_id"].isin(valid_dates)].index
        folds.append((idx_train, idx_valid))
        print(f"folds{i}: train={len(idx_train)}, valid={len(idx_valid)}")

    del train_dates, valid_dates, idx_train, idx_valid, splits
    gc.collect()
    return folds


def train_evaluate(X_train, y_train, X_valid, y_valid, params, pruning_callback=None):
    train_data = xgb.DMatrix(X_train, label=y_train)
    valid_data = xgb.DMatrix(X_valid, label=y_valid)

    model = xgb.train(
        params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(train_data, "Train"), (valid_data, "Valid")],
        callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS), pruning_callback] if pruning_callback is not None else [xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)],
    )
    return model.best_score


def objective(trial, X, y, idx_train, idx_valid):
    params = {
        "eval_metric": "mae",
        "tree_method": "gpu_hist",
        "sampling_method": "gradient_based",
        "lambda": trial.suggest_loguniform("lambda", 7.0, 17.0),
        "alpha": trial.suggest_loguniform("alpha", 7.0, 17.0),
        "eta": trial.suggest_categorical("eta", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "gamma": trial.suggest_categorical("gamma", [18, 19, 20, 21, 22, 23, 24, 25]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bynode": trial.suggest_categorical("colsample_bynode", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "min_child_weight": trial.suggest_int("min_child_weight", 8, 600),
        "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7]),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.6, 0.7, 0.8, 1.0]),
        "n_jobs": 10,
        "random_state": RANDOM_STATE,
    }

    X_train, y_train, X_valid, y_valid = X.iloc[idx_train, :], y.iloc[idx_train], X.iloc[idx_valid, :], y.iloc[idx_valid]
    return train_evaluate(X_train, y_train, X_valid, y_valid, params)


def refit_xgb(best_params, X_train, y_train, selected_features: List[str], window: int = 30, seed: int = 0):
    # attach more weights to recent data
    date_ids = X_train["date_id"].to_numpy()
    date_weights = np.ones_like(date_ids).astype(float)
    date_weights[date_ids >= X_train["date_id"].max() - window] = 1.5

    best_params["seed"] = seed
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train[selected_features],
        y_train,
        sample_weight=date_weights,
    )
    return model


def train_xgb(X_train, y_train, cv_folds, selected_features: List[str], verbose: bool = True, save_study: bool = True, n_trials: int = 100):
    fitted_models = []
    for i, (idx_train, idx_valid) in enumerate(cv_folds):
        with timer(f"Finishing fitting model with data in FOLD{i}"):
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, X_train.loc[:, selected_features], y_train, idx_train, idx_valid), n_trials=n_trials)
            best_params = study.best_trial.params

            if verbose:
                print(f"Fold{i} best params", best_params)
                print(f"Fold{i} best valid score", study.best_trial.value)
            if save_study:
                joblib.dump(study, os.path.join(OUT_DIR, f"xgb_80d_study_fold{i}.pkl"))

            model = refit_xgb(
                best_params,
                X_train.iloc[idx_train.join(idx_valid, how="outer"), :],
                y_train.iloc[idx_train.join(idx_valid, how="outer")],
                selected_features,
            )
            model.save_model(os.path.join(MODEL_DIR, f"xgb_feat_169_80d_fold{i}.model"))
            fitted_models.append(model)
    del model
    gc.collect()
    return fitted_models


@contextmanager
def timer(name: str):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f"[{name}] {elapsed: .3f}sec")


def clean_format(test: pd.DataFrame, revealed_targets: pd.DataFrame, curr_date_id: int):
    if len(revealed_targets) > 2:
        revealed_targets = revealed_targets.drop(columns=["date_id", "revealed_time_id"]).rename(columns={"revealed_date_id": "date_id", "revealed_target": "target"})
        full_index = pd.MultiIndex.from_product([[curr_date_id], list(range(200)), list(range(0, 541, 10))], names=ID_COLS)
        revealed_targets = revealed_targets.set_index(ID_COLS).reindex(full_index).reset_index()
        for col in ID_COLS:
            revealed_targets[col] = pd.to_numeric(revealed_targets[col], errors="coerce").astype(np.int16)
        revealed_targets["target"] = pd.to_numeric(revealed_targets["target"], errors="coerce").astype(np.float16)
    else:
        revealed_targets = pd.DataFrame()

    test["imbalance_buy_sell_flag"] = test["imbalance_buy_sell_flag"].astype(np.int8)
    for col in ["imbalance_size", "reference_price", "matched_size", "far_price", "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap"]:
        test[col] = test[col].astype(np.float32)
    for col in ["stock_id", "date_id", "seconds_in_bucket"]:
        test[col] = test[col].astype(np.int16)
    test = test.drop("currently_scored", axis=1)
    return test, revealed_targets


# from https://www.kaggle.com/code/lblhandsome/optiver-robust-best-single-model
def reduce_mem_usage(df, exclude_columns=[], verbose: bool = True):
    import time

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_time = time.time()
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if col in exclude_columns:
            continue
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print("Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        print("reduce memory use:", round(time.time() - start_time, 1))
    return df


# from https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/442851
weights = [
    0.004,
    0.001,
    0.002,
    0.006,
    0.004,
    0.004,
    0.002,
    0.006,
    0.006,
    0.002,
    0.002,
    0.008,
    0.006,
    0.002,
    0.008,
    0.006,
    0.002,
    0.006,
    0.004,
    0.002,
    0.004,
    0.001,
    0.006,
    0.004,
    0.002,
    0.002,
    0.004,
    0.002,
    0.004,
    0.004,
    0.001,
    0.001,
    0.002,
    0.002,
    0.006,
    0.004,
    0.004,
    0.004,
    0.006,
    0.002,
    0.002,
    0.04,
    0.002,
    0.002,
    0.004,
    0.04,
    0.002,
    0.001,
    0.006,
    0.004,
    0.004,
    0.006,
    0.001,
    0.004,
    0.004,
    0.002,
    0.006,
    0.004,
    0.006,
    0.004,
    0.006,
    0.004,
    0.002,
    0.001,
    0.002,
    0.004,
    0.002,
    0.008,
    0.004,
    0.004,
    0.002,
    0.004,
    0.006,
    0.002,
    0.004,
    0.004,
    0.002,
    0.004,
    0.004,
    0.004,
    0.001,
    0.002,
    0.002,
    0.008,
    0.02,
    0.004,
    0.006,
    0.002,
    0.02,
    0.002,
    0.002,
    0.006,
    0.004,
    0.002,
    0.001,
    0.02,
    0.006,
    0.001,
    0.002,
    0.004,
    0.001,
    0.002,
    0.006,
    0.006,
    0.004,
    0.006,
    0.001,
    0.002,
    0.004,
    0.006,
    0.006,
    0.001,
    0.04,
    0.006,
    0.002,
    0.004,
    0.002,
    0.002,
    0.006,
    0.002,
    0.002,
    0.004,
    0.006,
    0.006,
    0.002,
    0.002,
    0.008,
    0.006,
    0.004,
    0.002,
    0.006,
    0.002,
    0.004,
    0.006,
    0.002,
    0.004,
    0.001,
    0.004,
    0.002,
    0.004,
    0.008,
    0.006,
    0.008,
    0.002,
    0.004,
    0.002,
    0.001,
    0.004,
    0.004,
    0.004,
    0.006,
    0.008,
    0.004,
    0.001,
    0.001,
    0.002,
    0.006,
    0.004,
    0.001,
    0.002,
    0.006,
    0.004,
    0.006,
    0.008,
    0.002,
    0.002,
    0.004,
    0.002,
    0.04,
    0.002,
    0.002,
    0.004,
    0.002,
    0.002,
    0.006,
    0.02,
    0.004,
    0.002,
    0.006,
    0.02,
    0.001,
    0.002,
    0.006,
    0.004,
    0.006,
    0.004,
    0.004,
    0.004,
    0.004,
    0.002,
    0.004,
    0.04,
    0.002,
    0.008,
    0.002,
    0.004,
    0.001,
    0.004,
    0.006,
    0.004,
]

STOCK_WEIGHTS = {int(k): v for k, v in enumerate(weights)}


# from https://www.kaggle.com/code/lognorm/de-anonymizing-stock-id
TICKER_GICS = {
    166: "Consumer Goods",
    121: "Consumer Goods",
    105: "Consumer Goods",
    151: "Technology",
    170: "Healthcare",
    0: "Consumer Goods",
    65: "Utilities",
    109: "Transportation",
    123: "Healthcare",
    198: "Communication Services",
    131: "Healthcare",
    21: "Technology",
    148: "Utilities",
    38: "Communication Services",
    30: "Utilities",
    63: "Consumer Discretionary",
    24: "Technology",
    130: "Consumer Discretionary",
    120: "Communication Services",
    195: "Utilities",
    53: "Consumer Discretionary",
    81: "Healthcare",
    47: "Consumer Discretionary",
    154: "Technology",
    160: "Consumer Goods",
    55: "Industrial Goods",
    37: "Technology",
    90: "Technology",
    186: "Technology",
    187: "Communication Services",
    76: "Consumer Goods",
    117: "Financial",
    134: "Healthcare",
    3: "Industrial Goods",
    165: "Financial",
    97: "Healthcare",
    145: "Financial",
    25: "Services",
    68: "Services",
    52: "Financial",
    112: "Services",
    181: "Services",
    28: "Energy",
    43: "Technology",
    12: "Healthcare",
    149: "Technology",
    144: "Consumer Goods",
    192: "Technology",
    153: "Services",
    175: "Technology",
    189: "Technology",
    116: "Technology",
    35: "Consumer Discretionary",
    46: "Healthcare",
    164: "Consumer Discretionary",
    44: "Financial",
    146: "Industrial Goods",
    125: "Healthcare",
    171: "Financial",
    73: "Healthcare",
    196: "Healthcare",
    9: "Financial",
    199: "Energy",
    193: "Technology",
    106: "Technology",
    122: "Services",
    4: "Services",
    176: "Technology",
    2: "Industrial Goods",
    167: "Services",
    91: "Technology",
    128: "Technology",
    152: "Technology",
    155: "Energy",
    26: "Financial",
    132: "Consumer Discretionary",
    119: "Healthcare",
    27: "Technology",
    84: "Technology",
    23: "Technology",
    110: "Communication Services",
    182: "Services",
    157: "Technology",
    168: "Consumer Discretionary",
    147: "Energy",
    64: "Financial",
    190: "Healthcare",
    19: "Healthcare",
    1: "Consumer Discretionary",
    49: "Financial",
    194: "Consumer Discretionary",
    140: "Technology",
    133: "Healthcare",
    177: "Healthcare",
    32: "Technology",
    22: "Technology",
    77: "Healthcare",
    104: "Consumer Goods",
    107: "Healthcare",
    59: "Technology",
    72: "Technology",
    158: "Healthcare",
    169: "Technology",
    94: "Technology",
    66: "Communication Services",
    126: "Technology",
    139: "Consumer Discretionary",
    159: "Basic Materials",
    137: "Technology",
    78: "Financial",
    114: "Financial",
    141: "Technology",
    15: "Technology",
    60: "Technology",
    183: "Technology",
    10: "Financial",
    135: "Consumer Discretionary",
    197: "Technology",
    99: "Communication Services",
    56: "Consumer Discretionary",
    13: "Technology",
    62: "Healthcare",
    80: "Healthcare",
    67: "Healthcare",
    178: "Technology",
    39: "Consumer Discretionary",
    173: "Consumer Discretionary",
    184: "Technology",
    162: "Technology",
    16: "Technology",
    89: "Technology",
    45: "Technology",
    124: "Consumer Discretionary",
    42: "Services",
    115: "Healthcare",
    50: "Technology",
    98: "Services",
    103: "Technology",
    40: "Consumer Discretionary",
    108: "Technology",
    179: "Technology",
    6: "Consumer Discretionary",
    100: "Technology",
    48: "Technology",
    150: "Technology",
    74: "Healthcare",
    113: "Consumer Discretionary",
    163: "Consumer Discretionary",
    111: "Technology",
    36: "Consumer Discretionary",
    85: "Services",
    79: "Communication Services",
    83: "Technology",
    75: "Technology",
    57: "Healthcare",
    180: "Healthcare",
    93: "Financial",
    87: "Technology",
    51: "Technology",
    33: "Consumer Goods",
    58: "Industrial Goods",
    7: "Technology",
    172: "Technology",
    61: "Technology",
    185: "Services",
    102: "Technology",
    188: "Technology",
    17: "Technology",
    88: "Communication Services",
    95: "Technology",
    14: "Technology",
    54: "Technology",
    18: "Technology",
    129: "Technology",
    136: "Technology",
    191: "Consumer Goods",
    20: "Consumer Discretionary",
    161: "Consumer Discretionary",
    5: "Technology",
    71: "Technology",
    34: "Healthcare",
    142: "Technology",
    92: "Technology",
    156: "Consumer Goods",
    41: "Technology",
    69: "Financial",
    138: "Technology",
    174: "Technology",
    11: "Financial",
    70: "Consumer Goods",
    96: "Technology",
    118: "Financial",
    29: "Consumer Discretionary",
    143: "Healthcare",
    86: "Consumer Discretionary",
    82: "Technology",
    127: "Technology",
    101: "Financial",
    8: "Services",
    31: "Healthcare",
}
