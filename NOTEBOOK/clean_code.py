# %% imports
# imports
import random, warnings, json

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    pairwise_distances,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# %% variáveis
# variáveis
RANDOM_STATE = 42
N_SPLITS = 5
INNER_OOF_SPLITS = 5

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_DIR = Path.cwd().resolve().parent / "DATA"

GOLD_EVENTS_PATH = DATA_DIR / "gold_all_events_real.csv"
FEATURE_LIST_PATH = DATA_DIR / "gold1_column_lists.json"

TARGET_COL = "soh_top10_ref"
SOH_MIN, SOH_MAX, SOH_BIN_STEP = 0.920, 1.005, 0.005
SOH_BIN_EDGES = np.round(np.arange(SOH_MIN, SOH_MAX + SOH_BIN_STEP, SOH_BIN_STEP), 6)
if SOH_BIN_EDGES[-1] < SOH_MAX:
    SOH_BIN_EDGES = np.append(SOH_BIN_EDGES, SOH_MAX)

MILEAGE_SMOTER_MIN = 30_000
MAX_MILEAGE_DIFF = 15_000
K_NEIGHBORS_SMOTER = 5
TARGET_FIRST_BIN = 300
TARGET_OTHER_BINS = 500

ROUTER_BOUNDARY_A_B = 0.9425
ROUTER_BOUNDARY_B_C = 0.9725

FEATURES_REGRESSION_20 = [
    "min_single_volt_slope", "min_single_volt_iqr", "volt_slope", "temp_mean_mean",
    "cell_imbalance_SOC_40_60_std", "cell_imbalance_SOC_40_60_p95", "cell_imbalance_SOC_80_100_std",
    "power_proxy_SOC_60_80_std", "temp_mean_SOC_40_60_mean", "min_temp_median",
    "min_temp_p05", "current_abs_p05", "cell_imbalance_SOC_60_80_std", "current_p05",
    "current_abs_SOC_60_80_mean", "current_abs_SOC_40_60_mean", "cell_imbalance_SOC_60_80_max",
    "power_proxy_median", "power_proxy_p05", "cell_imbalance_p75",
]

FEATURES_GLOBAL = list(dict.fromkeys(FEATURES_REGRESSION_20 + ["mileage"]))
FEATURES_EXPERTS = FEATURES_GLOBAL.copy()
FEATURES_GATE = FEATURES_GLOBAL.copy()

EXPERT_CLASS_NAMES = {
    0: "A_XGBoost_Baixo",
    1: "B_ExtraTrees_Intermediario",
    2: "C_ExtraTrees_Alto",
}

EXPERT_CONFIG = {
    0: {"name": "A_XGBoost_Baixo", "model": "XGBRegressor", "soh_min": 0.920, "soh_max": 0.945},
    1: {"name": "B_ExtraTrees_Intermediario", "model": "ExtraTreesRegressor", "soh_min": 0.940, "soh_max": 0.975},
    2: {"name": "C_ExtraTrees_Alto", "model": "ExtraTreesRegressor", "soh_min": 0.970, "soh_max": 1.005},
}

GLOBAL_MODEL_PARAMS = {
    "n_estimators": 1200, "max_depth": None, "min_samples_split": 2,
    "min_samples_leaf": 1, "max_features": 1.0, "bootstrap": False,
    "random_state": RANDOM_STATE, "n_jobs": -1,
}

EXPERT_A_XGB_PARAMS = {
    "n_estimators": 1200, "learning_rate": 0.02, "max_depth": 6,
    "min_child_weight": 2, "subsample": 0.90, "colsample_bytree": 0.90,
    "reg_alpha": 0.05, "reg_lambda": 1.5, "objective": "reg:squarederror",
    "eval_metric": "rmse", "tree_method": "hist", "random_state": RANDOM_STATE, "n_jobs": -1,
}

EXPERT_BC_ET_PARAMS = {
    "n_estimators": 1000, "max_depth": None, "min_samples_split": 2,
    "min_samples_leaf": 1, "max_features": 1.0, "bootstrap": False,
    "random_state": RANDOM_STATE, "n_jobs": -1,
}

GATE_XGB_PARAMS = {
    "n_estimators": 1200, "learning_rate": 0.02, "max_depth": 6,
    "min_child_weight": 2, "subsample": 0.90, "colsample_bytree": 0.90,
    "reg_alpha": 0.05, "reg_lambda": 1.5, "objective": "multi:softprob",
    "num_class": 3, "eval_metric": "mlogloss", "tree_method": "hist",
    "random_state": RANDOM_STATE, "n_jobs": -1,
}

CV_FOLD_DATA = {}
FINAL_MODELS = {}
FINAL_PREDICTIONS = {}
# %% funções
# funções
def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}


def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision_Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        metrics["Log_Loss"] = log_loss(y_true, y_prob)
    return metrics


def apply_smoter_custom(df_train, feature_cols, target_col="soh_top10_ref"):
    df_resampled = df_train.copy()
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(df_train[feature_cols])

    bins = sorted(df_train["soh_bin"].unique())
    for b in bins:
        target_n = TARGET_FIRST_BIN if b == bins[0] else TARGET_OTHER_BINS
        bin_indices = df_train.index[df_train["soh_bin"] == b].tolist()
        current_n = len(bin_indices)

        if current_n == 0 or current_n >= target_n:
            continue

        n_samples_to_gen = target_n - current_n
        bin_pos = df_train.index.get_indexer(bin_indices)

        synthetic_rows = []
        for _ in range(n_samples_to_gen):
            idx = random.choice(bin_pos)
            sample_feats = scaled_features[idx].reshape(1, -1)
            sample_mileage = df_train.iloc[idx]["mileage"]

            distances = pairwise_distances(sample_feats, scaled_features).flatten()
            mileage_diffs = np.abs(df_train["mileage"].values - sample_mileage)

            valid_mask = (mileage_diffs <= MAX_MILEAGE_DIFF) & (distances > 0)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                valid_indices = np.where(distances > 0)[0]

            k = min(K_NEIGHBORS_SMOTER, len(valid_indices))
            if k == 0:
                continue

            nearest_k = valid_indices[np.argsort(distances[valid_indices])[:k]]
            neighbor_idx = random.choice(nearest_k)

            diff = scaled_features[neighbor_idx] - scaled_features[idx]
            synth_scaled = scaled_features[idx] + random.random() * diff
            synth_unscaled = scaler.inverse_transform(synth_scaled.reshape(1, -1))[0]

            row_dict = dict(zip(feature_cols, synth_unscaled))
            if "mileage" in df_train.columns and "mileage" not in feature_cols:
                row_dict["mileage"] = df_train.iloc[idx]["mileage"]

            row_dict[target_col] = df_train.iloc[idx][target_col] + random.uniform(-0.001, 0.001)
            row_dict["soh_bin"] = b
            if "expert_class" in df_train.columns:
                row_dict["expert_class"] = df_train.iloc[idx]["expert_class"]

            synthetic_rows.append(row_dict)

        if synthetic_rows:
            df_resampled = pd.concat([df_resampled, pd.DataFrame(synthetic_rows)], ignore_index=True)

    return df_resampled
# %% leiura
# leitura
with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as file:
    FEATURE_COLS = list(json.load(file)["MODEL_FEATURES_GOLD1_FINAL"])

df = pd.read_csv(GOLD_EVENTS_PATH)
# %% preparação
# preparação
df[TARGET_COL] = df[TARGET_COL].astype(float)
df["mileage"] = df["mileage"].astype(float)

FEATURES_GATE = [col for col in FEATURES_GATE if col in df.columns]
FEATURES_EXPERTS = [col for col in FEATURES_EXPERTS if col in df.columns]

df["soh_bin"] = pd.cut(
    df[TARGET_COL],
    bins=SOH_BIN_EDGES,
    include_lowest=True,
    labels=False,
)

df["expert_class"] = 1
df.loc[df[TARGET_COL] < ROUTER_BOUNDARY_A_B, "expert_class"] = 0
df.loc[df[TARGET_COL] >= ROUTER_BOUNDARY_B_C, "expert_class"] = 2

df = df.dropna(subset=[TARGET_COL, "expert_class", "soh_bin"]).reset_index(drop=True)
df["soh_bin"] = df["soh_bin"].astype(int)
df["expert_class"] = df["expert_class"].astype(int)
# %% validação cruzada
# validação cruzada
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_gate_probs = np.zeros((len(df), 3))
oof_predictions = np.zeros(len(df))
oof_global_preds = np.zeros(len(df))
oof_experts_preds = {c: np.zeros(len(df)) for c in EXPERT_CONFIG.keys()}

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["soh_bin"])):
    df_train, df_val = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    print("Treinando Gate...")
    gate_model = XGBClassifier(**GATE_XGB_PARAMS)
    gate_model.fit(df_train[FEATURES_GATE], df_train["expert_class"])
    val_gate_probs = gate_model.predict_proba(df_val[FEATURES_GATE])
    oof_gate_probs[val_idx] = val_gate_probs

    df_train_smoted = apply_smoter_custom(df_train, FEATURES_EXPERTS, target_col=TARGET_COL)

    print("Treinando Modelo Global...")
    global_model = ExtraTreesRegressor(**GLOBAL_MODEL_PARAMS)
    global_model.fit(df_train_smoted[FEATURES_GLOBAL], df_train_smoted[TARGET_COL])
    preds_val_global = global_model.predict(df_val[FEATURES_GLOBAL])
    oof_global_preds[val_idx] = preds_val_global

    experts = {}
    preds_val_experts_matrix = []
    print("Treinando Modelos Especialistas...")
    for class_id, cfg in EXPERT_CONFIG.items():
        subset = df_train_smoted[df_train_smoted["expert_class"] == class_id]
        if len(subset) == 0:
            subset = df_train_smoted

        exp_model = (
            XGBRegressor(**EXPERT_A_XGB_PARAMS)
            if cfg["model"] == "XGBRegressor"
            else ExtraTreesRegressor(**EXPERT_BC_ET_PARAMS)
        )

        exp_model.fit(subset[FEATURES_EXPERTS], subset[TARGET_COL])
        experts[class_id] = exp_model

        preds_exp = exp_model.predict(df_val[FEATURES_EXPERTS])
        oof_experts_preds[class_id][val_idx] = preds_exp
        preds_val_experts_matrix.append(preds_exp)

    preds_val_experts_matrix = np.column_stack(preds_val_experts_matrix)

    preds_val_routed = np.sum(preds_val_experts_matrix * val_gate_probs, axis=1)
    oof_predictions[val_idx] = preds_val_routed

    fold_metrics = {
        "gate": calculate_classification_metrics(df_val["expert_class"], np.argmax(val_gate_probs, axis=1), val_gate_probs),
        "global": calculate_regression_metrics(df_val[TARGET_COL], preds_val_global),
        "routed_combined": calculate_regression_metrics(df_val[TARGET_COL], preds_val_routed),
        "experts": {
            class_id: calculate_regression_metrics(df_val[TARGET_COL], preds_val_experts_matrix[:, class_id])
            for class_id in EXPERT_CONFIG.keys()
        }
    }

    CV_FOLD_DATA[fold] = {
        "gate_model": gate_model,
        "global_model": global_model,
        "experts": experts,
        "val_idx": val_idx,
        "metrics": fold_metrics
    }

# %% avaliação
# avaliação
FINAL_PREDICTIONS["oof_preds"] = oof_predictions
FINAL_PREDICTIONS["oof_gate_probs"] = oof_gate_probs
FINAL_PREDICTIONS["oof_global_preds"] = oof_global_preds
FINAL_PREDICTIONS["oof_experts_preds"] = oof_experts_preds

metrics_summary = {
    "gate_oof": calculate_classification_metrics(df["expert_class"], np.argmax(oof_gate_probs, axis=1), oof_gate_probs),
    "global_oof": calculate_regression_metrics(df[TARGET_COL], oof_global_preds),
    "routed_combined_oof": calculate_regression_metrics(df[TARGET_COL], oof_predictions),
    "experts_oof": {
        class_id: calculate_regression_metrics(df[TARGET_COL], oof_experts_preds[class_id])
        for class_id in EXPERT_CONFIG.keys()
    }
}