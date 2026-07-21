# %% imports
# imports
from pathlib import Path
import json
import random
import gc
import time
import warnings

import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.ensemble import ExtraTreesRegressor


from sklearn.metrics import pairwise_distances

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)

warnings.filterwarnings("ignore")
# %% variáveis
# variáveis
RANDOM_STATE = 42
N_SPLITS = 5

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

TARGET_COL = "soh_top10_ref"

SOH_MIN = 0.920
SOH_MAX = 1.005

SOH_BIN_SIZE = 0.005

MILEAGE_SMOTER_MIN = 30_000
MAX_MILEAGE_DISTANCE = 15_000
N_NEIGHBORS_SMOTER = 5

TARGET_FIRST_BIN = 300
TARGET_OTHER_BINS = 500

CURRENT_DIR = Path.cwd().resolve()

if CURRENT_DIR.name.upper() == "NOTEBOOK":
    PROJECT_ROOT = CURRENT_DIR.parent
elif (CURRENT_DIR / "NOTEBOOK").exists():
    PROJECT_ROOT = CURRENT_DIR
else:
    raise FileNotFoundError(
        "Não foi possível localizar a raiz do projeto.\n"
        "Abra o notebook dentro da pasta NOTEBOOK ou execute "
        "o Jupyter a partir da raiz SOH_EV_Real_Data."
    )

DATA_DIR = PROJECT_ROOT / "DATA"
CONFIG_DIR = PROJECT_ROOT / "CONFIG"
OUTPUT_DIR = PROJECT_ROOT / "OUTPUT"

AUDIT_DIR = OUTPUT_DIR / "audit"
FOLDS_DIR = OUTPUT_DIR / "folds"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = OUTPUT_DIR / "models"

for directory in [
    OUTPUT_DIR,
    AUDIT_DIR,
    FOLDS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

GOLD_EVENTS_PATH = (
    DATA_DIR
    / "gold_all_events_real.csv"
)

FEATURE_LIST_PATH = (
    DATA_DIR
    / "gold1_column_lists.json"
)


TARGET_COL = "soh_top10_ref"

SOH_MIN = 0.920
SOH_MAX = 1.005
SOH_BIN_STEP = 0.005


SOH_BIN_EDGES = np.round(
    np.arange(
        SOH_MIN,
        SOH_MAX + SOH_BIN_STEP,
        SOH_BIN_STEP
    ),
    6
)

if SOH_BIN_EDGES[-1] < SOH_MAX:
    SOH_BIN_EDGES = np.append(
        SOH_BIN_EDGES,
        SOH_MAX
    )

CLEAN_GOLD_PATH = (
    DATA_DIR
    / "gold_all_events_real_clean_092_1005.csv"
)


FEATURES_REGRESSION_20 = [
    "min_single_volt_slope",
    "min_single_volt_iqr",
    "volt_slope",
    "temp_mean_mean",
    "cell_imbalance_SOC_40_60_std",
    "cell_imbalance_SOC_40_60_p95",
    "cell_imbalance_SOC_80_100_std",
    "power_proxy_SOC_60_80_std",
    "temp_mean_SOC_40_60_mean",
    "min_temp_median",
    "min_temp_p05",
    "current_abs_p05",
    "cell_imbalance_SOC_60_80_std",
    "current_p05",
    "current_abs_SOC_60_80_mean",
    "current_abs_SOC_40_60_mean",
    "cell_imbalance_SOC_60_80_max",
    "power_proxy_median",
    "power_proxy_p05",
    "cell_imbalance_p75",
]


FEATURES_GLOBAL = list(
    dict.fromkeys(
        FEATURES_REGRESSION_20
        + ["mileage"]
    )
)

GATE_FEATURE_AUDIT_PATH = (
    CONFIG_DIR
    / "gate_80_features_audit.xlsx"
)

gate_features_audit = pd.read_excel(
    GATE_FEATURE_AUDIT_PATH
)


gate_features_audit["feature"] = (
    gate_features_audit["feature"]
    .astype(str)
    .str.strip()
)

gate_features_audit["status"] = (
    gate_features_audit["status"]
    .astype(str)
    .str.strip()
    .str.upper()
)
FEATURES_GATE_80_ORIGINAL = (
    gate_features_audit["feature"]
    .tolist()
)

FEATURES_GATE_REMOVED = (
    gate_features_audit.loc[
        gate_features_audit["status"].eq("REMOVER"),
        "feature",
    ]
    .tolist()
)

FEATURES_GATE_75 = (
    gate_features_audit.loc[
        gate_features_audit["status"].eq("MANTER"),
        "feature",
    ]
    .tolist()
)


FEATURES_EXPERTS = (
    FEATURES_GLOBAL.copy()
)


FEATURES_GATE = list(
    dict.fromkeys(
        FEATURES_GATE_75
        + ["mileage"]
    )
)


FEATURES_REGRESSION_20 = [
    "min_single_volt_slope",
    "min_single_volt_iqr",
    "volt_slope",
    "temp_mean_mean",
    "cell_imbalance_SOC_40_60_std",
    "cell_imbalance_SOC_40_60_p95",
    "cell_imbalance_SOC_80_100_std",
    "power_proxy_SOC_60_80_std",
    "temp_mean_SOC_40_60_mean",
    "min_temp_median",
    "min_temp_p05",
    "current_abs_p05",
    "cell_imbalance_SOC_60_80_std",
    "current_p05",
    "current_abs_SOC_60_80_mean",
    "current_abs_SOC_40_60_mean",
    "cell_imbalance_SOC_60_80_max",
    "power_proxy_median",
    "power_proxy_p05",
    "cell_imbalance_p75",
]

with open(
    FEATURE_LIST_PATH,
    "r",
    encoding="utf-8"
) as file:
    gold_column_lists = json.load(file)

FEATURE_COLS = list(
    gold_column_lists[
        "MODEL_FEATURES_GOLD1_FINAL"
    ]
)


MILEAGE_SMOTER_MIN = 30_000
MAX_MILEAGE_DIFF = 15_000
K_NEIGHBORS_SMOTER = 5

TARGET_FIRST_BIN = 300
TARGET_OTHER_BINS = 500

SMOTER_RANDOM_STATE = RANDOM_STATE


CV_FOLD_DATA = {}


MODEL_RANDOM_STATE = RANDOM_STATE

INNER_OOF_SPLITS = 5

ROUTER_BOUNDARY_A_B = 0.9425
ROUTER_BOUNDARY_B_C = 0.9725

EXPERT_CLASS_NAMES = {
    0: "A_XGBoost_Baixo",
    1: "B_ExtraTrees_Intermediario",
    2: "C_ExtraTrees_Alto",
}


GLOBAL_MODEL_PARAMS = {
    "n_estimators": 1200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": MODEL_RANDOM_STATE,
    "n_jobs": -1,
}

EXPERT_A_XGB_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "max_depth": 6,
    "min_child_weight": 2,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": MODEL_RANDOM_STATE,
    "n_jobs": -1,
}

EXPERT_BC_ET_PARAMS = {
    "n_estimators": 1000,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": MODEL_RANDOM_STATE,
    "n_jobs": -1,
}

GATE_XGB_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "max_depth": 6,
    "min_child_weight": 2,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": MODEL_RANDOM_STATE,
    "n_jobs": -1,
}


EXPERT_CONFIG = {
    0: {
        "name": "A_XGBoost_Baixo",
        "model": "XGBRegressor",
        "soh_min": 0.920,
        "soh_max": 0.945,
    },
    1: {
        "name": "B_ExtraTrees_Intermediario",
        "model": "ExtraTreesRegressor",
        "soh_min": 0.940,
        "soh_max": 0.975,
    },
    2: {
        "name": "C_ExtraTrees_Alto",
        "model": "ExtraTreesRegressor",
        "soh_min": 0.970,
        "soh_max": 1.005,
    },
}

REUSE_GATE_OOF = True

GATE_OOF_DIR = OUTPUT_DIR / "gate_oof"

GATE_OOF_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REQUIRED_GATE_OOF_COLUMNS = list(
    dict.fromkeys(
        [
            "event_id",
            "sample_origin",
            "inner_fold",
            TARGET_COL,
            "best_expert_class",
            "best_expert",
            "pred_oof_expert_A",
            "pred_oof_expert_B",
            "pred_oof_expert_C",
            "abs_error_oof_expert_A",
            "abs_error_oof_expert_B",
            "abs_error_oof_expert_C",
            "pred_oof_oracle",
            "expert_margin",
            "oracle_confidence",
        ]
        + FEATURES_GATE
    )
)


GATE_OOF_DATA = {}

FINAL_MODELS = {}

FINAL_PREDICTIONS = {}

ARCHITECTURE_PREDICTION_COLUMNS = {
    "ExtraTrees_Global_Mileage":
        "pred_global",

    "MoE_Gate_XGBoost_Experts_Mileage":
        "pred_moe_gate",

    "MoE_Router_ExtraTrees_Global":
        "pred_moe_global_router",

    "Oracle_Experts_Mileage":
        "pred_oracle",
}

FINAL_PREDICTIONS = {}
# %% funções
# funções

def definir_alvo_smoter(intervalo):
    """
    Define o número desejado de eventos no treino final.

    Regras:
    - primeiro bin de SOH: alvo de 300 eventos;
    - demais bins até 1.005: alvo de 500 eventos;
    - bins que já possuem quantidade maior ou igual ao alvo
      permanecem inalterados.
    """

    left = float(intervalo.left)
    right = float(intervalo.right)

    tolerance = 1e-8

    # Primeiro bin criado com include_lowest=True pode aparecer
    # com limite esquerdo ligeiramente inferior a 0.920.
    if (
        right <= 0.925 + tolerance
        and right > 0.920
    ):
        return TARGET_FIRST_BIN

    if (
        right > 0.925 + tolerance
        and right <= SOH_MAX + tolerance
    ):
        return TARGET_OTHER_BINS

    return None


def aplicar_smoter_fold(
    train_real_fold,
    fold_id,
    feature_cols=FEATURE_COLS,
    random_state=SMOTER_RANDOM_STATE,
):
    """
    Aplica o SMOTER somente ao treino real de um fold.

    Retorna
    -------
    train_balanced_fold : DataFrame
        Eventos reais e sintéticos do treino.

    synthetic_fold : DataFrame
        Apenas os eventos sintéticos.

    balance_plan_fold : DataFrame
        Plano de balanceamento por bin.

    neighbor_audit_fold : DataFrame
        Auditoria da disponibilidade de vizinhos.

    generation_audit_fold : DataFrame
        Identificação dos pais e parâmetros de interpolação.
    """

    rng = np.random.default_rng(
        random_state + int(fold_id)
    )

    train_real_fold = (
        train_real_fold
        .copy()
        .reset_index(drop=True)
    )

    train_real_fold["sample_origin"] = "real"
    train_real_fold["split"] = "train_real"
    train_real_fold["source_fold"] = int(fold_id)

    eligible = (
        train_real_fold.loc[
            train_real_fold["mileage"]
            >= MILEAGE_SMOTER_MIN
        ]
        .copy()
        .reset_index()
        .rename(
            columns={
                "index": "train_real_index"
            }
        )
    )

    X_eligible = (
        eligible[feature_cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    feature_medians = X_eligible.median()

    X_eligible_imputed = (
        X_eligible.fillna(
            feature_medians
        )
    )

    remaining_invalid = int(
        X_eligible_imputed
        .isna()
        .sum()
        .sum()
    )

    feature_nunique = (
        X_eligible_imputed
        .nunique(dropna=False)
    )

    constant_features = (
        feature_nunique.loc[
            feature_nunique <= 1
        ]
        .index
        .tolist()
    )

    smoter_distance_features = [
        feature
        for feature in feature_cols
        if feature not in constant_features
    ]

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(
        X_eligible_imputed[
            smoter_distance_features
        ]
    )

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=smoter_distance_features,
        index=eligible.index,
    )

    X_original = (
        X_eligible_imputed[
            feature_cols
        ]
        .reset_index(drop=True)
    )

    neighbor_map = {}
    neighbor_audit_rows = []

    for soh_bin_value, bin_df in eligible.groupby(
        "soh_bin",
        observed=True,
    ):

        bin_indices = (
            bin_df.index.to_numpy()
        )

        n_bin = len(bin_indices)

        if n_bin < 2:

            for local_idx in bin_indices:

                neighbor_map[int(local_idx)] = []

                neighbor_audit_rows.append(
                    {
                        "fold": int(fold_id),
                        "local_index": int(local_idx),
                        "event_id": eligible.loc[
                            local_idx,
                            "event_id",
                        ],
                        "soh_bin": str(
                            soh_bin_value
                        ),
                        "mileage": float(
                            eligible.loc[
                                local_idx,
                                "mileage",
                            ]
                        ),
                        "n_candidates_mileage": 0,
                        "n_neighbors_available": 0,
                    }
                )

            continue

        X_bin = (
            X_scaled_df.loc[
                bin_indices
            ]
            .to_numpy()
        )

        distance_matrix = pairwise_distances(
            X_bin,
            metric="euclidean",
        )

        mileage_bin = (
            bin_df["mileage"]
            .to_numpy(dtype=float)
        )

        mileage_diff_matrix = np.abs(
            mileage_bin[:, None]
            - mileage_bin[None, :]
        )

        allowed_mask = (
            mileage_diff_matrix
            <= MAX_MILEAGE_DIFF
        )

        np.fill_diagonal(
            allowed_mask,
            False,
        )

        constrained_distances = (
            distance_matrix.copy()
        )

        constrained_distances[
            ~allowed_mask
        ] = np.inf

        for position, global_idx in enumerate(
            bin_indices
        ):

            valid_positions = np.where(
                np.isfinite(
                    constrained_distances[
                        position
                    ]
                )
            )[0]

            if len(valid_positions) == 0:

                selected_neighbors = []

            else:

                ordered_positions = (
                    valid_positions[
                        np.argsort(
                            constrained_distances[
                                position,
                                valid_positions,
                            ]
                        )
                    ]
                )

                selected_positions = (
                    ordered_positions[
                        :K_NEIGHBORS_SMOTER
                    ]
                )

                selected_neighbors = (
                    bin_indices[
                        selected_positions
                    ]
                    .astype(int)
                    .tolist()
                )

            neighbor_map[
                int(global_idx)
            ] = selected_neighbors

            neighbor_audit_rows.append(
                {
                    "fold": int(fold_id),
                    "local_index": int(
                        global_idx
                    ),
                    "event_id": eligible.loc[
                        global_idx,
                        "event_id",
                    ],
                    "soh_bin": str(
                        soh_bin_value
                    ),
                    "mileage": float(
                        eligible.loc[
                            global_idx,
                            "mileage",
                        ]
                    ),
                    "n_candidates_mileage": int(
                        len(valid_positions)
                    ),
                    "n_neighbors_available": int(
                        len(selected_neighbors)
                    ),
                }
            )

    neighbor_audit_fold = pd.DataFrame(
        neighbor_audit_rows
    )

    train_bin_counts = (
        train_real_fold
        .groupby(
            "soh_bin",
            observed=True,
        )
        .size()
        .rename("n_real_train")
    )

    balance_plan_rows = []

    valid_seeds_by_bin = {}

    for soh_bin_value, n_real in (
        train_bin_counts.items()
    ):

        target_final = definir_alvo_smoter(
            soh_bin_value
        )

        eligible_indices = (
            eligible.index[
                eligible["soh_bin"]
                == soh_bin_value
            ]
            .tolist()
        )

        valid_seed_indices = [
            int(idx)
            for idx in eligible_indices
            if len(
                neighbor_map.get(
                    int(idx),
                    [],
                )
            ) > 0
        ]

        valid_seeds_by_bin[
            soh_bin_value
        ] = valid_seed_indices

        if target_final is None:

            action = "manter"
            n_requested = 0

        elif int(n_real) >= int(target_final):

            action = "manter_acima_do_alvo"
            n_requested = 0

        else:

            action = "gerar_smoter"
            n_requested = int(
                target_final
                - n_real
            )

        balance_plan_rows.append(
            {
                "fold": int(fold_id),
                "soh_bin": str(
                    soh_bin_value
                ),
                "n_real_train": int(n_real),
                "target_final": (
                    int(target_final)
                    if target_final is not None
                    else np.nan
                ),
                "n_synthetic_requested": int(
                    n_requested
                ),
                "n_eligible_mileage": int(
                    len(eligible_indices)
                ),
                "n_valid_seeds": int(
                    len(valid_seed_indices)
                ),
                "action": action,
            }
        )

    balance_plan_fold = pd.DataFrame(
        balance_plan_rows
    )

    synthetic_rows = []
    generation_audit_rows = []

    synthetic_counter = 0

    for _, plan_row in (
        balance_plan_fold.iterrows()
    ):

        n_to_generate = int(
            plan_row[
                "n_synthetic_requested"
            ]
        )

        if n_to_generate <= 0:
            continue

        matching_bins = [
            interval
            for interval in train_bin_counts.index
            if str(interval)
            == plan_row["soh_bin"]
        ]

        if len(matching_bins) != 1:
            raise ValueError(
                f"Fold {fold_id}: não foi possível "
                f"recuperar o bin {plan_row['soh_bin']}."
            )

        soh_bin_value = matching_bins[0]

        valid_seeds = (
            valid_seeds_by_bin[
                soh_bin_value
            ]
        )

        for _ in range(n_to_generate):

            seed_idx = int(
                rng.choice(
                    valid_seeds
                )
            )

            neighbor_idx = int(
                rng.choice(
                    neighbor_map[
                        seed_idx
                    ]
                )
            )

            interpolation_factor = float(
                rng.random()
            )

            parent_a = eligible.loc[
                seed_idx
            ]

            parent_b = eligible.loc[
                neighbor_idx
            ]

            synthetic_features = (
                X_original.loc[
                    seed_idx
                ]
                +
                interpolation_factor
                * (
                    X_original.loc[
                        neighbor_idx
                    ]
                    -
                    X_original.loc[
                        seed_idx
                    ]
                )
            )

            synthetic_soh = float(
                parent_a[TARGET_COL]
                +
                interpolation_factor
                * (
                    parent_b[TARGET_COL]
                    -
                    parent_a[TARGET_COL]
                )
            )

            synthetic_mileage = float(
                parent_a["mileage"]
                +
                interpolation_factor
                * (
                    parent_b["mileage"]
                    -
                    parent_a["mileage"]
                )
            )

            synthetic_counter += 1

            synthetic_event_id = (
                f"smoter_f{fold_id}_"
                f"{synthetic_counter:06d}"
            )

            synthetic_row = {
                column: np.nan
                for column
                in train_real_fold.columns
            }

            for column in train_real_fold.columns:

                if column in parent_a.index:
                    synthetic_row[column] = (
                        parent_a[column]
                    )

            for feature in feature_cols:

                synthetic_row[feature] = float(
                    synthetic_features[
                        feature
                    ]
                )

            synthetic_row["event_id"] = (
                synthetic_event_id
            )

            synthetic_row[TARGET_COL] = (
                synthetic_soh
            )

            synthetic_row["mileage"] = (
                synthetic_mileage
            )

            synthetic_row["soh_bin"] = (
                soh_bin_value
            )

            synthetic_row["sample_origin"] = (
                "smoter"
            )

            synthetic_row["split"] = (
                "train_smoter"
            )

            synthetic_row["source_fold"] = int(
                fold_id
            )

            synthetic_row["car"] = (
                f"SMOTER_F{fold_id}"
            )

            synthetic_row["charge_segment"] = (
                "synthetic"
            )

            synthetic_rows.append(
                synthetic_row
            )

            generation_audit_rows.append(
                {
                    "fold": int(fold_id),
                    "synthetic_event_id":
                        synthetic_event_id,

                    "soh_bin":
                        str(soh_bin_value),

                    "parent_a_event_id":
                        parent_a["event_id"],

                    "parent_b_event_id":
                        parent_b["event_id"],

                    "parent_a_soh":
                        float(
                            parent_a[TARGET_COL]
                        ),

                    "parent_b_soh":
                        float(
                            parent_b[TARGET_COL]
                        ),

                    "synthetic_soh":
                        synthetic_soh,

                    "parent_a_mileage":
                        float(
                            parent_a["mileage"]
                        ),

                    "parent_b_mileage":
                        float(
                            parent_b["mileage"]
                        ),

                    "synthetic_mileage":
                        synthetic_mileage,

                    "parent_mileage_diff":
                        float(
                            abs(
                                parent_a["mileage"]
                                - parent_b["mileage"]
                            )
                        ),

                    "interpolation_factor":
                        interpolation_factor,
                }
            )

    synthetic_fold = pd.DataFrame(
        synthetic_rows
    )

    generation_audit_fold = pd.DataFrame(
        generation_audit_rows
    )

    if len(synthetic_fold) > 0:

        synthetic_fold = synthetic_fold[
            train_real_fold.columns
        ].copy()

        train_balanced_fold = pd.concat(
            [
                train_real_fold,
                synthetic_fold,
            ],
            axis=0,
            ignore_index=True,
        )

    else:

        synthetic_fold = pd.DataFrame(
            columns=train_real_fold.columns
        )

        train_balanced_fold = (
            train_real_fold.copy()
        )
    return {
        "train_balanced":
            train_balanced_fold,

        "synthetic":
            synthetic_fold,

        "balance_plan":
            balance_plan_fold,

        "neighbor_audit":
            neighbor_audit_fold,

        "generation_audit":
            generation_audit_fold,

        "constant_features":
            constant_features,

        "distance_features":
            smoter_distance_features,

        "feature_medians":
            feature_medians,

        "scaler":
            scaler,
    }


def criar_modelo_global(
    random_state=MODEL_RANDOM_STATE,
):
    """Cria o ExtraTrees global campeão."""

    params = copy.deepcopy(
        GLOBAL_MODEL_PARAMS
    )

    params["random_state"] = int(
        random_state
    )

    return ExtraTreesRegressor(
        **params
    )


def criar_especialista(
    class_id,
    random_state=MODEL_RANDOM_STATE,
):
    """
    Cria um dos três especialistas.

    0 = XGBoost baixo
    1 = ExtraTrees intermediário
    2 = ExtraTrees alto
    """

    if class_id == 0:

        params = copy.deepcopy(
            EXPERT_A_XGB_PARAMS
        )

        params["random_state"] = int(
            random_state
        )

        return XGBRegressor(
            **params
        )

    if class_id in {1, 2}:

        params = copy.deepcopy(
            EXPERT_BC_ET_PARAMS
        )

        params["random_state"] = int(
            random_state
        )

        return ExtraTreesRegressor(
            **params
        )

    raise ValueError(
        f"class_id inválido: {class_id}"
    )


def criar_gate_xgboost(
    random_state=MODEL_RANDOM_STATE,
):
    """Cria o gate XGBoost de três classes."""

    params = copy.deepcopy(
        GATE_XGB_PARAMS
    )

    params["random_state"] = int(
        random_state
    )

    return XGBClassifier(
        **params
    )


def obter_mascara_especialista(
    dataframe,
    class_id,
    target_col=TARGET_COL,
):
    """Seleciona os eventos da faixa real do especialista."""

    if class_id not in EXPERT_CONFIG:
        raise ValueError(
            f"Especialista inexistente: {class_id}"
        )

    soh_min = float(
        EXPERT_CONFIG[class_id]["soh_min"]
    )

    soh_max = float(
        EXPERT_CONFIG[class_id]["soh_max"]
    )

    return dataframe[target_col].between(
        soh_min,
        soh_max,
        inclusive="both",
    )


def calcular_metricas_regressao(
    y_true,
    y_pred,
):
    """Calcula as métricas oficiais de regressão."""

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred possuem shapes diferentes: "
            f"{y_true.shape} e {y_pred.shape}."
        )

    if not np.isfinite(y_true).all():
        raise ValueError(
            "y_true contém NaN ou Inf."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "y_pred contém NaN ou Inf."
        )

    error = y_pred - y_true
    absolute_error = np.abs(error)

    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),

        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),

        "mape_%": float(
            np.mean(
                absolute_error
                / np.clip(
                    np.abs(y_true),
                    1e-12,
                    None,
                )
            )
            * 100
        ),

        "bias": float(
            np.mean(error)
        ),

        "erro_abs_p95": float(
            np.percentile(
                absolute_error,
                95,
            )
        ),

        "erro_abs_max": float(
            np.max(
                absolute_error
            )
        ),
    }


def calcular_metricas_gate(
    y_true,
    y_pred,
    y_proba=None,
):
    """Calcula as métricas oficiais do gate."""

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    result = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    if y_proba is not None:

        y_proba = np.asarray(
            y_proba,
            dtype=float,
        )

        result["log_loss"] = float(
            log_loss(
                y_true,
                y_proba,
                labels=[0, 1, 2],
            )
        )

    else:

        result["log_loss"] = np.nan

    return result

def combinar_predicoes_especialistas(
    expert_prediction_matrix,
    selected_classes,
):
    """
    Seleciona uma das três predições de especialista
    para cada evento.
    """

    matrix = np.asarray(
        expert_prediction_matrix,
        dtype=float,
    )

    classes = np.asarray(
        selected_classes,
        dtype=int,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "A matriz dos especialistas deve ser 2D."
        )

    if matrix.shape[1] != 3:
        raise ValueError(
            "A matriz deve possuir três colunas."
        )

    if len(classes) != matrix.shape[0]:
        raise ValueError(
            "O vetor de classes não está alinhado "
            "com a matriz dos especialistas."
        )

    if not np.isin(
        classes,
        [0, 1, 2],
    ).all():
        raise ValueError(
            "Existem classes inválidas no roteamento."
        )

    row_indices = np.arange(
        matrix.shape[0]
    )

    return matrix[
        row_indices,
        classes,
    ]

def rotear_pela_previsao_global(
    global_predictions,
    boundary_ab=ROUTER_BOUNDARY_A_B,
    boundary_bc=ROUTER_BOUNDARY_B_C,
):
    """
    Converte a previsão contínua do modelo global
    em classe de especialista.
    """

    predictions = np.asarray(
        global_predictions,
        dtype=float,
    )

    return np.select(
        [
            predictions < boundary_ab,
            predictions < boundary_bc,
        ],
        [
            0,
            1,
        ],
        default=2,
    ).astype(int)

def calcular_oracle_especialistas(
    y_true,
    expert_prediction_matrix,
):
    """
    Escolhe o especialista de menor erro absoluto para
    cada evento.

    Utilizado apenas para avaliação e para construção
    dos rótulos do gate.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    matrix = np.asarray(
        expert_prediction_matrix,
        dtype=float,
    )

    if matrix.shape != (
        len(y_true),
        3,
    ):
        raise ValueError(
            "Shape inválido da matriz dos especialistas: "
            f"{matrix.shape}."
        )

    absolute_errors = np.abs(
        matrix
        - y_true[:, None]
    )

    oracle_classes = np.argmin(
        absolute_errors,
        axis=1,
    ).astype(int)

    oracle_predictions = (
        combinar_predicoes_especialistas(
            matrix,
            oracle_classes,
        )
    )

    sorted_errors = np.sort(
        absolute_errors,
        axis=1,
    )

    expert_margin = (
        sorted_errors[:, 1]
        - sorted_errors[:, 0]
    )

    denominator = np.maximum(
        sorted_errors[:, 1],
        1e-12,
    )

    oracle_confidence = np.clip(
        expert_margin / denominator,
        0.0,
        1.0,
    )

    return {
        "classes": oracle_classes,
        "predictions": oracle_predictions,
        "absolute_errors": absolute_errors,
        "expert_margin": expert_margin,
        "oracle_confidence": oracle_confidence,
    }


def gerar_rotulos_gate_oof(
    train_balanced_fold,
    outer_fold_id,
    n_inner_splits=INNER_OOF_SPLITS,
    random_state=MODEL_RANDOM_STATE,
    verbose=True,
):
    """
    Gera predições OOF internas dos três especialistas e
    define o melhor especialista para cada evento do treino
    balanceado de um fold externo.

    Parâmetros
    ----------
    train_balanced_fold : pandas.DataFrame
        Treino real + SMOTER do fold externo.

    outer_fold_id : int
        Identificador do fold externo.

    n_inner_splits : int
        Número de divisões internas.

    random_state : int
        Semente-base.

    verbose : bool
        Controla a impressão do progresso.

    Retorna
    -------
    dict com:
        gate_train_dataset
        expert_oof_predictions
        oracle_classes
        oracle_predictions
        oracle_audit
        inner_fold_audit
        expert_inner_metrics
    """

    total_start = time.perf_counter()

    data = (
        train_balanced_fold
        .copy()
        .reset_index(drop=True)
    )

    n_events = len(data)

    required_columns = list(
        dict.fromkeys(
            [
                "event_id",
                "soh_bin",
                "sample_origin",
                TARGET_COL,
            ]
            + FEATURES_EXPERTS
            + FEATURES_GATE
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Fold externo {outer_fold_id}: colunas ausentes:\n"
            f"{missing_columns}"
        )

    if data["event_id"].duplicated().any():
        raise ValueError(
            f"Fold externo {outer_fold_id}: existem "
            "event_id duplicados no treino balanceado."
        )

    if data["soh_bin"].isna().any():
        raise ValueError(
            f"Fold externo {outer_fold_id}: existem eventos "
            "sem soh_bin."
        )

    model_features_union = list(
        dict.fromkeys(
            FEATURES_EXPERTS
            + FEATURES_GATE
        )
    )

    invalid_count = int(
        data[model_features_union]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        .sum()
        .sum()
    )

    if invalid_count > 0:
        raise ValueError(
            f"Fold externo {outer_fold_id}: existem "
            f"{invalid_count} valores inválidos nas features."
        )

    inner_bin_counts = (
        data["soh_bin"]
        .value_counts(sort=False)
    )

    insufficient_bins = inner_bin_counts.loc[
        inner_bin_counts < n_inner_splits
    ]

    if not insufficient_bins.empty:
        raise ValueError(
            f"Fold externo {outer_fold_id}: há bins com menos "
            f"de {n_inner_splits} eventos:\n"
            f"{insufficient_bins}"
        )
    
    inner_splitter = StratifiedKFold(
        n_splits=n_inner_splits,
        shuffle=True,
        random_state=(
            int(random_state)
            + int(outer_fold_id) * 100
        ),
    )

    stratification_labels = (
        data["soh_bin"]
        .astype(str)
        .to_numpy()
    )

    inner_fold_assignment = np.full(
        n_events,
        fill_value=-1,
        dtype=int,
    )

    for inner_fold_id, (_, valid_indices) in enumerate(
        inner_splitter.split(
            X=np.zeros(
                n_events,
                dtype=np.uint8,
            ),
            y=stratification_labels,
        )
    ):

        inner_fold_assignment[
            valid_indices
        ] = inner_fold_id

    if np.any(inner_fold_assignment < 0):
        raise ValueError(
            f"Fold externo {outer_fold_id}: existem eventos "
            "sem fold interno atribuído."
        )

    data["inner_fold"] = (
        inner_fold_assignment
    )

    expert_oof_matrix = np.full(
        shape=(n_events, 3),
        fill_value=np.nan,
        dtype=float,
    )

    prediction_count = np.zeros(
        n_events,
        dtype=int,
    )

    inner_audit_rows = []
    expert_metric_rows = []

    
    for inner_fold_id in range(
        n_inner_splits
    ):

        inner_start = time.perf_counter()

        inner_valid_mask = (
            data["inner_fold"]
            .eq(inner_fold_id)
        )

        inner_train = (
            data.loc[
                ~inner_valid_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        inner_valid = (
            data.loc[
                inner_valid_mask
            ]
            .copy()
        )

        inner_valid_indices = (
            inner_valid.index.to_numpy()
        )


        shared_ids = set(
            inner_train["event_id"]
        ).intersection(
            set(
                inner_valid["event_id"]
            )
        )

        if shared_ids:
            raise ValueError(
                f"Fold externo {outer_fold_id}, interno "
                f"{inner_fold_id}: vazamento de event_id."
            )

        # --------------------------------------------------
        # Treinar cada especialista
        # --------------------------------------------------

        for class_id in range(3):

            expert_config = (
                EXPERT_CONFIG[class_id]
            )

            train_range_mask = (
                obter_mascara_especialista(
                    inner_train,
                    class_id,
                )
            )

            expert_train = (
                inner_train.loc[
                    train_range_mask
                ]
                .copy()
                .reset_index(drop=True)
            )

            if len(expert_train) == 0:
                raise ValueError(
                    f"Fold externo {outer_fold_id}, interno "
                    f"{inner_fold_id}, especialista "
                    f"{class_id}: treino vazio."
                )

            expert_model = criar_especialista(
                class_id=class_id,
                random_state=(
                    int(random_state)
                    + int(outer_fold_id) * 1000
                    + int(inner_fold_id) * 10
                    + int(class_id)
                ),
            )

            train_start = time.perf_counter()

            expert_model.fit(
                expert_train[
                    FEATURES_EXPERTS
                ],
                expert_train[
                    TARGET_COL
                ],
            )

            expert_train_time = (
                time.perf_counter()
                - train_start
            )

            prediction_start = (
                time.perf_counter()
            )

            expert_predictions = np.asarray(
                expert_model.predict(
                    inner_valid[
                        FEATURES_EXPERTS
                    ]
                ),
                dtype=float,
            )

            expert_prediction_time = (
                time.perf_counter()
                - prediction_start
            )

            if not np.isfinite(
                expert_predictions
            ).all():

                raise ValueError(
                    f"Fold externo {outer_fold_id}, interno "
                    f"{inner_fold_id}, especialista "
                    f"{class_id}: predições inválidas."
                )

            expert_oof_matrix[
                inner_valid_indices,
                class_id,
            ] = expert_predictions

            # Avaliação do especialista apenas em sua faixa
            valid_range_mask = (
                obter_mascara_especialista(
                    inner_valid,
                    class_id,
                )
                .to_numpy()
            )

            if valid_range_mask.sum() > 0:

                metrics_range = (
                    calcular_metricas_regressao(
                        inner_valid.loc[
                            valid_range_mask,
                            TARGET_COL,
                        ],
                        expert_predictions[
                            valid_range_mask
                        ],
                    )
                )

            else:

                metrics_range = {
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                    "mape_%": np.nan,
                    "bias": np.nan,
                    "erro_abs_p95": np.nan,
                    "erro_abs_max": np.nan,
                }

            expert_metric_rows.append(
                {
                    "outer_fold":
                        int(outer_fold_id),

                    "inner_fold":
                        int(inner_fold_id),

                    "class_id":
                        int(class_id),

                    "especialista":
                        expert_config["name"],

                    "n_train_expert":
                        len(expert_train),

                    "n_valid_total":
                        len(inner_valid),

                    "n_valid_range":
                        int(
                            valid_range_mask.sum()
                        ),

                    "tempo_treino_s":
                        expert_train_time,

                    "tempo_predicao_s":
                        expert_prediction_time,

                    **metrics_range,
                }
            )

        prediction_count[
            inner_valid_indices
        ] += 1

        inner_time = (
            time.perf_counter()
            - inner_start
        )

        inner_audit_rows.append(
            {
                "outer_fold":
                    int(outer_fold_id),

                "inner_fold":
                    int(inner_fold_id),

                "n_train":
                    len(inner_train),

                "n_valid":
                    len(inner_valid),

                "n_train_real":
                    int(
                        inner_train[
                            "sample_origin"
                        ]
                        .eq("real")
                        .sum()
                    ),

                "n_train_smoter":
                    int(
                        inner_train[
                            "sample_origin"
                        ]
                        .eq("smoter")
                        .sum()
                    ),

                "n_valid_real":
                    int(
                        inner_valid[
                            "sample_origin"
                        ]
                        .eq("real")
                        .sum()
                    ),

                "n_valid_smoter":
                    int(
                        inner_valid[
                            "sample_origin"
                        ]
                        .eq("smoter")
                        .sum()
                    ),

                "event_ids_shared":
                    len(shared_ids),

                "tempo_total_s":
                    inner_time,
            }
        )

    # ------------------------------------------------------
    # 1.6. Auditoria final das predições OOF
    # ------------------------------------------------------

    if not np.all(
        prediction_count == 1
    ):
        values, counts = np.unique(
            prediction_count,
            return_counts=True,
        )

        raise ValueError(
            f"Fold externo {outer_fold_id}: cada evento "
            "deveria receber exatamente uma previsão OOF. "
            f"Contagens encontradas: "
            f"{dict(zip(values, counts))}"
        )

    if not np.isfinite(
        expert_oof_matrix
    ).all():

        invalid_positions = np.argwhere(
            ~np.isfinite(
                expert_oof_matrix
            )
        )

        raise ValueError(
            f"Fold externo {outer_fold_id}: existem "
            "predições OOF ausentes ou inválidas. "
            f"Primeiras posições: "
            f"{invalid_positions[:10].tolist()}"
        )

    # ------------------------------------------------------
    # 1.7. Construir o Oracle OOF
    # ------------------------------------------------------

    y_train = (
        data[TARGET_COL]
        .to_numpy(dtype=float)
    )

    oracle_result = (
        calcular_oracle_especialistas(
            y_true=y_train,
            expert_prediction_matrix=(
                expert_oof_matrix
            ),
        )
    )

    oracle_classes = (
        oracle_result["classes"]
    )

    oracle_predictions = (
        oracle_result["predictions"]
    )

    absolute_errors = (
        oracle_result["absolute_errors"]
    )

    # ------------------------------------------------------
    # 1.8. Dataset final de treino do gate
    # ------------------------------------------------------

    gate_train_dataset = (
        data.copy()
    )

    gate_train_dataset[
        "best_expert_class"
    ] = oracle_classes

    gate_train_dataset[
        "best_expert"
    ] = pd.Series(
        oracle_classes
    ).map(
        EXPERT_CLASS_NAMES
    )

    gate_train_dataset[
        "pred_oof_expert_A"
    ] = expert_oof_matrix[:, 0]

    gate_train_dataset[
        "pred_oof_expert_B"
    ] = expert_oof_matrix[:, 1]

    gate_train_dataset[
        "pred_oof_expert_C"
    ] = expert_oof_matrix[:, 2]

    gate_train_dataset[
        "abs_error_oof_expert_A"
    ] = absolute_errors[:, 0]

    gate_train_dataset[
        "abs_error_oof_expert_B"
    ] = absolute_errors[:, 1]

    gate_train_dataset[
        "abs_error_oof_expert_C"
    ] = absolute_errors[:, 2]

    gate_train_dataset[
        "pred_oof_oracle"
    ] = oracle_predictions

    gate_train_dataset[
        "expert_margin"
    ] = oracle_result[
        "expert_margin"
    ]

    gate_train_dataset[
        "oracle_confidence"
    ] = oracle_result[
        "oracle_confidence"
    ]

    # ------------------------------------------------------
    # 1.9. Auditoria dos rótulos
    # ------------------------------------------------------

    oracle_audit = (
        gate_train_dataset[
            "best_expert"
        ]
        .value_counts()
        .rename_axis(
            "best_expert"
        )
        .reset_index(
            name="n_eventos"
        )
    )

    oracle_audit[
        "percentual_%"
    ] = (
        100
        * oracle_audit[
            "n_eventos"
        ]
        / len(
            gate_train_dataset
        )
    ).round(2)

    oracle_audit.insert(
        0,
        "outer_fold",
        int(outer_fold_id),
    )

    inner_fold_audit = pd.DataFrame(
        inner_audit_rows
    )

    expert_inner_metrics = pd.DataFrame(
        expert_metric_rows
    )

    total_time = (
        time.perf_counter()
        - total_start
    )


    return {
        "gate_train_dataset":
            gate_train_dataset,

        "expert_oof_predictions":
            expert_oof_matrix,

        "oracle_classes":
            oracle_classes,

        "oracle_predictions":
            oracle_predictions,

        "oracle_audit":
            oracle_audit,

        "inner_fold_audit":
            inner_fold_audit,

        "expert_inner_metrics":
            expert_inner_metrics,

        "total_time_s":
            total_time,
    }


# %% leitura
# leitura


df_gold = pd.read_csv(
    GOLD_EVENTS_PATH
)

with open(
    FEATURE_LIST_PATH,
    "r",
    encoding="utf-8"
) as file:
    gold_column_lists = json.load(file)
# %% preparação
# preparação

df_gold[TARGET_COL] = pd.to_numeric(
    df_gold[TARGET_COL],
    errors="coerce"
)

df_gold["mileage"] = pd.to_numeric(
    df_gold["mileage"],
    errors="coerce"
)

df_gold_clean = (
    df_gold.loc[
        df_gold[TARGET_COL].between(
            SOH_MIN,
            SOH_MAX,
            inclusive="both"
        )
    ]
    .copy()
    .reset_index(drop=True)
)


df_gold_clean["soh_bin"] = pd.cut(
    df_gold_clean[TARGET_COL],
    bins=SOH_BIN_EDGES,
    include_lowest=True,
    right=True
)

df_cv = (
    df_gold_clean
    .copy()
    .reset_index(drop=True)
)

if "soh_bin" not in df_cv.columns:
    raise KeyError(
        "A coluna 'soh_bin' não foi encontrada. "
        "Execute primeiro a célula de limpeza."
    )

if df_cv["soh_bin"].isna().any():
    raise ValueError(
        "Existem eventos sem soh_bin."
    )

stratification_labels = (
    df_cv["soh_bin"]
    .astype(str)
    .to_numpy()
)


cv_splitter = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

df_cv["cv_fold"] = -1

for fold_id, (_, validation_indices) in enumerate(
    cv_splitter.split(
        X=np.zeros(
            len(df_cv),
            dtype=np.uint8
        ),
        y=stratification_labels,
    )
):

    df_cv.loc[
        validation_indices,
        "cv_fold"
    ] = fold_id

df_cv["cv_fold"] = (
    df_cv["cv_fold"]
    .astype(int)
)


smoter_fold_summary_rows = []
smoter_balance_plan_all = []
smoter_neighbor_audit_all = []
smoter_generation_audit_all = []

smoter_total_start = time.perf_counter()

for fold_id in range(N_SPLITS):

    fold_start = time.perf_counter()

    valid_mask = df_cv["cv_fold"].eq(fold_id)

    train_real_fold = (
        df_cv.loc[~valid_mask]
        .copy()
        .reset_index(drop=True)
    )

    valid_real_fold = (
        df_cv.loc[valid_mask]
        .copy()
        .reset_index(drop=True)
    )

    train_real_fold["split"] = "train_real"
    train_real_fold["source_fold"] = fold_id
    train_real_fold["sample_origin"] = "real"

    valid_real_fold["split"] = "validation_real"
    valid_real_fold["source_fold"] = fold_id
    valid_real_fold["sample_origin"] = "real"

    smoter_result = aplicar_smoter_fold(
        train_real_fold=train_real_fold,
        fold_id=fold_id,
        feature_cols=FEATURE_COLS,
        random_state=SMOTER_RANDOM_STATE,
    )

    train_balanced_fold = (
        smoter_result["train_balanced"]
        .copy()
        .reset_index(drop=True)
    )

    synthetic_fold = (
        smoter_result["synthetic"]
        .copy()
        .reset_index(drop=True)
    )

    balance_plan_fold = (
        smoter_result["balance_plan"]
        .copy()
    )

    neighbor_audit_fold = (
        smoter_result["neighbor_audit"]
        .copy()
    )

    generation_audit_fold = (
        smoter_result["generation_audit"]
        .copy()
    )

    distribution_train_real = (
        train_real_fold
        .groupby("soh_bin", observed=False)
        .size()
        .rename("n_train_real")
    )

    distribution_train_balanced = (
        train_balanced_fold
        .groupby("soh_bin", observed=False)
        .size()
        .rename("n_train_balanced")
    )

    distribution_valid_real = (
        valid_real_fold
        .groupby("soh_bin", observed=False)
        .size()
        .rename("n_valid_real")
    )

    fold_distribution = pd.concat(
        [
            distribution_train_real,
            distribution_train_balanced,
            distribution_valid_real,
        ],
        axis=1,
    ).fillna(0)

    fold_distribution = (
        fold_distribution
        .astype(int)
        .reset_index()
    )

    fold_distribution.insert(
        0,
        "fold",
        fold_id,
    )

    fold_distribution["n_synthetic"] = (
        fold_distribution["n_train_balanced"]
        - fold_distribution["n_train_real"]
    )

    CV_FOLD_DATA[fold_id] = {
        "train_real": train_real_fold,
        "train_balanced": train_balanced_fold,
        "synthetic": synthetic_fold,
        "validation_real": valid_real_fold,
        "balance_plan": balance_plan_fold,
        "fold_distribution": fold_distribution,
        "constant_features": (
            smoter_result["constant_features"]
        ),
        "distance_features": (
            smoter_result["distance_features"]
        ),
    }


    gc.collect()


gate_oof_summary_rows = []
oracle_audit_all = []
inner_fold_audit_all = []
expert_inner_metrics_all = []

gate_oof_total_start = time.perf_counter()


for outer_fold_id in range(N_SPLITS):

    fold_start = time.perf_counter()

    fold_gate_dir = (
        GATE_OOF_DIR
        / f"fold_{outer_fold_id}"
    )

    fold_gate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gate_train_path = (
        fold_gate_dir
        / "gate_train_oof.csv"
    )

    oracle_audit_path = (
        fold_gate_dir
        / "oracle_oof_distribution.xlsx"
    )

    inner_audit_path = (
        fold_gate_dir
        / "inner_fold_audit.xlsx"
    )

    expert_metrics_path = (
        fold_gate_dir
        / "expert_inner_metrics.xlsx"
    )

    train_balanced_fold = (
        CV_FOLD_DATA[outer_fold_id][
            "train_balanced"
        ]
        .copy()
        .reset_index(drop=True)
    )

    n_expected = len(
        train_balanced_fold
    )

    loaded_from_disk = False

    if (
        REUSE_GATE_OOF
        and gate_train_path.exists()
        and oracle_audit_path.exists()
        and inner_audit_path.exists()
        and expert_metrics_path.exists()
    ):
        gate_train_dataset = pd.read_csv(
            gate_train_path
        )

        oracle_audit = pd.read_excel(
            oracle_audit_path
        )

        inner_fold_audit = pd.read_excel(
            inner_audit_path
        )

        expert_inner_metrics = pd.read_excel(
            expert_metrics_path
        )

        missing_saved_columns = [
            column
            for column in REQUIRED_GATE_OOF_COLUMNS
            if column not in gate_train_dataset.columns
        ]

        saved_is_valid = True

        if missing_saved_columns:

            print(
                "Arquivo salvo incompleto. "
                "Colunas ausentes:"
            )

            print(missing_saved_columns)

            saved_is_valid = False

        if len(gate_train_dataset) != n_expected:

            print(
                "Número de eventos salvo diferente do "
                "treino balanceado atual:"
            )

            print(
                "Salvo   :",
                len(gate_train_dataset)
            )

            print(
                "Esperado:",
                n_expected
            )

            saved_is_valid = False

        if (
            "event_id"
            in gate_train_dataset.columns
        ):

            ids_saved = set(
                gate_train_dataset[
                    "event_id"
                ].astype(str)
            )

            ids_current = set(
                train_balanced_fold[
                    "event_id"
                ].astype(str)
            )

            if ids_saved != ids_current:

                print(
                    "Os event_id salvos não correspondem "
                    "ao treino balanceado atual."
                )

                saved_is_valid = False

        if saved_is_valid:

            numeric_oof_columns = [
                "pred_oof_expert_A",
                "pred_oof_expert_B",
                "pred_oof_expert_C",
                "abs_error_oof_expert_A",
                "abs_error_oof_expert_B",
                "abs_error_oof_expert_C",
                "pred_oof_oracle",
                "expert_margin",
                "oracle_confidence",
            ]

            n_invalid_saved = int(
                gate_train_dataset[
                    numeric_oof_columns
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .isna()
                .sum()
                .sum()
            )

            if n_invalid_saved > 0:

                print(
                    "O resultado salvo possui "
                    f"{n_invalid_saved} valores inválidos."
                )

                saved_is_valid = False

    if not loaded_from_disk:

        oof_result = gerar_rotulos_gate_oof(
            train_balanced_fold=(
                train_balanced_fold
            ),
            outer_fold_id=outer_fold_id,
            n_inner_splits=INNER_OOF_SPLITS,
            random_state=(
                MODEL_RANDOM_STATE
            ),
            verbose=True,
        )

        gate_train_dataset = (
            oof_result[
                "gate_train_dataset"
            ]
            .copy()
        )

        oracle_audit = (
            oof_result[
                "oracle_audit"
            ]
            .copy()
        )

        inner_fold_audit = (
            oof_result[
                "inner_fold_audit"
            ]
            .copy()
        )

        expert_inner_metrics = (
            oof_result[
                "expert_inner_metrics"
            ]
            .copy()
        )

        gate_train_dataset.to_csv(
            gate_train_path,
            index=False,
        )

        oracle_audit.to_excel(
            oracle_audit_path,
            index=False,
        )

        inner_fold_audit.to_excel(
            inner_audit_path,
            index=False,
        )

        expert_inner_metrics.to_excel(
            expert_metrics_path,
            index=False,
        )

    gate_train_dataset[
        "best_expert_class"
    ] = pd.to_numeric(
        gate_train_dataset[
            "best_expert_class"
        ],
        errors="raise",
    ).astype(int)

    gate_train_dataset[
        "inner_fold"
    ] = pd.to_numeric(
        gate_train_dataset[
            "inner_fold"
        ],
        errors="raise",
    ).astype(int)

    inner_fold_values = set(
        gate_train_dataset[
            "inner_fold"
        ].unique()
        .tolist()
    )

    expected_inner_folds = set(
        range(INNER_OOF_SPLITS)
    )

    current_id_order = (
        train_balanced_fold[
            "event_id"
        ]
        .astype(str)
        .tolist()
    )

    gate_id_order = (
        gate_train_dataset[
            "event_id"
        ]
        .astype(str)
        .tolist()
    )

    if current_id_order != gate_id_order:

        # É permitido reordenar se os conjuntos forem iguais
        current_order_map = {
            event_id: position
            for position, event_id
            in enumerate(current_id_order)
        }

        gate_train_dataset[
            "_current_order"
        ] = (
            gate_train_dataset[
                "event_id"
            ]
            .astype(str)
            .map(current_order_map)
        )


        gate_train_dataset = (
            gate_train_dataset
            .sort_values(
                "_current_order"
            )
            .drop(
                columns="_current_order"
            )
            .reset_index(drop=True)
        )


    # Verificar consistência do Oracle salvo
    expert_oof_matrix = (
        gate_train_dataset[
            [
                "pred_oof_expert_A",
                "pred_oof_expert_B",
                "pred_oof_expert_C",
            ]
        ]
        .to_numpy(dtype=float)
    )

    y_gate_oof = (
        gate_train_dataset[
            TARGET_COL
        ]
        .to_numpy(dtype=float)
    )
    
    oracle_check = (
        calcular_oracle_especialistas(
            y_true=y_gate_oof,
            expert_prediction_matrix=(
                expert_oof_matrix
            ),
        )
    )

    oracle_distribution = (
        gate_train_dataset[
            [
                "best_expert_class",
                "best_expert",
            ]
        ]
        .value_counts()
        .rename("n_eventos")
        .reset_index()
        .sort_values(
            "best_expert_class"
        )
        .reset_index(drop=True)
    )

    oracle_distribution[
        "percentual_%"
    ] = (
        100
        * oracle_distribution[
            "n_eventos"
        ]
        / len(
            gate_train_dataset
        )
    ).round(2)

    oracle_distribution.insert(
        0,
        "outer_fold",
        outer_fold_id,
    )

    oracle_oof_metrics = (
        calcular_metricas_regressao(
            y_true=y_gate_oof,
            y_pred=oracle_check[
                "predictions"
            ],
        )
    )

    GATE_OOF_DATA[
        outer_fold_id
    ] = {
        "gate_train_dataset":
            gate_train_dataset,

        "expert_oof_predictions":
            expert_oof_matrix,

        "oracle_classes":
            oracle_check["classes"],

        "oracle_predictions":
            oracle_check[
                "predictions"
            ],

        "oracle_distribution":
            oracle_distribution,

        "inner_fold_audit":
            inner_fold_audit,

        "expert_inner_metrics":
            expert_inner_metrics,
    }

    # Também conectar ao dicionário principal dos folds
    CV_FOLD_DATA[
        outer_fold_id
    ]["gate_train_oof"] = (
        gate_train_dataset
    )


    oracle_audit_all.append(
        oracle_distribution
    )

    inner_fold_audit_all.append(
        inner_fold_audit
    )

    expert_inner_metrics_all.append(
        expert_inner_metrics
    )

    elapsed_fold = (
        time.perf_counter()
        - fold_start
    )

    class_counts = (
        gate_train_dataset[
            "best_expert_class"
        ]
        .value_counts()
        .to_dict()
    )

    gate_oof_summary_rows.append(
        {
            "outer_fold":
                outer_fold_id,

            "n_train_balanced":
                len(gate_train_dataset),

            "n_real":
                int(
                    gate_train_dataset[
                        "sample_origin"
                    ]
                    .eq("real")
                    .sum()
                ),

            "n_smoter":
                int(
                    gate_train_dataset[
                        "sample_origin"
                    ]
                    .eq("smoter")
                    .sum()
                ),

            "n_class_A":
                int(
                    class_counts.get(
                        0,
                        0,
                    )
                ),

            "n_class_B":
                int(
                    class_counts.get(
                        1,
                        0,
                    )
                ),

            "n_class_C":
                int(
                    class_counts.get(
                        2,
                        0,
                    )
                ),

            "percent_class_A_%":
                100
                * class_counts.get(0, 0)
                / len(gate_train_dataset),

            "percent_class_B_%":
                100
                * class_counts.get(1, 0)
                / len(gate_train_dataset),

            "percent_class_C_%":
                100
                * class_counts.get(2, 0)
                / len(gate_train_dataset),

            "oracle_oof_rmse":
                oracle_oof_metrics["rmse"],

            "oracle_oof_mae":
                oracle_oof_metrics["mae"],

            "oracle_oof_r2":
                oracle_oof_metrics["r2"],

            "expert_margin_mean":
                float(
                    gate_train_dataset[
                        "expert_margin"
                    ].mean()
                ),

            "oracle_confidence_mean":
                float(
                    gate_train_dataset[
                        "oracle_confidence"
                    ].mean()
                ),

            "reused_from_disk":
                loaded_from_disk,

            "tempo_total_fold_s":
                elapsed_fold,
        }
    )

    gc.collect()

# %% treinamento
# treinamento
training_summary_rows = []
expert_training_summary_rows = []

for fold_id in range(N_SPLITS):

    fold_start = time.perf_counter()

    fold_data = CV_FOLD_DATA[fold_id]

    train_balanced_fold = (
        fold_data["train_balanced"]
        .copy()
        .reset_index(drop=True)
    )

    validation_real_fold = (
        fold_data["validation_real"]
        .copy()
        .reset_index(drop=True)
    )

    gate_train_oof = (
        fold_data["gate_train_oof"]
        .copy()
        .reset_index(drop=True)
    )


    print("\nTreinando modelo global...")

    global_start = time.perf_counter()

    global_model = criar_modelo_global(
        random_state=(
            MODEL_RANDOM_STATE
            + fold_id
        )
    )

    global_model.fit(
        train_balanced_fold[
            FEATURES_GLOBAL
        ],
        train_balanced_fold[
            TARGET_COL
        ],
    )

    global_training_time = (
        time.perf_counter()
        - global_start
    )

    experts = {}
    expert_training_times = {}
    expert_training_sizes = {}

    print("\nTreinando especialistas...")

    experts_start = time.perf_counter()

    for class_id in range(3):

        expert_name = (
            EXPERT_CLASS_NAMES[
                class_id
            ]
        )

        expert_mask = (
            obter_mascara_especialista(
                train_balanced_fold,
                class_id,
            )
        )

        expert_train = (
            train_balanced_fold.loc[
                expert_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(expert_train) == 0:
            raise ValueError(
                f"Fold {fold_id}: treino vazio para "
                f"o especialista {expert_name}."
            )

        expert_model = criar_especialista(
            class_id=class_id,
            random_state=(
                MODEL_RANDOM_STATE
                + fold_id * 10
                + class_id
            ),
        )

        expert_start = time.perf_counter()

        expert_model.fit(
            expert_train[
                FEATURES_EXPERTS
            ],
            expert_train[
                TARGET_COL
            ],
        )

        expert_time = (
            time.perf_counter()
            - expert_start
        )

        experts[class_id] = expert_model
        expert_training_times[class_id] = expert_time
        expert_training_sizes[class_id] = len(
            expert_train
        )

        expert_training_summary_rows.append(
            {
                "fold": fold_id,
                "class_id": class_id,
                "especialista": expert_name,
                "modelo": (
                    expert_model
                    .__class__
                    .__name__
                ),
                "soh_min": (
                    EXPERT_CONFIG[
                        class_id
                    ]["soh_min"]
                ),
                "soh_max": (
                    EXPERT_CONFIG[
                        class_id
                    ]["soh_max"]
                ),
                "n_train": len(
                    expert_train
                ),
                "n_train_real": int(
                    expert_train[
                        "sample_origin"
                    ]
                    .eq("real")
                    .sum()
                ),
                "n_train_smoter": int(
                    expert_train[
                        "sample_origin"
                    ]
                    .eq("smoter")
                    .sum()
                ),
                "n_features": len(
                    FEATURES_EXPERTS
                ),
                "tempo_treino_s": (
                    expert_time
                ),
            }
        )

        print(
            f"{expert_name}: "
            f"n={len(expert_train)} | "
            f"tempo={expert_time:.2f}s"
        )

    experts_training_time = (
        time.perf_counter()
        - experts_start
    )


    print("\nTreinando gate XGBoost...")

    X_gate_train = (
        gate_train_oof[
            FEATURES_GATE
        ]
    )

    y_gate_train = (
        gate_train_oof[
            "best_expert_class"
        ]
        .astype(int)
    )

    gate_classes = set(
        y_gate_train.unique()
        .tolist()
    )

    if gate_classes != {0, 1, 2}:
        raise ValueError(
            f"Fold {fold_id}: classes do gate "
            f"encontradas={gate_classes}; "
            "esperado={0, 1, 2}."
        )

    gate_start = time.perf_counter()

    gate_model = criar_gate_xgboost(
        random_state=(
            MODEL_RANDOM_STATE
            + fold_id
        )
    )

    gate_model.fit(
        X_gate_train,
        y_gate_train,
    )

    gate_training_time = (
        time.perf_counter()
        - gate_start
    )

    FINAL_MODELS[fold_id] = {
        "global": global_model,
        "experts": experts,
        "gate": gate_model,
    }

    CV_FOLD_DATA[
        fold_id
    ]["trained_models"] = (
        FINAL_MODELS[fold_id]
    )

    fold_total_time = (
        time.perf_counter()
        - fold_start
    )

    gate_class_counts = (
        y_gate_train
        .value_counts()
        .to_dict()
    )

    training_summary_rows.append(
        {
            "fold": fold_id,

            "n_train_balanced":
                len(train_balanced_fold),

            "n_train_real":
                int(
                    train_balanced_fold[
                        "sample_origin"
                    ]
                    .eq("real")
                    .sum()
                ),

            "n_train_smoter":
                int(
                    train_balanced_fold[
                        "sample_origin"
                    ]
                    .eq("smoter")
                    .sum()
                ),

            "n_validation_real":
                len(
                    validation_real_fold
                ),

            "n_global_features":
                len(FEATURES_GLOBAL),

            "n_expert_features":
                len(FEATURES_EXPERTS),

            "n_gate_features":
                len(FEATURES_GATE),

            "n_expert_A":
                expert_training_sizes[0],

            "n_expert_B":
                expert_training_sizes[1],

            "n_expert_C":
                expert_training_sizes[2],

            "n_gate_class_A":
                int(
                    gate_class_counts.get(
                        0,
                        0,
                    )
                ),

            "n_gate_class_B":
                int(
                    gate_class_counts.get(
                        1,
                        0,
                    )
                ),

            "n_gate_class_C":
                int(
                    gate_class_counts.get(
                        2,
                        0,
                    )
                ),

            "tempo_global_s":
                global_training_time,

            "tempo_expert_A_s":
                expert_training_times[0],

            "tempo_expert_B_s":
                expert_training_times[1],

            "tempo_expert_C_s":
                expert_training_times[2],

            "tempo_experts_total_s":
                experts_training_time,

            "tempo_gate_s":
                gate_training_time,

            "tempo_total_fold_s":
                fold_total_time,
        }
    )


training_summary = pd.DataFrame(
    training_summary_rows
)

expert_training_summary = pd.DataFrame(
    expert_training_summary_rows
)

# %% avaliação
# avaliação
fold_metric_rows = []
gate_metric_rows = []
expert_external_metric_rows = []
prediction_frames = []
routing_distribution_rows = []

inference_total_start = time.perf_counter()

ARCHITECTURE_PREDICTION_COLUMNS = {
    "ExtraTrees_Global_Mileage":
        "pred_global",

    "MoE_Gate_XGBoost_Experts_Mileage":
        "pred_moe_gate",

    "MoE_Router_ExtraTrees_Global":
        "pred_moe_global_router",

    "Oracle_Experts_Mileage":
        "pred_oracle",
}

for fold_id in range(N_SPLITS):

    fold_start = time.perf_counter()

    print("\n" + "=" * 110)
    print(f"INFERÊNCIA EXTERNA — FOLD {fold_id}")
    print("=" * 110)

    validation_real_fold = (
        CV_FOLD_DATA[fold_id][
            "validation_real"
        ]
        .copy()
        .reset_index(drop=True)
    )

    fold_models = FINAL_MODELS[fold_id]

    global_model = (
        fold_models["global"]
    )

    expert_models = (
        fold_models["experts"]
    )

    gate_model = (
        fold_models["gate"]
    )

    n_validation = len(
        validation_real_fold
    )

    y_true = (
        validation_real_fold[
            TARGET_COL
        ]
        .to_numpy(dtype=float)
    )

    if not validation_real_fold[
        "sample_origin"
    ].eq("real").all():

        raise ValueError(
            f"Fold {fold_id}: a validação externa "
            "contém eventos sintéticos."
        )

    global_prediction_start = (
        time.perf_counter()
    )

    pred_global = np.asarray(
        global_model.predict(
            validation_real_fold[
                FEATURES_GLOBAL
            ]
        ),
        dtype=float,
    )

    global_prediction_time = (
        time.perf_counter()
        - global_prediction_start
    )

    expert_prediction_matrix = np.zeros(
        shape=(n_validation, 3),
        dtype=float,
    )

    expert_prediction_times = {}

    for class_id in range(3):

        prediction_start = (
            time.perf_counter()
        )

        expert_prediction_matrix[
            :,
            class_id,
        ] = np.asarray(
            expert_models[
                class_id
            ].predict(
                validation_real_fold[
                    FEATURES_EXPERTS
                ]
            ),
            dtype=float,
        )

        expert_prediction_times[
            class_id
        ] = (
            time.perf_counter()
            - prediction_start
        )

    gate_prediction_start = (
        time.perf_counter()
    )

    gate_pred_class = np.asarray(
        gate_model.predict(
            validation_real_fold[
                FEATURES_GATE
            ]
        ),
        dtype=int,
    )

    gate_pred_proba = np.asarray(
        gate_model.predict_proba(
            validation_real_fold[
                FEATURES_GATE
            ]
        ),
        dtype=float,
    )

    gate_prediction_time = (
        time.perf_counter()
        - gate_prediction_start
    )

    if gate_pred_proba.shape != (
        n_validation,
        3,
    ):
        raise ValueError(
            f"Fold {fold_id}: predict_proba do gate "
            f"possui shape {gate_pred_proba.shape}; "
            f"esperado=({n_validation}, 3)."
        )

    pred_moe_gate = (
        combinar_predicoes_especialistas(
            expert_prediction_matrix,
            gate_pred_class,
        )
    )

    global_router_class = (
        rotear_pela_previsao_global(
            pred_global
        )
    )

    pred_moe_global_router = (
        combinar_predicoes_especialistas(
            expert_prediction_matrix,
            global_router_class,
        )
    )

    oracle_result = (
        calcular_oracle_especialistas(
            y_true=y_true,
            expert_prediction_matrix=(
                expert_prediction_matrix
            ),
        )
    )

    oracle_class = (
        oracle_result["classes"]
    )

    pred_oracle = (
        oracle_result["predictions"]
    )

    prediction_vectors = {
        "pred_global":
            pred_global,

        "pred_expert_A":
            expert_prediction_matrix[:, 0],

        "pred_expert_B":
            expert_prediction_matrix[:, 1],

        "pred_expert_C":
            expert_prediction_matrix[:, 2],

        "pred_moe_gate":
            pred_moe_gate,

        "pred_moe_global_router":
            pred_moe_global_router,

        "pred_oracle":
            pred_oracle,
    }

    for name, values in (
        prediction_vectors.items()
    ):

        if len(values) != n_validation:
            raise ValueError(
                f"Fold {fold_id}: {name} possui "
                f"{len(values)} valores; "
                f"esperado={n_validation}."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Fold {fold_id}: {name} contém "
                "NaN ou Inf."
            )

    for name, classes in {
        "gate_pred_class":
            gate_pred_class,

        "global_router_class":
            global_router_class,

        "oracle_class":
            oracle_class,
    }.items():

        if not np.isin(
            classes,
            [0, 1, 2],
        ).all():

            raise ValueError(
                f"Fold {fold_id}: {name} possui "
                "classes inválidas."
            )

    architecture_predictions = {
        "ExtraTrees_Global_Mileage":
            pred_global,

        "MoE_Gate_XGBoost_Experts_Mileage":
            pred_moe_gate,

        "MoE_Router_ExtraTrees_Global":
            pred_moe_global_router,

        "Oracle_Experts_Mileage":
            pred_oracle,
    }

    for architecture_name, predictions in (
        architecture_predictions.items()
    ):

        architecture_metrics = (
            calcular_metricas_regressao(
                y_true=y_true,
                y_pred=predictions,
            )
        )

        fold_metric_rows.append(
            {
                "fold":
                    fold_id,

                "arquitetura":
                    architecture_name,

                "n_validation":
                    n_validation,

                **architecture_metrics,
            }
        )

    for class_id in range(3):

        expert_name = (
            EXPERT_CLASS_NAMES[
                class_id
            ]
        )

        expert_predictions = (
            expert_prediction_matrix[
                :,
                class_id
            ]
        )

        metrics_all_validation = (
            calcular_metricas_regressao(
                y_true=y_true,
                y_pred=expert_predictions,
            )
        )

        range_mask = (
            obter_mascara_especialista(
                validation_real_fold,
                class_id,
            )
            .to_numpy()
        )

        if range_mask.sum() > 0:

            metrics_expert_range = (
                calcular_metricas_regressao(
                    y_true=y_true[
                        range_mask
                    ],
                    y_pred=expert_predictions[
                        range_mask
                    ],
                )
            )

        else:

            metrics_expert_range = {
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "mape_%": np.nan,
                "bias": np.nan,
                "erro_abs_p95": np.nan,
                "erro_abs_max": np.nan,
            }

        expert_external_metric_rows.append(
            {
                "fold":
                    fold_id,

                "class_id":
                    class_id,

                "especialista":
                    expert_name,

                "n_validation_total":
                    n_validation,

                "n_validation_range":
                    int(
                        range_mask.sum()
                    ),

                "rmse_all":
                    metrics_all_validation[
                        "rmse"
                    ],

                "mae_all":
                    metrics_all_validation[
                        "mae"
                    ],

                "r2_all":
                    metrics_all_validation[
                        "r2"
                    ],

                "rmse_range":
                    metrics_expert_range[
                        "rmse"
                    ],

                "mae_range":
                    metrics_expert_range[
                        "mae"
                    ],

                "r2_range":
                    metrics_expert_range[
                        "r2"
                    ],

                "bias_range":
                    metrics_expert_range[
                        "bias"
                    ],

                "erro_abs_p95_range":
                    metrics_expert_range[
                        "erro_abs_p95"
                    ],

                "tempo_predicao_s":
                    expert_prediction_times[
                        class_id
                    ],
            }
        )

    gate_metrics = (
        calcular_metricas_gate(
            y_true=oracle_class,
            y_pred=gate_pred_class,
            y_proba=gate_pred_proba,
        )
    )

    gate_metric_rows.append(
        {
            "fold":
                fold_id,

            "n_validation":
                n_validation,

            "n_oracle_A":
                int(
                    np.sum(
                        oracle_class == 0
                    )
                ),

            "n_oracle_B":
                int(
                    np.sum(
                        oracle_class == 1
                    )
                ),

            "n_oracle_C":
                int(
                    np.sum(
                        oracle_class == 2
                    )
                ),

            **gate_metrics,
        }
    )

    for routing_name, classes in {
        "Gate_XGBoost":
            gate_pred_class,

        "Router_Global":
            global_router_class,

        "Oracle":
            oracle_class,
    }.items():

        class_counts = pd.Series(
            classes
        ).value_counts()

        for class_id in range(3):

            n_class = int(
                class_counts.get(
                    class_id,
                    0,
                )
            )

            routing_distribution_rows.append(
                {
                    "fold":
                        fold_id,

                    "roteamento":
                        routing_name,

                    "class_id":
                        class_id,

                    "especialista":
                        EXPERT_CLASS_NAMES[
                            class_id
                        ],

                    "n_eventos":
                        n_class,

                    "percentual_%":
                        100
                        * n_class
                        / n_validation,
                }
            )

    prediction_columns_metadata = [
        column
        for column in [
            "event_id",
            "car",
            "mileage",
            "charge_segment",
            "soh_bin",
            "sample_origin",
            TARGET_COL,
        ]
        if column
        in validation_real_fold.columns
    ]

    fold_predictions_df = (
        validation_real_fold[
            prediction_columns_metadata
        ]
        .copy()
        .reset_index(drop=True)
    )

    fold_predictions_df.insert(
        0,
        "fold",
        fold_id,
    )

    fold_predictions_df = (
        fold_predictions_df.rename(
            columns={
                TARGET_COL: "soh_real"
            }
        )
    )

    fold_predictions_df[
        "pred_global"
    ] = pred_global

    fold_predictions_df[
        "pred_expert_A"
    ] = expert_prediction_matrix[:, 0]

    fold_predictions_df[
        "pred_expert_B"
    ] = expert_prediction_matrix[:, 1]

    fold_predictions_df[
        "pred_expert_C"
    ] = expert_prediction_matrix[:, 2]

    fold_predictions_df[
        "gate_pred_class"
    ] = gate_pred_class

    fold_predictions_df[
        "gate_pred_expert"
    ] = pd.Series(
        gate_pred_class
    ).map(
        EXPERT_CLASS_NAMES
    )

    fold_predictions_df[
        "gate_proba_A"
    ] = gate_pred_proba[:, 0]

    fold_predictions_df[
        "gate_proba_B"
    ] = gate_pred_proba[:, 1]

    fold_predictions_df[
        "gate_proba_C"
    ] = gate_pred_proba[:, 2]

    fold_predictions_df[
        "gate_max_probability"
    ] = np.max(
        gate_pred_proba,
        axis=1,
    )

    fold_predictions_df[
        "pred_moe_gate"
    ] = pred_moe_gate

    fold_predictions_df[
        "global_router_class"
    ] = global_router_class

    fold_predictions_df[
        "global_router_expert"
    ] = pd.Series(
        global_router_class
    ).map(
        EXPERT_CLASS_NAMES
    )

    fold_predictions_df[
        "pred_moe_global_router"
    ] = pred_moe_global_router

    fold_predictions_df[
        "oracle_class"
    ] = oracle_class

    fold_predictions_df[
        "oracle_expert"
    ] = pd.Series(
        oracle_class
    ).map(
        EXPERT_CLASS_NAMES
    )

    fold_predictions_df[
        "pred_oracle"
    ] = pred_oracle

    fold_predictions_df[
        "oracle_expert_margin"
    ] = oracle_result[
        "expert_margin"
    ]

    fold_predictions_df[
        "oracle_confidence"
    ] = oracle_result[
        "oracle_confidence"
    ]

    for architecture_name, pred_col in (
        ARCHITECTURE_PREDICTION_COLUMNS.items()
    ):

        error_col = (
            "error_"
            + architecture_name
        )

        absolute_error_col = (
            "abs_error_"
            + architecture_name
        )

        fold_predictions_df[
            error_col
        ] = (
            fold_predictions_df[
                pred_col
            ]
            - fold_predictions_df[
                "soh_real"
            ]
        )

        fold_predictions_df[
            absolute_error_col
        ] = np.abs(
            fold_predictions_df[
                error_col
            ]
        )


    prediction_frames.append(
        fold_predictions_df
    )

    FINAL_PREDICTIONS[fold_id] = {
        "y_true":
            y_true,

        "pred_global":
            pred_global,

        "expert_prediction_matrix":
            expert_prediction_matrix,

        "gate_pred_class":
            gate_pred_class,

        "gate_pred_proba":
            gate_pred_proba,

        "pred_moe_gate":
            pred_moe_gate,

        "global_router_class":
            global_router_class,

        "pred_moe_global_router":
            pred_moe_global_router,

        "oracle_class":
            oracle_class,

        "pred_oracle":
            pred_oracle,

        "predictions_df":
            fold_predictions_df,
    }

    CV_FOLD_DATA[
        fold_id
    ]["external_predictions"] = (
        FINAL_PREDICTIONS[fold_id]
    )

    fold_metrics_display = (
        pd.DataFrame(
            [
                row
                for row in fold_metric_rows
                if row["fold"] == fold_id
            ]
        )
        .sort_values(
            "rmse"
        )
        .reset_index(drop=True)
    )

    fold_elapsed = (
        time.perf_counter()
        - fold_start
    )

    print("\nMétricas do fold:")    

    print(
        "Tempo de inferência do fold:",
        round(fold_elapsed, 3),
        "s",
    )

    gc.collect()

oof_predictions = pd.concat(
    prediction_frames,
    axis=0,
    ignore_index=True,
)

oof_predictions = (
    oof_predictions
    .sort_values(
        [
            "fold",
            "event_id",
        ]
    )
    .reset_index(drop=True)
)

fold_metrics = pd.DataFrame(
    fold_metric_rows
)

gate_metrics_external = pd.DataFrame(
    gate_metric_rows
)

expert_external_metrics = pd.DataFrame(
    expert_external_metric_rows
)

routing_distribution = pd.DataFrame(
    routing_distribution_rows
)

inference_total_time = (
    time.perf_counter()
    - inference_total_start
)

fold_assignment_check = (
    oof_predictions[
        [
            "event_id",
            "fold",
        ]
    ]
    .merge(
        df_cv[
            [
                "event_id",
                "cv_fold",
            ]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
)

if not fold_assignment_check[
    "fold"
].eq(
    fold_assignment_check[
        "cv_fold"
    ]
).all():

    raise ValueError(
        "Alguma predição OOF foi produzida pelo "
        "fold externo incorreto."
    )

print("\n" + "=" * 110)
print("INFERÊNCIA EXTERNA CONCLUÍDA")
print("=" * 110)

print(
    "Eventos reais com previsão OOF:",
    len(oof_predictions),
)

print(
    "Event IDs únicos:",
    oof_predictions[
        "event_id"
    ].nunique(),
)

print(
    "Tempo total de inferência:",
    round(
        inference_total_time,
        3,
    ),
    "s",
)

print("\nMétricas por fold:")
print("\nMétricas externas do gate:")
print("\nPredições OOF salvas em:")
print(
    PREDICTIONS_DIR
    / "oof_predictions_all_events.csv"
)