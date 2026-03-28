"""
_03_model_engine.py
Motor predictivo: Dixon-Coles + XGBoost ensemble con calibración isotónica.

Correcciones respecto a versión anterior:
  - Dixon-Coles usa SLSQP (respeta la restricción de identificabilidad)
  - Modelos se guardan con shutil.copy2 en lugar de os.symlink
    (symlinks fallan en Oracle Cloud con algunos filesystems)
  - Platt scaling como alternativa a IsotonicRegression cuando hay pocas muestras
"""

import os
import sys
import shutil
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_PROCESSED, MODELS_DIR, RANDOM_SEED,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Mínimo de samples positivos para usar IsotonicRegression
# Con menos, se usa Platt scaling (LogisticRegression)
MIN_SAMPLES_ISOTONIC = 50

FEATURE_COLS = [
    "home_xg_scored",   "home_xg_conceded",  "home_goals_scored",
    "home_goals_conceded","home_forma",       "home_btts_rate",
    "home_over25_rate", "home_corners_avg",  "home_fouls_avg",
    "home_days_rest",   "away_xg_scored",    "away_xg_conceded",
    "away_goals_scored","away_goals_conceded","away_forma",
    "away_btts_rate",   "away_over25_rate",  "away_corners_avg",
    "away_fouls_avg",   "away_days_rest",    "h2h_home_wins",
    "h2h_draws",        "h2h_away_wins",     "h2h_avg_goals",
    "h2h_btts_rate",    "elo_diff",          "xg_diff",
    "xg_total_exp",     "goals_diff",        "forma_diff",
    "rest_diff",        "fatiga_flag",
    "market_prob_home", "market_prob_draw",  "market_prob_away",
    "rain_flag",        "wind_flag",
]


# ─── DIXON-COLES (CORREGIDO) ─────────────────────────────────────────────────

class DixonColesModel:
    """
    Modelo Dixon-Coles para predicción de goles.

    FIX CRÍTICO: La versión anterior usaba L-BFGS-B que ignora constraints.
    Ahora se usa SLSQP que sí respeta la restricción de identificabilidad
    (suma de parámetros de ataque = 0).
    """

    def __init__(self):
        self.attack   = {}
        self.defense  = {}
        self.home_adv = 0.0
        self.rho      = -0.1
        self.fitted   = False

    def _tau(self, x: int, y: int, mu: float, lam: float, rho: float) -> float:
        if   x == 0 and y == 0: return 1 - mu * lam * rho
        elif x == 0 and y == 1: return 1 + mu * rho
        elif x == 1 and y == 0: return 1 + lam * rho
        elif x == 1 and y == 1: return 1 - rho
        return 1.0

    def _neg_log_likelihood(self, params, teams, home_teams, away_teams,
                            home_goals, away_goals):
        n        = len(teams)
        attack   = dict(zip(teams, params[:n]))
        defense  = dict(zip(teams, params[n:2*n]))
        home_adv = params[2*n]
        rho      = params[2*n + 1]

        ll = 0.0
        for i in range(len(home_teams)):
            h, a   = home_teams[i], away_teams[i]
            hg, ag = home_goals[i], away_goals[i]
            if h not in attack or a not in attack:
                continue
            mu  = np.exp(attack[h]  + defense[a] + home_adv)
            lam = np.exp(attack[a]  + defense[h])
            tau = self._tau(hg, ag, mu, lam, rho)
            if tau <= 0 or mu <= 0 or lam <= 0:
                continue
            ll += np.log(tau) + poisson.logpmf(hg, mu) + poisson.logpmf(ag, lam)
        return -ll

    def fit(self, df: pd.DataFrame):
        log.info("Entrenando Dixon-Coles (SLSQP con restricción de identificabilidad)...")
        required = ["home_team_norm", "away_team_norm", "home_goals", "away_goals"]
        if not all(c in df.columns for c in required):
            log.error(f"Columnas requeridas: {required}")
            return self

        df = df.dropna(subset=required)
        home_teams = df["home_team_norm"].tolist()
        away_teams = df["away_team_norm"].tolist()
        home_goals = df["home_goals"].astype(int).tolist()
        away_goals = df["away_goals"].astype(int).tolist()

        teams = sorted(set(home_teams + away_teams))
        n     = len(teams)

        x0 = np.zeros(2 * n + 2)
        x0[2*n]     =  0.1   # home advantage inicial
        x0[2*n + 1] = -0.1   # rho inicial

        # CORRECCIÓN: usar SLSQP para respetar la restricción de igualdad
        constraints = [{
            "type": "eq",
            "fun":  lambda x, n=n: np.sum(x[:n]),  # sum(attack) = 0
        }]

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(teams, home_teams, away_teams, home_goals, away_goals),
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-7},
        )

        params        = result.x
        self.attack   = dict(zip(teams, params[:n]))
        self.defense  = dict(zip(teams, params[n:2*n]))
        self.home_adv = float(params[2*n])
        self.rho      = float(params[2*n + 1])
        self.teams    = teams
        self.fitted   = True

        log.info(
            f"Dixon-Coles listo: {n} equipos | "
            f"home_adv={self.home_adv:.3f} | rho={self.rho:.3f} | "
            f"convergió={result.success}"
        )
        return self

    def predict_proba(self, home_team: str, away_team: str,
                      max_goals: int = 8) -> dict:
        if not self.fitted:
            return self._default_proba()

        h = home_team.lower().strip()
        a = away_team.lower().strip()

        mean_att = np.mean(list(self.attack.values()))  if self.attack  else 0.0
        mean_def = np.mean(list(self.defense.values())) if self.defense else 0.0

        mu  = np.exp(self.attack.get(h, mean_att) + self.defense.get(a, mean_def) + self.home_adv)
        lam = np.exp(self.attack.get(a, mean_att) + self.defense.get(h, mean_def))

        M = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                M[i][j] = self._tau(i, j, mu, lam, self.rho) * poisson.pmf(i, mu) * poisson.pmf(j, lam)
        total = M.sum()
        if total > 0:
            M /= total

        p_home   = float(np.tril(M, -1).sum())
        p_draw   = float(np.diag(M).sum())
        p_away   = float(np.triu(M, 1).sum())
        p_btts   = float(M[1:, 1:].sum())
        p_over25 = float(sum(M[i][j] for i in range(max_goals+1)
                              for j in range(max_goals+1) if i+j > 2.5))

        return {
            "dc_prob_home":      round(max(p_home,   0.01), 4),
            "dc_prob_draw":      round(max(p_draw,   0.01), 4),
            "dc_prob_away":      round(max(p_away,   0.01), 4),
            "dc_prob_btts":      round(max(p_btts,   0.01), 4),
            "dc_prob_over25":    round(max(p_over25, 0.01), 4),
            "dc_exp_home_goals": round(mu,       3),
            "dc_exp_away_goals": round(lam,      3),
            "dc_exp_total_goals":round(mu + lam, 3),
        }

    def _default_proba(self):
        return {
            "dc_prob_home": 0.45, "dc_prob_draw": 0.27, "dc_prob_away": 0.28,
            "dc_prob_btts": 0.50, "dc_prob_over25": 0.50,
            "dc_exp_home_goals": 1.4, "dc_exp_away_goals": 1.1,
            "dc_exp_total_goals": 2.5,
        }


# ─── CALIBRACIÓN ROBUSTA ─────────────────────────────────────────────────────

def _fit_calibrator(raw_probs: np.ndarray, y: np.ndarray):
    """
    Elige el calibrador según la cantidad de muestras positivas:
    - IsotonicRegression si hay suficientes (+)
    - Platt scaling (LogisticRegression) con pocas muestras
    Retorna un objeto con método .transform(x) → array
    """
    n_pos = int(y.sum())

    if n_pos >= MIN_SAMPLES_ISOTONIC:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_probs, y)
        return ir
    else:
        log.info(f"  Pocas muestras positivas ({n_pos}), usando Platt scaling.")

        class PlattWrapper:
            def __init__(self):
                self.lr = LogisticRegression(C=1.0)

            def fit(self, x, y):
                self.lr.fit(x.reshape(-1, 1), y)
                return self

            def transform(self, x):
                return self.lr.predict_proba(
                    np.array(x).reshape(-1, 1)
                )[:, 1]

        pw = PlattWrapper()
        pw.fit(raw_probs, y)
        return pw


# ─── XGBOOST ENSEMBLE ────────────────────────────────────────────────────────

class FootbotEnsemble:
    MARKETS = ["home_win", "draw", "away_win", "btts", "over25"]

    def __init__(self):
        self.models               = {}
        self.calibrators          = {}
        self.feature_importances  = {}
        self.fitted_features      = []
        self.fitted               = False

    def _get_xgb(self):
        return XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, df: pd.DataFrame):
        targets = {
            "home_win": "target_home_win",
            "draw":     "target_draw",
            "away_win": "target_away_win",
            "btts":     "target_btts",
            "over25":   "target_over25",
        }

        available = [f for f in FEATURE_COLS if f in df.columns]
        X = df[available].fillna(0).values
        self.fitted_features = available

        split = int(len(X) * 0.8)

        for market, target_col in targets.items():
            if target_col not in df.columns:
                log.warning(f"Target {target_col} no disponible, omitiendo.")
                continue

            y = df[target_col].values
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            log.info(f"Entrenando XGBoost [{market}]: "
                     f"{int(y.sum())} positivos de {len(y)}")

            xgb = self._get_xgb()
            xgb.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)

            raw_val = xgb.predict_proba(X_val)[:, 1]
            cal     = _fit_calibrator(raw_val, y_val)

            cal_probs = cal.transform(raw_val)
            acc = ((cal_probs > 0.5).astype(int) == y_val).mean()
            log.info(f"  accuracy validación: {acc:.3f}")

            self.models[market]      = xgb
            self.calibrators[market] = cal

            fi = pd.Series(xgb.feature_importances_, index=available)
            self.feature_importances[market] = fi.sort_values(ascending=False).head(10)

        self.fitted = True
        log.info("Ensemble entrenado.")
        return self

    def predict(self, features_row: dict) -> dict:
        if not self.fitted:
            return {}
        X = np.array([[features_row.get(f, 0) for f in self.fitted_features]])
        result = {}
        for market, xgb in self.models.items():
            raw  = xgb.predict_proba(X)[0, 1]
            cal  = float(self.calibrators[market].transform(
                np.array([raw])
            )[0])
            result[f"xgb_prob_{market}"] = round(cal, 4)
        return result

    def get_top_features(self, market: str, n: int = 5) -> list:
        if market not in self.feature_importances:
            return []
        return self.feature_importances[market].head(n).index.tolist()


# ─── BLEND ───────────────────────────────────────────────────────────────────

def blend_predictions(dc_probs: dict, xgb_probs: dict,
                      dc_weight: float = 0.35) -> dict:
    xw = 1 - dc_weight
    mapping = {
        "home_win": ("dc_prob_home",   "xgb_prob_home_win"),
        "draw":     ("dc_prob_draw",   "xgb_prob_draw"),
        "away_win": ("dc_prob_away",   "xgb_prob_away_win"),
        "btts":     ("dc_prob_btts",   "xgb_prob_btts"),
        "over25":   ("dc_prob_over25", "xgb_prob_over25"),
    }
    blended = {}
    for market, (dk, xk) in mapping.items():
        dp = dc_probs.get(dk, 0.33)
        xp = xgb_probs.get(xk, dp)
        blended[f"prob_{market}"] = round(dc_weight * dp + xw * xp, 4) if xp > 0 else round(dp, 4)

    # Normalizar 1X2
    s = sum(blended.get(f"prob_{m}", 0) for m in ["home_win", "draw", "away_win"])
    if s > 0:
        for m in ["home_win", "draw", "away_win"]:
            blended[f"prob_{m}"] = round(blended[f"prob_{m}"] / s, 4)
    return blended


# ─── PERSISTENCIA (sin symlinks) ─────────────────────────────────────────────

def _save_as_latest(src: str, name: str):
    """
    Copia el modelo versionado como 'latest'.
    Usa shutil.copy2 en lugar de os.symlink para compatibilidad con
    Oracle Cloud y otros filesystems que no soportan symlinks.
    """
    dst = os.path.join(MODELS_DIR, f"{name}_latest.pkl")
    shutil.copy2(src, dst)
    log.info(f"Modelo guardado: {dst}")


def train_and_save(training_df: pd.DataFrame = None) -> tuple:
    if training_df is None:
        path = os.path.join(DATA_PROCESSED, "training_dataset.csv")
        if not os.path.exists(path):
            log.error("No hay dataset de entrenamiento.")
            return None, None
        training_df = pd.read_csv(path)

    log.info(f"Entrenando con {len(training_df)} partidos históricos...")

    if "home_team_norm" not in training_df.columns:
        from src._02_feature_builder import normalize_team_name
        training_df["home_team_norm"] = training_df["home_team"].apply(normalize_team_name)
        training_df["away_team_norm"] = training_df["away_team"].apply(normalize_team_name)
    if "home_goals" not in training_df.columns and "home_goals_actual" in training_df.columns:
        training_df["home_goals"] = training_df["home_goals_actual"]
        training_df["away_goals"] = training_df["away_goals_actual"]

    dc       = DixonColesModel().fit(training_df)
    ensemble = FootbotEnsemble().fit(training_df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    version  = date.today().strftime("%Y%m%d")
    dc_path  = os.path.join(MODELS_DIR, f"dixon_coles_{version}.pkl")
    xgb_path = os.path.join(MODELS_DIR, f"ensemble_{version}.pkl")

    joblib.dump(dc,       dc_path)
    joblib.dump(ensemble, xgb_path)
    _save_as_latest(dc_path,  "dixon_coles")
    _save_as_latest(xgb_path, "ensemble")

    log.info(f"Modelos guardados: versión {version}")
    return dc, ensemble


def load_models() -> tuple:
    dc_path  = os.path.join(MODELS_DIR, "dixon_coles_latest.pkl")
    xgb_path = os.path.join(MODELS_DIR, "ensemble_latest.pkl")
    if not os.path.exists(dc_path) or not os.path.exists(xgb_path):
        log.warning("No hay modelos guardados. Entrenando desde cero...")
        return train_and_save()
    dc       = joblib.load(dc_path)
    ensemble = joblib.load(xgb_path)
    log.info("Modelos cargados desde disco.")
    return dc, ensemble


def predict_match(home_team: str, away_team: str,
                  features_row: dict,
                  dc: DixonColesModel,
                  ensemble: FootbotEnsemble) -> dict:
    from src._02_feature_builder import normalize_team_name
    dc_probs  = dc.predict_proba(normalize_team_name(home_team),
                                  normalize_team_name(away_team))
    xgb_probs = ensemble.predict(features_row)
    blended   = blend_predictions(dc_probs, xgb_probs)

    top_features = {m: ensemble.get_top_features(m, n=3)
                    for m in ["home_win", "btts", "over25"]}
    return {
        **blended, **dc_probs,
        "top_features":       top_features,
        "dc_exp_home_goals":  dc_probs.get("dc_exp_home_goals",  0),
        "dc_exp_away_goals":  dc_probs.get("dc_exp_away_goals",  0),
        "dc_exp_total_goals": dc_probs.get("dc_exp_total_goals", 0),
    }