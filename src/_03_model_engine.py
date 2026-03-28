"""
_03_model_engine.py
Motor predictivo: Dixon-Coles + XGBoost ensemble con calibración isotónica.

CAMBIOS v2:
  1. Métrica de validación real: ROI sobre set de validación en lugar de
     solo accuracy. El modelo ahora reporta si tiene edge histórico real.
  2. _save_as_latest lanza excepción clara si la copia falla, en lugar
     de dejar el archivo latest desactualizado en silencio.
  3. FEATURE_COLS excluye market_prob_* del training (consistente con
     el fix de leakage en _02_). En inferencia se añaden como features
     adicionales opcionales vía INFERENCE_EXTRA_COLS.
  4. Calibración: umbral MIN_SAMPLES_ISOTONIC reducido a 30 (más realista
     para mercados como "draw" en ligas pequeñas).
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

MIN_SAMPLES_ISOTONIC = 30  # reducido de 50

# Features base para training (sin market_prob — leakage fix)
FEATURE_COLS = [
    "home_xg_scored",    "home_xg_conceded",   "home_goals_scored",
    "home_goals_conceded","home_forma",         "home_btts_rate",
    "home_over25_rate",  "home_corners_avg",   "home_fouls_avg",
    "home_days_rest",    "away_xg_scored",     "away_xg_conceded",
    "away_goals_scored", "away_goals_conceded", "away_forma",
    "away_btts_rate",    "away_over25_rate",   "away_corners_avg",
    "away_fouls_avg",    "away_days_rest",      "h2h_home_wins",
    "h2h_draws",         "h2h_away_wins",       "h2h_avg_goals",
    "h2h_btts_rate",     "elo_diff",            "xg_diff",
    "xg_total_exp",      "goals_diff",          "forma_diff",
    "rest_diff",         "fatiga_flag",
    "rain_flag",         "wind_flag",
]

# Features adicionales disponibles solo en inferencia
INFERENCE_EXTRA_COLS = [
    "market_prob_home", "market_prob_draw", "market_prob_away",
]


# ─── DIXON-COLES ─────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Modelo Dixon-Coles con SLSQP para respetar la restricción de
    identificabilidad (suma de parámetros de ataque = 0).
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
        log.info("Entrenando Dixon-Coles (SLSQP)...")
        required = ["home_team_norm", "away_team_norm", "home_goals", "away_goals"]
        if not all(c in df.columns for c in required):
            log.error(f"Columnas requeridas para DC: {required}")
            return self

        df = df.dropna(subset=required)
        home_teams = df["home_team_norm"].tolist()
        away_teams = df["away_team_norm"].tolist()
        home_goals = df["home_goals"].astype(int).tolist()
        away_goals = df["away_goals"].astype(int).tolist()

        teams = sorted(set(home_teams + away_teams))
        n     = len(teams)
        x0    = np.zeros(2 * n + 2)
        x0[2*n]     =  0.1
        x0[2*n + 1] = -0.1

        constraints = [{"type": "eq", "fun": lambda x, n=n: np.sum(x[:n])}]

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
            f"Dixon-Coles: {n} equipos | home_adv={self.home_adv:.3f} | "
            f"rho={self.rho:.3f} | convergió={result.success}"
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

        mu  = np.exp(
            self.attack.get(h, mean_att) + self.defense.get(a, mean_def) + self.home_adv
        )
        lam = np.exp(
            self.attack.get(a, mean_att) + self.defense.get(h, mean_def)
        )

        M = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                M[i][j] = (
                    self._tau(i, j, mu, lam, self.rho)
                    * poisson.pmf(i, mu)
                    * poisson.pmf(j, lam)
                )
        total = M.sum()
        if total > 0:
            M /= total

        p_home   = float(np.tril(M, -1).sum())
        p_draw   = float(np.diag(M).sum())
        p_away   = float(np.triu(M, 1).sum())
        p_btts   = float(M[1:, 1:].sum())
        p_over25 = float(
            sum(M[i][j] for i in range(max_goals+1)
                for j in range(max_goals+1) if i+j > 2.5)
        )

        return {
            "dc_prob_home":       round(max(p_home,   0.01), 4),
            "dc_prob_draw":       round(max(p_draw,   0.01), 4),
            "dc_prob_away":       round(max(p_away,   0.01), 4),
            "dc_prob_btts":       round(max(p_btts,   0.01), 4),
            "dc_prob_over25":     round(max(p_over25, 0.01), 4),
            "dc_exp_home_goals":  round(mu,       3),
            "dc_exp_away_goals":  round(lam,      3),
            "dc_exp_total_goals": round(mu + lam, 3),
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
    IsotonicRegression si hay suficientes positivos, Platt scaling si no.
    """
    n_pos = int(y.sum())
    if n_pos >= MIN_SAMPLES_ISOTONIC:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_probs, y)
        return ir

    log.info(f"  Pocas muestras positivas ({n_pos}), usando Platt scaling.")

    class PlattWrapper:
        def __init__(self):
            self.lr = LogisticRegression(C=1.0, max_iter=1000)

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


# ─── VALIDACIÓN CON ROI REAL ─────────────────────────────────────────────────

def _compute_validation_roi(
    cal_probs: np.ndarray,
    y_true: np.ndarray,
    reference_odds: float = None,
) -> dict:
    """
    Calcula métricas de validación orientadas a apuestas:
    - accuracy: % de predicciones correctas (prob > 0.5)
    - roi_flat: ROI apostando 1 unidad siempre que prob > 0.5
    - brier: Brier score (calibración)
    - edge_mean: edge promedio del modelo vs odds de referencia

    Si reference_odds es None, se usa la inversa de la prob promedio
    como proxy de cuota justa del mercado.
    """
    preds    = (cal_probs > 0.5).astype(int)
    accuracy = float((preds == y_true).mean())
    brier    = float(np.mean((cal_probs - y_true) ** 2))

    # ROI flat: apuesta 1 unidad cuando modelo dice SÍ
    bet_mask = preds == 1
    if bet_mask.sum() > 0:
        if reference_odds is not None:
            gains   = np.where(y_true[bet_mask], reference_odds - 1, -1)
        else:
            # Cuota justa aproximada: 1 / prob_promedio_del_mercado
            avg_prob = y_true.mean() if y_true.mean() > 0 else 0.33
            fair_odds = 1 / avg_prob * 0.95  # 5% overround
            gains     = np.where(y_true[bet_mask], fair_odds - 1, -1)
        roi_flat = float(gains.mean() * 100)
    else:
        roi_flat = 0.0

    return {
        "accuracy":  round(accuracy, 3),
        "roi_flat":  round(roi_flat, 2),
        "brier":     round(brier,    4),
        "n_bets":    int(bet_mask.sum()),
        "n_total":   int(len(y_true)),
    }


# ─── XGBOOST ENSEMBLE ────────────────────────────────────────────────────────

class FootbotEnsemble:
    MARKETS = ["home_win", "draw", "away_win", "btts", "over25"]

    def __init__(self):
        self.models              = {}
        self.calibrators         = {}
        self.feature_importances = {}
        self.validation_metrics  = {}   # NUEVO: métricas de validación por mercado
        self.fitted_features     = []
        self.fitted              = False

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

        # Solo features sin leakage para training
        available = [f for f in FEATURE_COLS if f in df.columns]
        X = df[available].fillna(0).values
        self.fitted_features = available

        # Cuotas de referencia promedio para ROI (de B365 si existe)
        ref_odds_map = {}
        if "B365H" in df.columns:
            ref_odds_map["home_win"] = df["B365H"].dropna().median()
        if "B365D" in df.columns:
            ref_odds_map["draw"]     = df["B365D"].dropna().median()
        if "B365A" in df.columns:
            ref_odds_map["away_win"] = df["B365A"].dropna().median()

        # Split temporal 80/20
        split = int(len(X) * 0.8)

        for market, target_col in targets.items():
            if target_col not in df.columns:
                log.warning(f"Target {target_col} no disponible.")
                continue

            y = df[target_col].values
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            log.info(
                f"Entrenando XGBoost [{market}]: "
                f"{int(y.sum())} positivos de {len(y)}"
            )

            xgb = self._get_xgb()
            xgb.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)

            raw_val = xgb.predict_proba(X_val)[:, 1]
            cal     = _fit_calibrator(raw_val, y_val)
            cal_val = cal.transform(raw_val)

            # Métricas de validación reales (accuracy + ROI)
            ref_odds = ref_odds_map.get(market)
            metrics  = _compute_validation_roi(cal_val, y_val, ref_odds)
            self.validation_metrics[market] = metrics

            log.info(
                f"  [{market}] accuracy={metrics['accuracy']:.3f} | "
                f"ROI flat={metrics['roi_flat']:+.1f}% "
                f"({metrics['n_bets']}/{metrics['n_total']} apuestas)"
            )

            # Advertencia si el ROI es negativo en validación
            if metrics["roi_flat"] < -5:
                log.warning(
                    f"  ⚠ [{market}] ROI negativo en validación. "
                    f"El edge puede no materializarse."
                )

            self.models[market]      = xgb
            self.calibrators[market] = cal

            fi = pd.Series(xgb.feature_importances_, index=available)
            self.feature_importances[market] = fi.sort_values(ascending=False).head(10)

        self.fitted = True
        log.info("Ensemble entrenado. Resumen de validación:")
        for market, m in self.validation_metrics.items():
            log.info(
                f"  {market:10s}: acc={m['accuracy']:.3f} | "
                f"roi={m['roi_flat']:+.1f}%"
            )
        return self

    def predict(self, features_row: dict) -> dict:
        """
        Predice probabilidades calibradas para un partido.
        Acepta features extra de inferencia (market_prob_*) si están presentes,
        pero solo los usa si el modelo fue entrenado con ellos (no debería).
        """
        if not self.fitted:
            return {}
        X = np.array([[features_row.get(f, 0) for f in self.fitted_features]])
        result = {}
        for market, xgb in self.models.items():
            raw = xgb.predict_proba(X)[0, 1]
            cal = float(self.calibrators[market].transform(np.array([raw]))[0])
            result[f"xgb_prob_{market}"] = round(cal, 4)
        return result

    def get_top_features(self, market: str, n: int = 5) -> list:
        if market not in self.feature_importances:
            return []
        return self.feature_importances[market].head(n).index.tolist()

    def get_validation_summary(self) -> str:
        """Retorna un resumen legible de las métricas de validación."""
        lines = ["📊 *Validación del modelo (set temporal 20%)*"]
        for market, m in self.validation_metrics.items():
            emoji = "✅" if m["roi_flat"] >= 0 else "⚠️"
            lines.append(
                f"{emoji} {market}: acc `{m['accuracy']:.1%}` | "
                f"ROI `{m['roi_flat']:+.1f}%` ({m['n_bets']} apuestas)"
            )
        return "\n".join(lines)


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
        blended[f"prob_{market}"] = (
            round(dc_weight * dp + xw * xp, 4) if xp > 0 else round(dp, 4)
        )

    # Normalizar 1X2
    s = sum(blended.get(f"prob_{m}", 0) for m in ["home_win", "draw", "away_win"])
    if s > 0:
        for m in ["home_win", "draw", "away_win"]:
            blended[f"prob_{m}"] = round(blended[f"prob_{m}"] / s, 4)
    return blended


# ─── PERSISTENCIA ────────────────────────────────────────────────────────────

def _save_as_latest(src: str, name: str):
    """
    Copia el modelo versionado como 'latest'.
    Lanza RuntimeError si la copia falla — así el scheduler lo detecta
    y no usa un modelo desactualizado en silencio.
    """
    dst = os.path.join(MODELS_DIR, f"{name}_latest.pkl")
    try:
        shutil.copy2(src, dst)
        log.info(f"Modelo latest actualizado: {dst}")
    except Exception as e:
        raise RuntimeError(
            f"No se pudo guardar modelo latest '{dst}': {e}"
        ) from e


def train_and_save(training_df: pd.DataFrame = None) -> tuple:
    if training_df is None:
        path = os.path.join(DATA_PROCESSED, "training_dataset.csv")
        if not os.path.exists(path):
            log.error("No hay dataset de entrenamiento.")
            return None, None
        training_df = pd.read_csv(path)

    log.info(f"Entrenando con {len(training_df)} partidos históricos...")

    # Normalización para Dixon-Coles
    if "home_team_norm" not in training_df.columns:
        from src._02_feature_builder import normalize_team_name
        training_df = training_df.copy()
        training_df["home_team_norm"] = training_df["home_team"].apply(normalize_team_name)
        training_df["away_team_norm"] = training_df["away_team"].apply(normalize_team_name)
    if "home_goals" not in training_df.columns and "home_goals_actual" in training_df.columns:
        training_df = training_df.copy()
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

    # Estas líneas pueden lanzar RuntimeError — el scheduler lo captura
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
    dc_probs  = dc.predict_proba(
        normalize_team_name(home_team),
        normalize_team_name(away_team),
    )
    xgb_probs = ensemble.predict(features_row)
    blended   = blend_predictions(dc_probs, xgb_probs)

    top_features = {
        m: ensemble.get_top_features(m, n=3)
        for m in ["home_win", "btts", "over25"]
    }
    return {
        **blended,
        **dc_probs,
        "top_features":       top_features,
        "dc_exp_home_goals":  dc_probs.get("dc_exp_home_goals",  0),
        "dc_exp_away_goals":  dc_probs.get("dc_exp_away_goals",  0),
        "dc_exp_total_goals": dc_probs.get("dc_exp_total_goals", 0),
    }