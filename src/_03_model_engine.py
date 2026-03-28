"""
_03_model_engine.py
Motor predictivo: Dixon-Coles por liga + XGBoost ensemble con calibración isotónica.

CAMBIOS v4:
  1. Dixon-Coles por liga: en lugar de un modelo global que mezcla equipos
     de 7 ligas distintas, se entrena un modelo DC separado por liga.
     predict_proba recibe el league_id para usar el modelo correcto,
     con fallback al modelo global si la liga no tiene suficientes datos.
  2. blend_predictions ahora usa pesos optimizados por mercado, calculados
     en validación temporal (minimizando Brier score). Los pesos se persisten
     en disco junto con los modelos y se cargan automáticamente.
  3. FEATURE_COLS excluye market_prob_* del training (leakage fix).
  4. Calibración: umbral MIN_SAMPLES_ISOTONIC reducido a 30.

CAMBIOS v5 (fix):
  1. Optimizador SLSQP reemplazado por L-BFGS-B — mucho más rápido en alta
     dimensionalidad (163+ equipos). SLSQP ciclaba indefinidamente.
  2. Eliminado el constraint de suma cero (incompatible con L-BFGS-B y no
     crítico para la convergencia del modelo).
  3. Guard en DC global: si hay más de MAX_TEAMS_GLOBAL_DC equipos, se omite
     el modelo global para evitar inestabilidad numérica.
  4. maxfun=15000 como hard stop de seguridad.

CAMBIOS v6 (fix league_id lookup):
  1. DixonColesEnsemble.predict_proba: las ligas ESPN se guardan con
     league_name como clave (no league_id numérico) porque no están en LIGAS
     (que solo tiene las 7 ligas EU). Se añade un segundo paso de lookup que
     resuelve league_id numérico → league_name usando LIGAS_ESPN y
     COMPETICIONES_NACIONALES_ESPN, evitando que col.1, arg.1, etc. caigan
     siempre al modelo global con probabilidades por defecto.
  2. FootbotEnsemble.fit: el mismo fix se aplica al pre-cálculo de probs DC
     en el set de validación para la optimización de blend weights.
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
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_PROCESSED, MODELS_DIR, RANDOM_SEED,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, LIGAS,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MIN_SAMPLES_ISOTONIC  = 30
MIN_MATCHES_PER_LIGA  = 200    # mínimo para entrenar DC por liga individual
MIN_MATCHES_GLOBAL_DC = 300    # mínimo para entrenar DC global
MAX_TEAMS_GLOBAL_DC   = 80     # si hay más equipos, DC global es inestable
DEFAULT_DC_WEIGHT     = 0.35   # fallback si no hay optimización

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

INFERENCE_EXTRA_COLS = [
    "market_prob_home", "market_prob_draw", "market_prob_away",
]


# ─── HELPER: resolver league_id numérico → league_name ───────────────────────

def _resolve_league_name(league_id: int | str) -> str | None:
    """
    Resuelve un league_id numérico al nombre de liga usado como clave
    en DixonColesEnsemble.models para ligas ESPN/LATAM.

    Las ligas EU tienen clave numérica (ej: 39 → Premier League).
    Las ligas ESPN tienen clave string (ej: 501 → 'Liga BetPlay') porque
    no están en LIGAS y el fit() usa league_name como fallback de clave.

    Retorna el nombre si se encuentra, None si no.
    """
    try:
        from config.settings import LIGAS_ESPN, COMPETICIONES_NACIONALES_ESPN
        _todos = {**LIGAS_ESPN, **COMPETICIONES_NACIONALES_ESPN}
        return next(
            (name for slug, (lid, name) in _todos.items() if lid == league_id),
            None,
        )
    except Exception:
        return None


# ─── DIXON-COLES (por liga) ───────────────────────────────────────────────────

class DixonColesModel:
    """
    Modelo Dixon-Coles con L-BFGS-B (v5).

    Cambio principal: reemplazado SLSQP por L-BFGS-B que es O(n) en memoria
    y converge en segundos incluso con 100+ equipos.
    """

    def __init__(self, league_id: int | str = "global"):
        self.league_id = league_id
        self.attack    = {}
        self.defense   = {}
        self.home_adv  = 0.0
        self.rho       = -0.1
        self.fitted    = False
        self.n_teams   = 0
        self.n_matches = 0

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
        label = f"liga={self.league_id}"
        log.info(f"Entrenando Dixon-Coles ({label})...")

        required = ["home_team_norm", "away_team_norm", "home_goals", "away_goals"]
        if not all(c in df.columns for c in required):
            log.error(f"Columnas requeridas para DC: {required}")
            return self

        df = df.dropna(subset=required)
        if len(df) < MIN_MATCHES_PER_LIGA:
            log.warning(
                f"DC ({label}): solo {len(df)} partidos, "
                f"mínimo requerido {MIN_MATCHES_PER_LIGA}. No se entrena."
            )
            return self

        home_teams = df["home_team_norm"].tolist()
        away_teams = df["away_team_norm"].tolist()
        home_goals = df["home_goals"].astype(int).tolist()
        away_goals = df["away_goals"].astype(int).tolist()

        teams = sorted(set(home_teams + away_teams))
        n     = len(teams)
        x0    = np.zeros(2 * n + 2)
        x0[2*n]     =  0.1   # home_adv inicial
        x0[2*n + 1] = -0.1   # rho inicial

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(teams, home_teams, away_teams, home_goals, away_goals),
            method="L-BFGS-B",
            options={
                "maxiter": 200,
                "ftol":    1e-6,
                "gtol":    1e-5,
                "maxfun":  15000,
            },
        )

        params         = result.x
        self.attack    = dict(zip(teams, params[:n]))
        self.defense   = dict(zip(teams, params[n:2*n]))
        self.home_adv  = float(params[2*n])
        self.rho       = float(params[2*n + 1])
        self.teams     = teams
        self.fitted    = True
        self.n_teams   = n
        self.n_matches = len(df)

        log.info(
            f"DC ({label}): {n} equipos | {len(df)} partidos | "
            f"home_adv={self.home_adv:.3f} | rho={self.rho:.3f} | "
            f"convergió={result.success} | iter={result.nit}"
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


class DixonColesEnsemble:
    """
    Contenedor de modelos DC por liga + modelo global como fallback.

    v5: el modelo global se omite si hay más de MAX_TEAMS_GLOBAL_DC equipos.
    v6: predict_proba resuelve league_id numérico → league_name para ligas
        ESPN que se guardaron con nombre como clave durante el fit().
    """

    def __init__(self):
        self.models: dict[int | str, DixonColesModel] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        n_teams_global = df["home_team_norm"].nunique() if "home_team_norm" in df.columns else 0

        # ── Modelo global (fallback) ──────────────────────────────────────
        log.info("─── DC global (fallback) ───")
        if n_teams_global > MAX_TEAMS_GLOBAL_DC:
            log.warning(
                f"DC global: {n_teams_global} equipos > límite {MAX_TEAMS_GLOBAL_DC}. "
                f"Se omite el modelo global para evitar inestabilidad. "
                f"Las ligas individuales actuarán como fallback entre sí."
            )
            self.models["global"] = DixonColesModel(league_id="global")
        else:
            global_model = DixonColesModel(league_id="global").fit(df)
            self.models["global"] = global_model
            log.info(
                f"DC global: {'OK' if global_model.fitted else 'insuficientes datos'}"
            )

        # ── Modelos por liga ──────────────────────────────────────────────
        if "league_name" in df.columns:
            for league_name, group in df.groupby("league_name"):
                # Intentar mapear a league_id numérico (ligas EU)
                # Para ligas ESPN no está en LIGAS → usa league_name como clave
                lid = next(
                    (lid for lid, (ln, _, _) in LIGAS.items() if ln == league_name),
                    league_name,
                )
                log.info(f"─── DC {league_name} ({len(group)} partidos) ───")
                model = DixonColesModel(league_id=lid).fit(group)
                if model.fitted:
                    self.models[lid] = model
                    log.info(f"  DC {league_name}: OK (clave={repr(lid)})")
                else:
                    log.warning(
                        f"  DC {league_name}: insuficientes datos, "
                        "se usará fallback."
                    )

        self.fitted = True
        fitted_leagues = [k for k in self.models if k != "global" and self.models[k].fitted]
        log.info(
            f"DixonColesEnsemble: {len(fitted_leagues)} ligas entrenadas. "
            f"Ligas: {fitted_leagues}"
        )
        return self

    def predict_proba(self, home_team: str, away_team: str,
                      league_id: int | str = None) -> dict:

        # 1. Buscar por league_id exacto (ligas EU — clave numérica)
        if league_id is not None and league_id in self.models:
            model = self.models[league_id]
            if model.fitted:
                return model.predict_proba(home_team, away_team)

        # 2. Resolver league_id numérico → league_name para ligas ESPN/LATAM
        #    Ej: 501 → 'Liga BetPlay', 502 → 'Liga Profesional Argentina'
        if league_id is not None:
            league_name = _resolve_league_name(league_id)
            if league_name and league_name in self.models:
                model = self.models[league_name]
                if model.fitted:
                    log.debug(
                        f"DC: resuelto league_id={league_id} → '{league_name}'"
                    )
                    return model.predict_proba(home_team, away_team)

        # 3. Fallback al modelo global
        global_model = self.models.get("global")
        if global_model and global_model.fitted:
            return global_model.predict_proba(home_team, away_team)

        # 4. Último fallback: cualquier liga entrenada
        for lid, model in self.models.items():
            if model.fitted:
                log.debug(f"Usando DC de liga {lid} como fallback para {home_team}")
                return model.predict_proba(home_team, away_team)

        return DixonColesModel()._default_proba()

    def get_model_for_league(self, league_id) -> DixonColesModel:
        # Intentar clave directa
        if league_id in self.models:
            return self.models[league_id]
        # Intentar resolución por nombre (ligas ESPN)
        league_name = _resolve_league_name(league_id)
        if league_name and league_name in self.models:
            return self.models[league_name]
        return self.models.get("global")


# ─── CALIBRACIÓN ─────────────────────────────────────────────────────────────

def _fit_calibrator(raw_probs: np.ndarray, y: np.ndarray):
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

    return PlattWrapper().fit(raw_probs, y)


# ─── VALIDACIÓN CON ROI ───────────────────────────────────────────────────────

def _compute_validation_roi(cal_probs, y_true, reference_odds=None):
    preds    = (cal_probs > 0.5).astype(int)
    accuracy = float((preds == y_true).mean())
    brier    = float(np.mean((cal_probs - y_true) ** 2))

    bet_mask = preds == 1
    if bet_mask.sum() > 0:
        if reference_odds is not None:
            gains = np.where(y_true[bet_mask], reference_odds - 1, -1)
        else:
            avg_prob  = y_true.mean() if y_true.mean() > 0 else 0.33
            fair_odds = 1 / avg_prob * 0.95
            gains     = np.where(y_true[bet_mask], fair_odds - 1, -1)
        roi_flat = float(gains.mean() * 100)
    else:
        roi_flat = 0.0

    return {
        "accuracy": round(accuracy, 3),
        "roi_flat": round(roi_flat, 2),
        "brier":    round(brier,    4),
        "n_bets":   int(bet_mask.sum()),
        "n_total":  int(len(y_true)),
    }


# ─── OPTIMIZACIÓN DE BLEND WEIGHT ────────────────────────────────────────────

def _optimize_blend_weight(
    dc_probs_val: np.ndarray,
    xgb_probs_val: np.ndarray,
    y_true: np.ndarray,
    market: str,
) -> float:
    def brier(dc_w):
        blended = dc_w * dc_probs_val + (1 - dc_w) * xgb_probs_val
        return float(np.mean((blended - y_true) ** 2))

    result = minimize_scalar(
        brier,
        bounds=(0.05, 0.70),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 100},
    )

    optimal = float(result.x)
    brier_at_optimal  = brier(optimal)
    brier_at_default  = brier(DEFAULT_DC_WEIGHT)

    log.info(
        f"  Blend [{market}]: dc_weight={optimal:.3f} "
        f"(Brier={brier_at_optimal:.4f} vs default={brier_at_default:.4f})"
    )
    return optimal


# ─── XGBOOST ENSEMBLE ────────────────────────────────────────────────────────

class FootbotEnsemble:
    MARKETS = ["home_win", "draw", "away_win", "btts", "over25"]

    def __init__(self):
        self.models              = {}
        self.calibrators         = {}
        self.feature_importances = {}
        self.validation_metrics  = {}
        self.fitted_features     = []
        self.fitted              = False
        self.blend_weights: dict[str, float] = {}

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

    def fit(self, df: pd.DataFrame, dc_ensemble: "DixonColesEnsemble" = None):
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

        ref_odds_map = {}
        if "B365H" in df.columns:
            ref_odds_map["home_win"] = df["B365H"].dropna().median()
        if "B365D" in df.columns:
            ref_odds_map["draw"]     = df["B365D"].dropna().median()
        if "B365A" in df.columns:
            ref_odds_map["away_win"] = df["B365A"].dropna().median()

        split = int(len(X) * 0.8)

        # Pre-calcular probabilidades DC en el set de validación para blend opt
        dc_val_probs: dict[str, np.ndarray] = {}
        if dc_ensemble is not None:
            log.info("Pre-calculando probs DC en set de validación para blend opt...")
            val_df = df.iloc[split:].reset_index(drop=True)
            dc_mapping = {
                "home_win": "dc_prob_home",
                "draw":     "dc_prob_draw",
                "away_win": "dc_prob_away",
                "btts":     "dc_prob_btts",
                "over25":   "dc_prob_over25",
            }
            dc_rows = []
            for _, row in val_df.iterrows():
                from src._02_feature_builder import normalize_team_name

                # FIX v6: resolver league_id para ligas EU Y ESPN
                lid = None
                if "league_name" in row:
                    # Buscar en ligas EU primero (clave numérica)
                    lid = next(
                        (k for k, (ln, _, _) in LIGAS.items()
                         if ln == row["league_name"]),
                        None,
                    )
                    # Si no está en EU, buscar en ligas ESPN por nombre
                    if lid is None:
                        try:
                            from config.settings import LIGAS_ESPN, COMPETICIONES_NACIONALES_ESPN
                            _todos = {**LIGAS_ESPN, **COMPETICIONES_NACIONALES_ESPN}
                            lid = next(
                                (league_id for slug, (league_id, name) in _todos.items()
                                 if name == row["league_name"]),
                                None,
                            )
                        except Exception:
                            pass

                probs = dc_ensemble.predict_proba(
                    normalize_team_name(str(row.get("home_team", ""))),
                    normalize_team_name(str(row.get("away_team", ""))),
                    league_id=lid,
                )
                dc_rows.append(probs)
            dc_val_df = pd.DataFrame(dc_rows)
            for market, dc_col in dc_mapping.items():
                if dc_col in dc_val_df.columns:
                    dc_val_probs[market] = dc_val_df[dc_col].values

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

            ref_odds = ref_odds_map.get(market)
            metrics  = _compute_validation_roi(cal_val, y_val, ref_odds)
            self.validation_metrics[market] = metrics

            log.info(
                f"  [{market}] accuracy={metrics['accuracy']:.3f} | "
                f"ROI flat={metrics['roi_flat']:+.1f}% "
                f"({metrics['n_bets']}/{metrics['n_total']} apuestas)"
            )
            if metrics["roi_flat"] < -5:
                log.warning(
                    f"  ⚠ [{market}] ROI negativo en validación."
                )

            self.models[market]      = xgb
            self.calibrators[market] = cal

            fi = pd.Series(xgb.feature_importances_, index=available)
            self.feature_importances[market] = fi.sort_values(ascending=False).head(10)

            if market in dc_val_probs:
                dc_probs_arr = dc_val_probs[market]
                if len(dc_probs_arr) == len(cal_val):
                    optimal_w = _optimize_blend_weight(
                        dc_probs_arr, cal_val, y_val, market
                    )
                    self.blend_weights[market] = optimal_w
                else:
                    self.blend_weights[market] = DEFAULT_DC_WEIGHT
            else:
                self.blend_weights[market] = DEFAULT_DC_WEIGHT

        self.fitted = True
        log.info("Ensemble entrenado. Pesos DC optimizados:")
        for market, w in self.blend_weights.items():
            log.info(f"  {market:10s}: dc_weight={w:.3f}")
        return self

    def predict(self, features_row: dict) -> dict:
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
        lines = ["📊 *Validación del modelo (set temporal 20%)*"]
        for market, m in self.validation_metrics.items():
            emoji = "✅" if m["roi_flat"] >= 0 else "⚠️"
            w     = self.blend_weights.get(market, DEFAULT_DC_WEIGHT)
            lines.append(
                f"{emoji} {market}: acc `{m['accuracy']:.1%}` | "
                f"ROI `{m['roi_flat']:+.1f}%` | dc_w=`{w:.2f}` "
                f"({m['n_bets']} apuestas)"
            )
        return "\n".join(lines)


# ─── BLEND CON PESOS OPTIMIZADOS ─────────────────────────────────────────────

def blend_predictions(dc_probs: dict, xgb_probs: dict,
                      blend_weights: dict = None) -> dict:
    mapping = {
        "home_win": ("dc_prob_home",   "xgb_prob_home_win"),
        "draw":     ("dc_prob_draw",   "xgb_prob_draw"),
        "away_win": ("dc_prob_away",   "xgb_prob_away_win"),
        "btts":     ("dc_prob_btts",   "xgb_prob_btts"),
        "over25":   ("dc_prob_over25", "xgb_prob_over25"),
    }
    blended = {}
    for market, (dk, xk) in mapping.items():
        dp    = dc_probs.get(dk, 0.33)
        xp    = xgb_probs.get(xk, dp)
        dc_w  = (blend_weights or {}).get(market, DEFAULT_DC_WEIGHT)
        xgb_w = 1 - dc_w
        blended[f"prob_{market}"] = (
            round(dc_w * dp + xgb_w * xp, 4) if xp > 0 else round(dp, 4)
        )

    # Normalizar 1X2
    s = sum(blended.get(f"prob_{m}", 0) for m in ["home_win", "draw", "away_win"])
    if s > 0:
        for m in ["home_win", "draw", "away_win"]:
            blended[f"prob_{m}"] = round(blended[f"prob_{m}"] / s, 4)
    return blended


# ─── PERSISTENCIA ────────────────────────────────────────────────────────────

def _save_as_latest(src: str, name: str):
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

    # Normalizar columnas para DC
    if "home_team_norm" not in training_df.columns:
        from src._02_feature_builder import normalize_team_name
        training_df = training_df.copy()
        training_df["home_team_norm"] = training_df["home_team"].apply(normalize_team_name)
        training_df["away_team_norm"] = training_df["away_team"].apply(normalize_team_name)
    if "home_goals" not in training_df.columns and "home_goals_actual" in training_df.columns:
        training_df = training_df.copy()
        training_df["home_goals"] = training_df["home_goals_actual"]
        training_df["away_goals"] = training_df["away_goals_actual"]

    # 1. Entrenar DC por liga
    dc_ensemble = DixonColesEnsemble().fit(training_df)

    # 2. Entrenar XGBoost con optimización de blend weights
    ensemble = FootbotEnsemble().fit(training_df, dc_ensemble=dc_ensemble)

    os.makedirs(MODELS_DIR, exist_ok=True)
    version    = date.today().strftime("%Y%m%d")
    dc_path    = os.path.join(MODELS_DIR, f"dixon_coles_{version}.pkl")
    xgb_path   = os.path.join(MODELS_DIR, f"ensemble_{version}.pkl")

    joblib.dump(dc_ensemble, dc_path)
    joblib.dump(ensemble,    xgb_path)

    _save_as_latest(dc_path,  "dixon_coles")
    _save_as_latest(xgb_path, "ensemble")

    log.info(f"Modelos guardados: versión {version}")
    return dc_ensemble, ensemble


def load_models() -> tuple:
    dc_path  = os.path.join(MODELS_DIR, "dixon_coles_latest.pkl")
    xgb_path = os.path.join(MODELS_DIR, "ensemble_latest.pkl")
    if not os.path.exists(dc_path) or not os.path.exists(xgb_path):
        log.warning("No hay modelos guardados. Entrenando desde cero...")
        return train_and_save()
    dc_ensemble = joblib.load(dc_path)
    ensemble    = joblib.load(xgb_path)
    log.info("Modelos cargados desde disco.")
    return dc_ensemble, ensemble


def predict_match(home_team: str, away_team: str,
                  features_row: dict,
                  dc: DixonColesEnsemble,
                  ensemble: FootbotEnsemble) -> dict:
    from src._02_feature_builder import normalize_team_name

    home_norm  = normalize_team_name(home_team)
    away_norm  = normalize_team_name(away_team)

    league_id  = features_row.get("league_id")
    dc_probs   = dc.predict_proba(home_norm, away_norm, league_id=league_id)

    xgb_probs  = ensemble.predict(features_row)

    blended    = blend_predictions(dc_probs, xgb_probs,
                                   blend_weights=ensemble.blend_weights)

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