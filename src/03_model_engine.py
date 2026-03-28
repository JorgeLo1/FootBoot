"""
03_model_engine.py
Motor predictivo: Dixon-Coles + XGBoost ensemble con calibración isotónica.
Predice probabilidades para 5 mercados:
  1X2 (home/draw/away), BTTS, Over 2.5, Corners over, HT result
"""

import os
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_PROCESSED, MODELS_DIR, RANDOM_SEED,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Features que usa el modelo (mismo orden siempre)
FEATURE_COLS = [
    "home_xg_scored", "home_xg_conceded", "home_goals_scored",
    "home_goals_conceded", "home_forma", "home_btts_rate",
    "home_over25_rate", "home_corners_avg", "home_fouls_avg",
    "home_days_rest", "away_xg_scored", "away_xg_conceded",
    "away_goals_scored", "away_goals_conceded", "away_forma",
    "away_btts_rate", "away_over25_rate", "away_corners_avg",
    "away_fouls_avg", "away_days_rest", "h2h_home_wins",
    "h2h_draws", "h2h_away_wins", "h2h_avg_goals", "h2h_btts_rate",
    "elo_diff", "xg_diff", "xg_total_exp", "goals_diff",
    "forma_diff", "rest_diff", "fatiga_flag",
    "market_prob_home", "market_prob_draw", "market_prob_away",
    "rain_flag", "wind_flag",
]


# ─── DIXON-COLES ─────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Modelo Dixon-Coles para predicción de goles.
    Corrige la distribución Poisson para marcadores bajos (0-0, 1-0, 0-1, 1-1).
    """
    def __init__(self):
        self.attack  = {}
        self.defense = {}
        self.home_adv = 0.0
        self.rho      = -0.1
        self.fitted   = False
    
    def _tau(self, x, y, mu, lam, rho):
        """Factor de corrección Dixon-Coles para marcadores bajos."""
        if x == 0 and y == 0:
            return 1 - mu * lam * rho
        elif x == 0 and y == 1:
            return 1 + mu * rho
        elif x == 1 and y == 0:
            return 1 + lam * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0
    
    def _log_likelihood(self, params, teams, home_teams, away_teams,
                        home_goals, away_goals):
        """Función de log-verosimilitud negativa para optimización."""
        n = len(teams)
        attack  = dict(zip(teams, params[:n]))
        defense = dict(zip(teams, params[n:2*n]))
        home_adv = params[2*n]
        rho      = params[2*n + 1]
        
        ll = 0.0
        for i in range(len(home_teams)):
            h, a = home_teams[i], away_teams[i]
            hg, ag = home_goals[i], away_goals[i]
            
            if h not in attack or a not in attack:
                continue
            
            mu  = np.exp(attack[h]  + defense[a] + home_adv)  # goles esperados local
            lam = np.exp(attack[a]  + defense[h])              # goles esperados visitante
            
            tau = self._tau(hg, ag, mu, lam, rho)
            if tau <= 0 or mu <= 0 or lam <= 0:
                continue
            
            ll += (np.log(tau) +
                   poisson.logpmf(hg, mu) +
                   poisson.logpmf(ag, lam))
        
        return -ll
    
    def fit(self, df: pd.DataFrame):
        """Entrena el modelo con datos históricos."""
        log.info("Entrenando Dixon-Coles...")
        
        required = ["home_team_norm","away_team_norm","home_goals","away_goals"]
        for col in required:
            if col not in df.columns:
                log.error(f"Columna requerida para Dixon-Coles: {col}")
                return self
        
        df = df.dropna(subset=required)
        
        home_teams = df["home_team_norm"].tolist()
        away_teams = df["away_team_norm"].tolist()
        home_goals = df["home_goals"].astype(int).tolist()
        away_goals = df["away_goals"].astype(int).tolist()
        
        teams = sorted(set(home_teams + away_teams))
        n = len(teams)
        
        # Parámetros iniciales
        x0 = np.zeros(2 * n + 2)
        x0[2*n]     = 0.1   # home advantage
        x0[2*n + 1] = -0.1  # rho (correlación)
        
        # Restricción: suma de ataques = 0 (identificabilidad)
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:n])}]
        
        result = minimize(
            self._log_likelihood,
            x0,
            args=(teams, home_teams, away_teams, home_goals, away_goals),
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-6}
        )
        
        params = result.x
        self.attack   = dict(zip(teams, params[:n]))
        self.defense  = dict(zip(teams, params[n:2*n]))
        self.home_adv = params[2*n]
        self.rho      = params[2*n + 1]
        self.teams    = teams
        self.fitted   = True
        log.info(f"Dixon-Coles entrenado: {n} equipos, home_adv={self.home_adv:.3f}, rho={self.rho:.3f}")
        return self
    
    def predict_proba(self, home_team: str, away_team: str,
                      max_goals: int = 8) -> dict:
        """
        Predice la distribución completa de goles y probabilidades
        para todos los mercados.
        """
        if not self.fitted:
            return self._default_proba()
        
        h = home_team.lower().strip()
        a = away_team.lower().strip()
        
        # Usar media de la liga si el equipo no fue visto en entrenamiento
        mean_att = np.mean(list(self.attack.values()))  if self.attack  else 0.0
        mean_def = np.mean(list(self.defense.values())) if self.defense else 0.0
        
        att_h = self.attack.get(h,  mean_att)
        att_a = self.attack.get(a,  mean_att)
        def_h = self.defense.get(h, mean_def)
        def_a = self.defense.get(a, mean_def)
        
        mu  = np.exp(att_h + def_a + self.home_adv)
        lam = np.exp(att_a + def_h)
        
        # Matriz de probabilidades de marcadores
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = self._tau(i, j, mu, lam, self.rho)
                prob_matrix[i][j] = (tau *
                                     poisson.pmf(i, mu) *
                                     poisson.pmf(j, lam))
        
        # Normalizar
        total = prob_matrix.sum()
        if total > 0:
            prob_matrix /= total
        
        # Extraer probabilidades por mercado
        p_home = float(np.tril(prob_matrix, -1).sum())
        p_draw = float(np.diag(prob_matrix).sum())
        p_away = float(np.triu(prob_matrix, 1).sum())
        
        p_btts = float(prob_matrix[1:, 1:].sum())
        
        p_over25 = float(sum(
            prob_matrix[i][j]
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)
            if i + j > 2.5
        ))
        
        # Expected goals
        exp_home = mu
        exp_away = lam
        exp_total = mu + lam
        
        # HT result (aproximar usando mitad de los goles esperados)
        mu_ht  = mu * 0.5
        lam_ht = lam * 0.5
        p_ht_home = 1 - poisson.cdf(0, mu_ht - lam_ht) if mu_ht > lam_ht else 0.3
        p_ht_draw = poisson.pmf(0, abs(mu_ht - lam_ht)) * 0.4
        p_ht_away = 1 - p_ht_home - p_ht_draw
        
        return {
            "dc_prob_home":   round(max(p_home, 0.01), 4),
            "dc_prob_draw":   round(max(p_draw, 0.01), 4),
            "dc_prob_away":   round(max(p_away, 0.01), 4),
            "dc_prob_btts":   round(max(p_btts, 0.01), 4),
            "dc_prob_over25": round(max(p_over25, 0.01), 4),
            "dc_exp_home_goals": round(exp_home, 3),
            "dc_exp_away_goals": round(exp_away, 3),
            "dc_exp_total_goals": round(exp_total, 3),
        }
    
    def _default_proba(self):
        return {
            "dc_prob_home": 0.45, "dc_prob_draw": 0.27, "dc_prob_away": 0.28,
            "dc_prob_btts": 0.50, "dc_prob_over25": 0.50,
            "dc_exp_home_goals": 1.4, "dc_exp_away_goals": 1.1,
            "dc_exp_total_goals": 2.5,
        }


# ─── XGBOOST ENSEMBLE ────────────────────────────────────────────────────────

class FootbotEnsemble:
    """
    Ensemble XGBoost + calibración isotónica para 5 mercados.
    """
    MARKETS = ["home_win", "draw", "away_win", "btts", "over25"]
    
    def __init__(self):
        self.models      = {}
        self.calibrators = {}
        self.feature_importances = {}
        self.fitted = False
    
    def _get_xgb(self):
        return XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
    
    def fit(self, df: pd.DataFrame):
        """Entrena un modelo XGBoost por mercado con calibración isotónica."""
        targets = {
            "home_win": "target_home_win",
            "draw":     "target_draw",
            "away_win": "target_away_win",
            "btts":     "target_btts",
            "over25":   "target_over25",
        }
        
        # Preparar features
        available_features = [f for f in FEATURE_COLS if f in df.columns]
        X = df[available_features].fillna(0).values
        
        for market, target_col in targets.items():
            if target_col not in df.columns:
                log.warning(f"Target {target_col} no disponible, omitiendo.")
                continue
            
            y = df[target_col].values
            
            log.info(f"Entrenando XGBoost para {market} ({y.sum()} positivos de {len(y)})...")
            
            xgb = self._get_xgb()
            
            # Walk-forward split (últimos 20% para validar)
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            xgb.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)
            
            # Calibración isotónica sobre el set de validación
            proba_val = xgb.predict_proba(X_val)[:, 1]
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(proba_val, y_val)
            
            # Accuracy en validación
            proba_full = ir.transform(xgb.predict_proba(X_val)[:, 1])
            acc = ((proba_full > 0.5).astype(int) == y_val).mean()
            log.info(f"  {market}: accuracy validación = {acc:.3f}")
            
            self.models[market]      = xgb
            self.calibrators[market] = ir
            
            # Importancia de features
            fi = pd.Series(xgb.feature_importances_, index=available_features)
            self.feature_importances[market] = fi.sort_values(ascending=False).head(10)
        
        self.fitted_features = available_features
        self.fitted = True
        log.info("Ensemble entrenado correctamente.")
        return self
    
    def predict(self, features_row: dict) -> dict:
        """Predice probabilidades calibradas para un partido."""
        if not self.fitted:
            return {}
        
        X = np.array([[features_row.get(f, 0) for f in self.fitted_features]])
        
        result = {}
        for market, xgb in self.models.items():
            raw_prob = xgb.predict_proba(X)[0, 1]
            cal_prob = float(self.calibrators[market].transform([raw_prob])[0])
            result[f"xgb_prob_{market}"] = round(cal_prob, 4)
        
        return result
    
    def get_top_features(self, market: str, n: int = 5) -> list:
        """Retorna los top N features más importantes para un mercado."""
        if market not in self.feature_importances:
            return []
        return self.feature_importances[market].head(n).index.tolist()


# ─── ENSEMBLE FINAL ───────────────────────────────────────────────────────────

def blend_predictions(dc_probs: dict, xgb_probs: dict,
                      dc_weight: float = 0.35) -> dict:
    """
    Combina Dixon-Coles con XGBoost via weighted average.
    DC es más sólido en muestras pequeñas; XGBoost domina con muchos datos.
    """
    xgb_weight = 1 - dc_weight
    
    markets_map = {
        "home_win": ("dc_prob_home",   "xgb_prob_home_win"),
        "draw":     ("dc_prob_draw",   "xgb_prob_draw"),
        "away_win": ("dc_prob_away",   "xgb_prob_away_win"),
        "btts":     ("dc_prob_btts",   "xgb_prob_btts"),
        "over25":   ("dc_prob_over25", "xgb_prob_over25"),
    }
    
    blended = {}
    for market, (dc_key, xgb_key) in markets_map.items():
        dc_p  = dc_probs.get(dc_key, 0.33)
        xgb_p = xgb_probs.get(xgb_key, dc_p)
        
        if xgb_p > 0:
            blended[f"prob_{market}"] = round(
                dc_weight * dc_p + xgb_weight * xgb_p, 4
            )
        else:
            blended[f"prob_{market}"] = round(dc_p, 4)
    
    # Normalizar 1X2 para que sumen 1
    total_1x2 = (blended.get("prob_home_win", 0) +
                 blended.get("prob_draw", 0) +
                 blended.get("prob_away_win", 0))
    if total_1x2 > 0:
        for k in ["prob_home_win", "prob_draw", "prob_away_win"]:
            blended[k] = round(blended[k] / total_1x2, 4)
    
    return blended


# ─── ENTRENAMIENTO Y PERSISTENCIA ────────────────────────────────────────────

def train_and_save(training_df: pd.DataFrame = None) -> tuple:
    """
    Entrena el modelo completo (DC + XGBoost) y lo guarda en disco.
    """
    if training_df is None:
        path = os.path.join(DATA_PROCESSED, "training_dataset.csv")
        if not os.path.exists(path):
            log.error("No hay dataset de entrenamiento. Ejecuta 02 primero.")
            return None, None
        training_df = pd.read_csv(path)
    
    log.info(f"Entrenando con {len(training_df)} partidos históricos...")
    
    # Añadir columnas de normalización para DC
    if "home_team_norm" not in training_df.columns:
        from src._02_feature_builder import normalize_team_name
        training_df["home_team_norm"] = training_df["home_team"].apply(normalize_team_name)
        training_df["away_team_norm"] = training_df["away_team"].apply(normalize_team_name)
    if "home_goals" not in training_df.columns and "home_goals_actual" in training_df.columns:
        training_df["home_goals"] = training_df["home_goals_actual"]
        training_df["away_goals"] = training_df["away_goals_actual"]
    
    # Dixon-Coles
    dc = DixonColesModel()
    dc.fit(training_df)
    
    # XGBoost Ensemble
    ensemble = FootbotEnsemble()
    ensemble.fit(training_df)
    
    # Guardar modelos versionados
    version = date.today().strftime("%Y%m%d")
    dc_path  = os.path.join(MODELS_DIR, f"dixon_coles_{version}.pkl")
    xgb_path = os.path.join(MODELS_DIR, f"ensemble_{version}.pkl")
    
    joblib.dump(dc,       dc_path)
    joblib.dump(ensemble, xgb_path)
    
    # Symlinks a la versión "latest"
    for src, dst in [(dc_path, os.path.join(MODELS_DIR, "dixon_coles_latest.pkl")),
                     (xgb_path, os.path.join(MODELS_DIR, "ensemble_latest.pkl"))]:
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    
    log.info(f"Modelos guardados: versión {version}")
    return dc, ensemble


def load_models() -> tuple:
    """Carga los modelos más recientes desde disco."""
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
    """
    Genera predicción completa para un partido.
    Retorna probabilidades blended para todos los mercados.
    """
    from src._02_feature_builder import normalize_team_name
    
    dc_probs  = dc.predict_proba(
        normalize_team_name(home_team),
        normalize_team_name(away_team)
    )
    xgb_probs = ensemble.predict(features_row)
    blended   = blend_predictions(dc_probs, xgb_probs)
    
    # Top features explicativos para el mensaje de Telegram
    top_features = {}
    for market in ["home_win", "btts", "over25"]:
        top_features[market] = ensemble.get_top_features(market, n=3)
    
    return {
        **blended,
        **dc_probs,        # incluir predicciones DC para trazabilidad
        "top_features": top_features,
        "dc_exp_home_goals":  dc_probs.get("dc_exp_home_goals", 0),
        "dc_exp_away_goals":  dc_probs.get("dc_exp_away_goals", 0),
        "dc_exp_total_goals": dc_probs.get("dc_exp_total_goals", 0),
    }


if __name__ == "__main__":
    dc, ensemble = train_and_save()
    if dc and ensemble:
        # Test de predicción
        test_features = {f: 0.5 for f in FEATURE_COLS}
        test_features.update({
            "home_xg_scored": 1.8, "home_xg_conceded": 1.1,
            "away_xg_scored": 1.2, "away_xg_conceded": 1.5,
            "elo_diff": 150, "home_forma": 2.1, "away_forma": 1.3,
        })
        result = predict_match("Arsenal", "Chelsea", test_features, dc, ensemble)
        for k, v in result.items():
            if k != "top_features":
                print(f"  {k}: {v}")
