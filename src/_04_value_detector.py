"""
_04_value_detector.py
Detecta value bets comparando probabilidades del modelo con cuotas del mercado.

CAMBIOS v3:
  1. get_current_season_odds: busca cuotas de la temporada EN CURSO en lugar
     de los últimos enfrentamientos H2H (que podían ser de hace 3 años).
     Las cuotas de temporada actual reflejan la fuerza relativa presente.
  2. build_odds_dict usa una jerarquía de fuentes más robusta:
       a) Cuotas del partido exacto (mismo H vs A esta temporada)
       b) Cuotas promedio ponderadas de partidos recientes de cada equipo
          (últimos 6 partidos de casa del local, últimas 6 de fuera del visitante)
       c) Fallback a cuotas estimadas
  3. Imports al nivel del módulo con manejo claro de ImportError.
  4. edge_vs_market como campo de transparencia adicional.
  5. classify_confidence usa n_matches_total.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    UMBRAL_EDGE_ALTA, UMBRAL_EDGE_MEDIA,
    UMBRAL_PROB_ALTA, UMBRAL_PROB_MEDIA,
    MIN_PARTIDOS_ALTA, MIN_PARTIDOS_MEDIA,
    KELLY_FRACCION, DATA_PROCESSED,
)

try:
    from src._02_feature_builder import normalize_team_name
    _HAS_NORMALIZER = True
except ImportError as e:
    _HAS_NORMALIZER = False
    logging.getLogger(__name__).warning(f"normalize_team_name no disponible: {e}")
    def normalize_team_name(name):
        return name.lower().strip()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Cuotas fallback con overround ~5-6%
CUOTAS_FALLBACK = {
    "home_win":  2.10,
    "draw":      3.40,
    "away_win":  3.60,
    "btts_si":   1.85,
    "btts_no":   1.95,
    "over25":    1.85,
    "under25":   1.95,
    "double_1x": 1.35,
    "double_x2": 1.55,
    "double_12": 1.30,
}

MARKET_TO_PROB = {
    "home_win": "prob_home_win",
    "draw":     "prob_draw",
    "away_win": "prob_away_win",
    "btts_si":  "prob_btts",
    "over25":   "prob_over25",
}

# Nº de partidos recientes a promediar para cuotas contextuales
_N_RECENT_CONTEXT = 6


# ─── CUOTAS DE TEMPORADA ACTUAL ───────────────────────────────────────────────

def _get_odds_columns(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Devuelve las columnas de cuotas disponibles en orden de prioridad."""
    candidates = [
        ("PSH",   "PSD",   "PSA"),
        ("B365H", "B365D", "B365A"),
        ("BWH",   "BWD",   "BWA"),
        ("WHH",   "WHD",   "WHA"),
        ("VCH",   "VCD",   "VCA"),
    ]
    return [
        (h, d, a) for h, d, a in candidates
        if all(c in df.columns for c in [h, d, a])
    ]


def _extract_odds_from_row(row: pd.Series,
                            odds_cols: list[tuple]) -> dict | None:
    """Extrae cuotas de una fila usando la primera fuente disponible."""
    for col_h, col_d, col_a in odds_cols:
        try:
            h_odd = float(row[col_h])
            d_odd = float(row[col_d])
            a_odd = float(row[col_a])
            if h_odd > 1.0 and d_odd > 1.0 and a_odd > 1.0:
                over25  = None
                under25 = None
                if "B365>2.5" in row.index:
                    try:    over25  = float(row["B365>2.5"])
                    except: pass
                if "B365<2.5" in row.index:
                    try:    under25 = float(row["B365<2.5"])
                    except: pass
                return {
                    "home":    h_odd,
                    "draw":    d_odd,
                    "away":    a_odd,
                    "over25":  over25,
                    "under25": under25,
                    "source":  col_h[:3],
                }
        except (ValueError, TypeError, KeyError):
            continue
    return None


def get_current_season_odds(home_team: str, away_team: str,
                             historical: pd.DataFrame) -> dict | None:
    """
    Estrategia de cuotas v3 — prioridad decreciente:

    1. Partido exacto (home vs away) en la temporada más reciente disponible.
       Es la mejor referencia porque las cuotas de cierre son el mejor
       predictor del mercado eficiente.

    2. Si no hay enfrentamiento directo reciente: promedio ponderado de los
       últimos _N_RECENT_CONTEXT partidos de CASA del equipo local y de
       FUERA del equipo visitante en la misma temporada.
       Esto refleja la fuerza relativa actual de cada equipo en su contexto
       (local/visitante) sin depender de que se hayan enfrentado antes.

    3. Retorna None si no hay cuotas disponibles → fallback a estimadas.
    """
    if historical is None or historical.empty:
        return None
    if "home_team_norm" not in historical.columns:
        return None

    h_norm = normalize_team_name(home_team)
    a_norm = normalize_team_name(away_team)
    odds_cols = _get_odds_columns(historical)
    if not odds_cols:
        return None

    # ── Estrategia 1: partido exacto más reciente ─────────────────────────
    mask_exact = (
        (historical["home_team_norm"] == h_norm) &
        (historical["away_team_norm"] == a_norm)
    )
    exact = historical[mask_exact].sort_values("match_date", ascending=False)

    if not exact.empty:
        row = exact.iloc[0]
        result = _extract_odds_from_row(row, odds_cols)
        if result:
            log.debug(
                f"Cuotas exactas para {home_team} vs {away_team}: "
                f"{result['home']}/{result['draw']}/{result['away']} "
                f"({result['source']})"
            )
            result["method"] = "exact_match"
            return result

    # ── Estrategia 2: promedio contextual (casa/fuera) ────────────────────
    # Últimos N partidos del local jugando en casa
    home_home = historical[
        historical["home_team_norm"] == h_norm
    ].sort_values("match_date", ascending=False).head(_N_RECENT_CONTEXT)

    # Últimos N partidos del visitante jugando fuera
    away_away = historical[
        historical["away_team_norm"] == a_norm
    ].sort_values("match_date", ascending=False).head(_N_RECENT_CONTEXT)

    if home_home.empty or away_away.empty:
        return None

    home_odds_list = []
    away_odds_list = []

    for _, row in home_home.iterrows():
        o = _extract_odds_from_row(row, odds_cols)
        if o:
            home_odds_list.append(o["home"])  # cuota del equipo local en esos partidos

    for _, row in away_away.iterrows():
        o = _extract_odds_from_row(row, odds_cols)
        if o:
            away_odds_list.append(o["away"])  # cuota del equipo visitante en esos partidos

    if not home_odds_list or not away_odds_list:
        return None

    # Promedio de cuotas recientes
    avg_home_odds = float(np.mean(home_odds_list))
    avg_away_odds = float(np.mean(away_odds_list))

    # La cuota de empate la inferimos desde las probabilidades implícitas:
    # P(home) + P(draw) + P(away) = 1 con overround ~5%
    p_home = (1 / avg_home_odds) * 0.95
    p_away = (1 / avg_away_odds) * 0.95
    p_draw = max(0.05, 1.0 - p_home - p_away)
    # Normalizar (el overround puede hacer que sumen > 1)
    total  = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total
    draw_odds = round(1 / p_draw, 3) if p_draw > 0 else 3.40

    # Over/Under 2.5: tomamos el promedio de los partidos del local en casa
    over25_list  = []
    under25_list = []
    for _, row in home_home.iterrows():
        if "B365>2.5" in row.index:
            try:    over25_list.append(float(row["B365>2.5"]))
            except: pass
        if "B365<2.5" in row.index:
            try:    under25_list.append(float(row["B365<2.5"]))
            except: pass

    result = {
        "home":    round(avg_home_odds, 3),
        "draw":    draw_odds,
        "away":    round(avg_away_odds, 3),
        "over25":  round(float(np.mean(over25_list)),  3) if over25_list  else None,
        "under25": round(float(np.mean(under25_list)), 3) if under25_list else None,
        "source":  "contextual",
        "method":  "contextual_avg",
        "n_home":  len(home_odds_list),
        "n_away":  len(away_odds_list),
    }

    log.debug(
        f"Cuotas contextuales para {home_team} vs {away_team}: "
        f"{result['home']}/{result['draw']}/{result['away']} "
        f"(n_home={result['n_home']}, n_away={result['n_away']})"
    )
    return result


# ─── CÁLCULOS CORE ───────────────────────────────────────────────────────────

def calculate_edge(model_prob: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0 or model_prob <= 0:
        return -999.0
    return round((model_prob * decimal_odds - 1) * 100, 2)


def kelly_fraction(model_prob: float, decimal_odds: float,
                   fraction: float = KELLY_FRACCION) -> float:
    if decimal_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b     = decimal_odds - 1
    kelly = (model_prob * b - (1 - model_prob)) / b
    return round(max(kelly * fraction, 0.0) * 100, 2)


def _double_chance_odds(odds_a: float, odds_b: float,
                        overround: float = 0.95) -> float:
    if odds_a <= 1.0 or odds_b <= 1.0:
        return 1.10
    fair_prob = (1 / odds_a) + (1 / odds_b)
    if fair_prob >= 1.0:
        return round(overround, 2)
    return round((1 / fair_prob) * overround, 3)


def get_model_prob_for_market(market: str, predictions: dict) -> float:
    if market == "double_1x":
        return predictions.get("prob_home_win", 0) + predictions.get("prob_draw", 0)
    if market == "double_x2":
        return predictions.get("prob_draw", 0) + predictions.get("prob_away_win", 0)
    if market == "double_12":
        return predictions.get("prob_home_win", 0) + predictions.get("prob_away_win", 0)
    if market == "btts_no":
        return 1 - predictions.get("prob_btts", 0.5)
    if market == "under25":
        return 1 - predictions.get("prob_over25", 0.5)
    key = MARKET_TO_PROB.get(market)
    return predictions.get(key, 0) if key else 0


def build_odds_dict(home_team: str, away_team: str,
                    historical: pd.DataFrame) -> tuple[dict, bool, str]:
    """
    Construye el diccionario de cuotas para un partido.
    Retorna (odds_dict, son_reales, método).
    """
    real = None
    if historical is not None and not historical.empty:
        try:
            real = get_current_season_odds(home_team, away_team, historical)
        except Exception as e:
            log.debug(f"Error obteniendo cuotas para {home_team}: {e}")

    if real:
        h_odds = real["home"]
        d_odds = real["draw"]
        a_odds = real["away"]
        odds = {
            "home_win":  h_odds,
            "draw":      d_odds,
            "away_win":  a_odds,
            "btts_si":   CUOTAS_FALLBACK["btts_si"],
            "btts_no":   CUOTAS_FALLBACK["btts_no"],
            "over25":    real.get("over25")  or CUOTAS_FALLBACK["over25"],
            "under25":   real.get("under25") or CUOTAS_FALLBACK["under25"],
            "double_1x": _double_chance_odds(h_odds, d_odds),
            "double_x2": _double_chance_odds(d_odds, a_odds),
            "double_12": _double_chance_odds(h_odds, a_odds),
        }
        method = real.get("method", "real")
        return odds, True, method

    return CUOTAS_FALLBACK.copy(), False, "fallback"


def classify_confidence(edge: float, model_prob: float,
                        n_home: int, n_away: int) -> str | None:
    if (edge >= UMBRAL_EDGE_ALTA and
        model_prob >= UMBRAL_PROB_ALTA and
        n_home >= MIN_PARTIDOS_ALTA and
        n_away >= MIN_PARTIDOS_ALTA):
        return "alta"
    if (edge >= UMBRAL_EDGE_MEDIA and
        model_prob >= UMBRAL_PROB_MEDIA and
        n_home >= MIN_PARTIDOS_MEDIA and
        n_away >= MIN_PARTIDOS_MEDIA):
        return "media"
    return None


def build_explanation(market: str, features_row: dict,
                       top_features: list) -> str:
    templates = {
        "home_xg_scored":    lambda v: f"xG ofensivo local {v:.2f}",
        "away_xg_scored":    lambda v: f"xG ofensivo visitante {v:.2f}",
        "home_xg_conceded":  lambda v: f"xG concedido local {v:.2f}",
        "away_xg_conceded":  lambda v: f"xG concedido visitante {v:.2f}",
        "home_forma":        lambda v: f"forma local {v:.1f} pts/PJ",
        "away_forma":        lambda v: f"forma visitante {v:.1f} pts/PJ",
        "elo_diff":          lambda v: f"ventaja ELO {abs(v):.0f} pts ({'local' if v>0 else 'visitante'})",
        "h2h_btts_rate":     lambda v: f"BTTS en {v*100:.0f}% H2H",
        "h2h_avg_goals":     lambda v: f"promedio {v:.1f} goles H2H",
        "xg_total_exp":      lambda v: f"xG total {v:.2f}",
        "home_days_rest":    lambda v: f"local {v:.0f}d descanso",
        "away_days_rest":    lambda v: f"visitante {v:.0f}d descanso",
        "fatiga_flag":       lambda v: "fatiga detectada (<4d)" if v else "",
        "rain_flag":         lambda v: "lluvia prevista" if v else "",
        "home_over25_rate":  lambda v: f"local Over2.5 en {v*100:.0f}%",
        "away_over25_rate":  lambda v: f"visitante Over2.5 en {v*100:.0f}%",
    }
    parts = []
    for feat in top_features[:3]:
        val = features_row.get(feat)
        if val is not None and feat in templates:
            try:
                text = templates[feat](float(val))
                if text:
                    parts.append(text)
            except Exception:
                pass
    return " | ".join(parts) or "análisis estadístico del equipo"


def _compute_edge_vs_market(model_prob: float,
                             fixture_row: pd.Series,
                             market: str) -> float | None:
    market_col = {
        "home_win":  "market_prob_home",
        "draw":      "market_prob_draw",
        "away_win":  "market_prob_away",
    }.get(market)

    if not market_col:
        return None

    market_prob = fixture_row.get(market_col)
    if market_prob is None or market_prob <= 0:
        return None

    return round((model_prob - float(market_prob)) * 100, 2)


# ─── ANALIZADOR PRINCIPAL ────────────────────────────────────────────────────

def analyze_fixture(fixture_row: pd.Series, predictions: dict,
                    historical: pd.DataFrame = None) -> list:
    home   = fixture_row.get("home_team", "")
    away   = fixture_row.get("away_team", "")
    n_home = int(fixture_row.get("n_home_matches", 0))
    n_away = int(fixture_row.get("n_away_matches", 0))

    reference_odds, odds_are_real, odds_method = build_odds_dict(
        home, away, historical
    )

    if not odds_are_real:
        log.debug(f"{home} vs {away}: cuotas estimadas (sin datos reales)")
    else:
        log.debug(f"{home} vs {away}: cuotas reales (método={odds_method})")

    markets_to_check = ["home_win", "away_win", "btts_si", "over25"]
    if n_home >= MIN_PARTIDOS_ALTA and n_away >= MIN_PARTIDOS_ALTA:
        markets_to_check += ["draw", "double_1x", "double_x2"]

    bets = []
    for market in markets_to_check:
        model_prob = get_model_prob_for_market(market, predictions)
        if model_prob <= 0:
            continue

        odds       = reference_odds.get(market, CUOTAS_FALLBACK.get(market, 2.0))
        edge       = calculate_edge(model_prob, odds)
        confidence = classify_confidence(edge, model_prob, n_home, n_away)

        # Más conservador si las cuotas son estimadas
        if not odds_are_real and confidence == "media":
            confidence = None
        if confidence is None:
            continue

        kelly          = kelly_fraction(model_prob, odds)
        top_feats      = predictions.get("top_features", {}).get(
            market.replace("_si", "").replace("_no", ""), []
        )
        explanation    = build_explanation(market, fixture_row.to_dict(), top_feats)
        edge_vs_market = _compute_edge_vs_market(model_prob, fixture_row, market)

        display_map = {
            "home_win":  f"Victoria {home}",
            "draw":      "Empate",
            "away_win":  f"Victoria {away}",
            "btts_si":   "Ambos marcan — SÍ",
            "btts_no":   "Ambos marcan — NO",
            "over25":    "Over 2.5 goles",
            "under25":   "Under 2.5 goles",
            "double_1x": f"DC {home}/Empate",
            "double_x2": f"DC Empate/{away}",
            "double_12": f"DC {home}/{away}",
        }

        bets.append({
            "home_team":       home,
            "away_team":       away,
            "league":          fixture_row.get("league_name", ""),
            "match_date":      fixture_row.get("match_date", str(date.today())),
            "market":          market,
            "market_display":  display_map.get(market, market),
            "model_prob":      round(model_prob, 4),
            "model_prob_pct":  f"{model_prob*100:.1f}%",
            "reference_odds":  odds,
            "odds_source":     odds_method,
            "edge_pct":        edge,
            "edge_vs_market":  edge_vs_market,
            "kelly_pct":       kelly,
            "confidence":      confidence,
            "explanation":     explanation,
            "exp_home_goals":  round(predictions.get("dc_exp_home_goals", 0), 2),
            "exp_away_goals":  round(predictions.get("dc_exp_away_goals", 0), 2),
            "n_home_matches":  n_home,
            "n_away_matches":  n_away,
        })

    bets.sort(key=lambda x: (
        {"alta": 0, "media": 1}.get(x["confidence"], 9),
        -x["edge_pct"]
    ))
    return bets


def detect_all_value_bets(features_df: pd.DataFrame,
                           all_predictions: list,
                           historical: pd.DataFrame = None) -> pd.DataFrame:
    all_bets = []

    for i, (_, fixture) in enumerate(features_df.iterrows()):
        if i >= len(all_predictions):
            break
        preds = all_predictions[i]
        bets  = analyze_fixture(fixture, preds, historical)
        all_bets.extend(bets)

        home, away = fixture["home_team"], fixture["away_team"]
        if bets:
            log.info(f"{home} vs {away}: {len(bets)} value bet(s)")
        else:
            log.info(f"{home} vs {away}: sin valor")

    if not all_bets:
        return pd.DataFrame()

    df   = pd.DataFrame(all_bets)
    path = os.path.join(DATA_PROCESSED, f"bets_{date.today()}.csv")
    df.to_csv(path, index=False)
    log.info(f"{len(df)} value bets guardadas → {path}")
    return df


def summarize_bets(bets_df: pd.DataFrame) -> dict:
    if bets_df.empty:
        return {"total": 0, "alta": 0, "media": 0}
    result = {
        "total":    len(bets_df),
        "alta":     len(bets_df[bets_df["confidence"] == "alta"]),
        "media":    len(bets_df[bets_df["confidence"] == "media"]),
        "edge_max": round(bets_df["edge_pct"].max(), 2),
        "edge_avg": round(bets_df["edge_pct"].mean(), 2),
    }
    if "odds_source" in bets_df.columns:
        real_count = len(bets_df[~bets_df["odds_source"].isin(["fallback", "estimada"])])
        result["real_odds_pct"] = round(real_count / len(bets_df) * 100, 1)
        # Desglose por método
        for method, group in bets_df.groupby("odds_source"):
            result[f"odds_{method}"] = len(group)
    return result