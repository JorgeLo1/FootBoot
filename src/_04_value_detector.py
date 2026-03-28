"""
_04_value_detector.py
Detecta value bets comparando probabilidades del modelo con cuotas del mercado.

CAMBIOS v4:
  1. build_odds_dict ahora tiene 3 niveles de fuentes:
       a) Cuotas ESPN en tiempo real (Core API /odds) — mejor fuente disponible
       b) Cuotas históricas Football-Data.co.uk (cierre, partido exacto o contextual)
       c) Fallback a cuotas estimadas
  2. classify_confidence usa n_matches_total correctamente (fix bug v3).
  3. Para ligas ESPN sin cuotas históricas, los umbrales de edge se elevan
     automáticamente (más conservador sin cuotas reales como referencia).
  4. odds_source en el output distingue: espn_live, exact_match, contextual_avg, fallback.
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
    UMBRAL_EDGE_ALTA,  UMBRAL_EDGE_MEDIA,
    UMBRAL_PROB_ALTA,  UMBRAL_PROB_MEDIA,
    MIN_PARTIDOS_ALTA, MIN_PARTIDOS_MEDIA,
    KELLY_FRACCION,    DATA_PROCESSED,
    UMBRAL_EDGE_ALTA_ESPN, UMBRAL_EDGE_MEDIA_ESPN,
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

# ─── CUOTAS FALLBACK ─────────────────────────────────────────────────────────

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

_N_RECENT_CONTEXT = 6


# ─── CUOTAS HISTÓRICAS (Football-Data.co.uk) ─────────────────────────────────

def _get_odds_columns(df: pd.DataFrame) -> list[tuple[str, str, str]]:
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
    Estrategia de cuotas históricas (Football-Data.co.uk):
    1. Partido exacto más reciente
    2. Promedio contextual (últimos N partidos en casa del local + fuera del visitante)
    """
    if historical is None or historical.empty:
        return None
    if "home_team_norm" not in historical.columns:
        return None

    h_norm    = normalize_team_name(home_team)
    a_norm    = normalize_team_name(away_team)
    odds_cols = _get_odds_columns(historical)
    if not odds_cols:
        return None

    # ── Estrategia 1: partido exacto ─────────────────────────────────────
    mask_exact = (
        (historical["home_team_norm"] == h_norm) &
        (historical["away_team_norm"] == a_norm)
    )
    exact = historical[mask_exact].sort_values("match_date", ascending=False)

    if not exact.empty:
        result = _extract_odds_from_row(exact.iloc[0], odds_cols)
        if result:
            result["method"] = "exact_match"
            return result

    # ── Estrategia 2: promedio contextual ────────────────────────────────
    home_home = historical[
        historical["home_team_norm"] == h_norm
    ].sort_values("match_date", ascending=False).head(_N_RECENT_CONTEXT)

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
            home_odds_list.append(o["home"])

    for _, row in away_away.iterrows():
        o = _extract_odds_from_row(row, odds_cols)
        if o:
            away_odds_list.append(o["away"])

    if not home_odds_list or not away_odds_list:
        return None

    avg_home_odds = float(np.mean(home_odds_list))
    avg_away_odds = float(np.mean(away_odds_list))

    p_home = (1 / avg_home_odds) * 0.95
    p_away = (1 / avg_away_odds) * 0.95
    p_draw = max(0.05, 1.0 - p_home - p_away)
    total  = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total
    draw_odds = round(1 / p_draw, 3) if p_draw > 0 else 3.40

    over25_list  = []
    under25_list = []
    for _, row in home_home.iterrows():
        if "B365>2.5" in row.index:
            try:    over25_list.append(float(row["B365>2.5"]))
            except: pass
        if "B365<2.5" in row.index:
            try:    under25_list.append(float(row["B365<2.5"]))
            except: pass

    return {
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


# ─── CUOTAS ESPN EN TIEMPO REAL ───────────────────────────────────────────────

def _get_espn_odds_from_fixture(fixture_row: pd.Series) -> dict | None:
    """
    Extrae cuotas ESPN en tiempo real del fixture_row si están disponibles.
    Estas cuotas se añadieron en build_features_for_fixtures via enrich_fixtures_with_odds.
    """
    if not fixture_row.get("espn_odds_available"):
        return None

    h = fixture_row.get("espn_odds_home")
    d = fixture_row.get("espn_odds_draw")
    a = fixture_row.get("espn_odds_away")

    if not h or not a or float(h) <= 1.0:
        return None

    return {
        "home":    float(h),
        "draw":    float(d) if d else 3.40,
        "away":    float(a),
        "over25":  None,
        "under25": None,
        "source":  "espn_odds",
        "method":  "espn_live",
        "provider": fixture_row.get("espn_odds_provider", "ESPN"),
    }


# ─── CONSTRUCCIÓN UNIFICADA DE CUOTAS ────────────────────────────────────────

def build_odds_dict(home_team: str, away_team: str,
                    historical: pd.DataFrame,
                    fixture_row: pd.Series = None) -> tuple[dict, bool, str]:
    """
    Construye el diccionario de cuotas para un partido con 3 niveles:

    Nivel 1 — ESPN en tiempo real (mejor): cuotas del día del mercado.
    Nivel 2 — Football-Data.co.uk (bueno): cierre histórico real.
    Nivel 3 — Fallback estimado (peor): usar con umbrales más altos.

    Retorna (odds_dict, son_reales, método).
    """
    # ── Nivel 1: ESPN tiempo real ─────────────────────────────────────────
    if fixture_row is not None:
        espn_odds = _get_espn_odds_from_fixture(fixture_row)
        if espn_odds:
            h_odds = espn_odds["home"]
            d_odds = espn_odds["draw"]
            a_odds = espn_odds["away"]
            odds = {
                "home_win":  h_odds,
                "draw":      d_odds,
                "away_win":  a_odds,
                "btts_si":   CUOTAS_FALLBACK["btts_si"],
                "btts_no":   CUOTAS_FALLBACK["btts_no"],
                "over25":    espn_odds.get("over25")  or CUOTAS_FALLBACK["over25"],
                "under25":   espn_odds.get("under25") or CUOTAS_FALLBACK["under25"],
                "double_1x": _double_chance_odds(h_odds, d_odds),
                "double_x2": _double_chance_odds(d_odds, a_odds),
                "double_12": _double_chance_odds(h_odds, a_odds),
            }
            return odds, True, "espn_live"

    # ── Nivel 2: Football-Data.co.uk histórico ───────────────────────────
    real = None
    if historical is not None and not historical.empty:
        try:
            real = get_current_season_odds(home_team, away_team, historical)
        except Exception as e:
            log.debug(f"Error cuotas históricas {home_team}: {e}")

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
        return odds, True, real.get("method", "fd_historical")

    # ── Nivel 3: Fallback estimado ────────────────────────────────────────
    return CUOTAS_FALLBACK.copy(), False, "fallback"


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
    if not odds_a or not odds_b or odds_a <= 1.0 or odds_b <= 1.0:
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


def classify_confidence(edge: float, model_prob: float,
                        n_home: int, n_away: int,
                        odds_method: str = "fd_historical") -> str | None:
    """
    Clasifica la confianza de una apuesta.

    FIX v4: n_home y n_away deben ser n_matches_total (no el conteo de ventana).
    Los umbrales de edge se elevan para ligas sin cuotas reales (fallback).
    """
    # Umbrales según calidad de cuotas
    if odds_method == "espn_live":
        # Cuotas en tiempo real — umbrales estándar
        edge_alta  = UMBRAL_EDGE_ALTA
        edge_media = UMBRAL_EDGE_MEDIA
    elif odds_method in ("exact_match", "contextual_avg", "fd_historical"):
        # Cuotas históricas reales — umbrales estándar
        edge_alta  = UMBRAL_EDGE_ALTA
        edge_media = UMBRAL_EDGE_MEDIA
    else:
        # Fallback estimado — umbrales más altos (más conservador)
        edge_alta  = UMBRAL_EDGE_ALTA_ESPN
        edge_media = UMBRAL_EDGE_MEDIA_ESPN

    if (edge >= edge_alta and
        model_prob >= UMBRAL_PROB_ALTA and
        n_home >= MIN_PARTIDOS_ALTA and
        n_away >= MIN_PARTIDOS_ALTA):
        return "alta"

    if (edge >= edge_media and
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
        "home_win": "market_prob_home",
        "draw":     "market_prob_draw",
        "away_win": "market_prob_away",
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

    # FIX: usar n_matches_total para classify_confidence
    n_home = int(fixture_row.get("n_home_matches", 0))
    n_away = int(fixture_row.get("n_away_matches", 0))

    reference_odds, odds_are_real, odds_method = build_odds_dict(
        home, away, historical, fixture_row
    )

    log.debug(
        f"{home} vs {away}: cuotas {odds_method} "
        f"({'real' if odds_are_real else 'estimada'})"
    )

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

        # FIX: pasar odds_method a classify_confidence para umbrales dinámicos
        confidence = classify_confidence(edge, model_prob, n_home, n_away, odds_method)

        # Descartar media si cuotas son fallback (sin datos de mercado reales)
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
            "odds_are_real":   odds_are_real,
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
        for method in ["espn_live", "exact_match", "contextual_avg", "fallback"]:
            count = len(bets_df[bets_df["odds_source"] == method])
            if count:
                result[f"odds_{method}"] = count
        real_count = len(bets_df[bets_df.get("odds_are_real", False) == True])  # noqa
        result["real_odds_pct"] = round(real_count / len(bets_df) * 100, 1)
    return result