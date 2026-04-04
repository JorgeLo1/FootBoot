"""
_04_value_detector.py — v6
Detecta value bets comparando probabilidades del modelo con cuotas del mercado.

CAMBIOS v6:
  Tres niveles de confianza: alta, media y baja.

  - ALTA  (edge >= 8%,  prob >= 62%, n >= 30): señal fuerte, kelly 2-3.5%
  - MEDIA (edge >= 4%,  prob >= 55%, n >= 15): señal moderada, kelly 0.5-1.5%
  - BAJA  (edge >= 2%,  prob >= 52%, n >= 6):  señal débil, solo con cuotas
                                                reales (ESPN o fd.co.uk).
                                                kelly reducido al 50%.

  El nivel BAJA tiene estas restricciones adicionales:
    1. SOLO se emite con cuotas reales (espn_live, exact_match,
       contextual_avg, fd_historical). Con model_implied nunca.
    2. SOLO para mercados estándar (1X2, Over/Under, BTTS, Doble oportunidad,
       AH estándar). Los mercados nicho (exactos, combinadas) ya tienen
       umbrales propios en model_implied.
    3. Kelly se aplica al 50% de la fracción normal para reflejar
       la menor certeza estadística.

MERCADOS EXPANDIDOS (sin cambios desde v5):
  - Goles totales: Over/Under 0.5, 1.5, 2.5, 3.5, 4.5
  - Goles por equipo: Over/Under 0.5 y 1.5 local y visitante
  - Goles exactos: 0, 1, 2, 3, 4+ goles
  - Combinadas: BTTS + resultado
  - Asian Handicap: usa spread real ESPN cuando está disponible
  - 1X2 y Doble Oportunidad

CUOTAS ESPN INTEGRADAS (schema confirmado Core API /odds):
  Nivel 1a — ESPN 1X2:        moneyLine home/away/draw (decimal convertido)
  Nivel 1b — ESPN Over/Under: overOdds + underOdds sobre la línea espn_total_line
  Nivel 1c — ESPN Spread:     spreadOdds sobre espn_spread_line (AH real)

ESTRATEGIA model_implied (sin cuotas reales):
  Mercados estándar (1X2, Over/Under 2.5): BLOQUEADOS — el mercado es
  tan eficiente que sin referencia externa el edge es ruido.
  Mercados nicho (exactos, por equipo, combinadas, AH nicho): permitidos
  solo con umbral elevado (+12%/+8%) Y prob > 65%.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import poisson
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    UMBRAL_EDGE_ALTA,  UMBRAL_EDGE_MEDIA,  UMBRAL_EDGE_BAJA,
    UMBRAL_PROB_ALTA,  UMBRAL_PROB_MEDIA,  UMBRAL_PROB_BAJA,
    MIN_PARTIDOS_ALTA, MIN_PARTIDOS_MEDIA, MIN_PARTIDOS_BAJA,
    KELLY_FRACCION,    DATA_PROCESSED,
    UMBRAL_EDGE_ALTA_ESPN, UMBRAL_EDGE_MEDIA_ESPN,
)

try:
    from src._02_feature_builder import normalize_team_name
    _HAS_NORMALIZER = True
except ImportError:
    _HAS_NORMALIZER = False
    def normalize_team_name(name):
        return name.lower().strip()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MAX_GOALS = 10

# Umbrales para mercados nicho sin cuotas reales
_EDGE_NICHO_ALTA  = UMBRAL_EDGE_ALTA  + 4.0   # 12%
_EDGE_NICHO_MEDIA = UMBRAL_EDGE_MEDIA + 4.0   # 8%
_PROB_NICHO       = 0.65

# Mercados estándar bloqueados con model_implied (muy eficientes)
_MERCADOS_BLOQUEADOS_MODEL_IMPLIED = {
    "home_win", "draw", "away_win",
    "over25", "under25",
    "double_1x", "double_x2", "double_12",
}

# Mercados permitidos en nivel BAJA — solo estándar con cuotas reales
# Los nicho tienen sus propios umbrales elevados en model_implied
_MERCADOS_NIVEL_BAJA = {
    "home_win", "draw", "away_win",
    "btts_si", "btts_no",
    "over15", "under15",
    "over25", "under25",
    "over35", "under35",
    "double_1x", "double_x2", "double_12",
    "ah_home_minus05", "ah_away_minus05",
    "ah_home_plus05",  "ah_away_plus05",
}

# Cuotas fallback — SOLO si no hay ninguna fuente real ni ESPN
CUOTAS_FALLBACK = {
    "home_win":   2.10, "draw":       3.40, "away_win":   3.60,
    "btts_si":    1.85, "btts_no":    1.95,
    "over05":     1.15, "under05":    5.50,
    "over15":     1.50, "under15":    2.50,
    "over25":     1.85, "under25":    1.95,
    "over35":     2.40, "under35":    1.55,
    "over45":     3.20, "under45":    1.30,
    "home_over05": 1.55, "home_under05": 2.30,
    "home_over15": 2.30, "home_under15": 1.60,
    "away_over05": 1.75, "away_under05": 2.00,
    "away_over15": 2.80, "away_under15": 1.45,
    "exact_0":    7.00, "exact_1":    4.00,
    "exact_2":    3.40, "exact_3":    3.80, "exact_4plus": 2.80,
    "home_and_btts": 3.20, "draw_and_btts": 5.50, "away_and_btts": 5.00,
    "double_1x":  1.35, "double_x2":  1.55, "double_12":   1.30,
    "ah_home_minus05": 1.90, "ah_home_plus05":  1.90,
    "ah_away_minus05": 1.90, "ah_away_plus05":  1.90,
}

_N_RECENT_CONTEXT = 6


# ─── DISTRIBUCIÓN POISSON ────────────────────────────────────────────────────

def _poisson_matrix(mu: float, lam: float,
                    max_g: int = MAX_GOALS) -> np.ndarray:
    """Matriz P(home=i, away=j) normalizada."""
    M = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            M[i][j] = poisson.pmf(i, mu) * poisson.pmf(j, lam)
    total = M.sum()
    return M / total if total > 0 else M


def compute_all_market_probs(mu: float, lam: float) -> dict:
    """
    Calcula probabilidades para todos los mercados desde goles esperados DC.

    Los mercados 1X2, BTTS y Over/Under 2.5 serán sobreescritos en
    analyze_fixture con el blend DC+XGBoost (más preciso).
    El resto (exactos, por equipo, combinadas) usan Poisson puro,
    que es la única fuente disponible para estos mercados.
    """
    M = _poisson_matrix(mu, lam)
    n = MAX_GOALS

    # 1X2
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.diag(M).sum())
    p_away = float(np.triu(M, 1).sum())
    s = p_home + p_draw + p_away
    if s > 0:
        p_home /= s; p_draw /= s; p_away /= s

    # BTTS
    p_btts_si = float(M[1:, 1:].sum())

    # Totales
    def p_over(line):
        return float(sum(M[i][j] for i in range(n+1) for j in range(n+1) if i+j > line))

    # Marginales por equipo
    home_pmf = M.sum(axis=1)
    away_pmf = M.sum(axis=0)

    # Exactos
    p_exact = {k: float(sum(M[i][j] for i in range(n+1) for j in range(n+1) if i+j == k))
               for k in range(4)}
    p_exact["4plus"] = float(sum(M[i][j] for i in range(n+1) for j in range(n+1) if i+j >= 4))

    # Combinadas resultado + BTTS
    p_home_btts = float(sum(M[i][j] for i in range(1,n+1) for j in range(1,n+1) if i > j))
    p_draw_btts = float(sum(M[i][j] for i in range(1,n+1) for j in range(1,n+1) if i == j))
    p_away_btts = float(sum(M[i][j] for i in range(1,n+1) for j in range(1,n+1) if j > i))

    # AH estándar ±0.5
    p_ah_home_minus05 = p_home
    p_ah_away_minus05 = p_away + p_draw
    p_ah_home_plus05  = p_home + p_draw
    p_ah_away_plus05  = p_away

    # AH -1 (local gana por 2+)
    p_ah_home_minus1 = float(sum(M[i][j] for i in range(n+1) for j in range(n+1) if i-j >= 2))
    p_ah_away_minus1 = 1.0 - p_ah_home_minus1

    return {
        "prob_home_win":    round(p_home, 4),
        "prob_draw":        round(p_draw, 4),
        "prob_away_win":    round(p_away, 4),
        "prob_btts":        round(p_btts_si,        4),
        "prob_btts_no":     round(1 - p_btts_si,    4),
        "prob_over05":      round(p_over(0.5),       4),
        "prob_under05":     round(1 - p_over(0.5),   4),
        "prob_over15":      round(p_over(1.5),       4),
        "prob_under15":     round(1 - p_over(1.5),   4),
        "prob_over25":      round(p_over(2.5),       4),
        "prob_under25":     round(1 - p_over(2.5),   4),
        "prob_over35":      round(p_over(3.5),       4),
        "prob_under35":     round(1 - p_over(3.5),   4),
        "prob_over45":      round(p_over(4.5),       4),
        "prob_under45":     round(1 - p_over(4.5),   4),
        "prob_home_over05": round(float(home_pmf[1:].sum()),   4),
        "prob_home_under05":round(float(home_pmf[0]),          4),
        "prob_home_over15": round(float(home_pmf[2:].sum()),   4),
        "prob_home_under15":round(float(home_pmf[:2].sum()),   4),
        "prob_away_over05": round(float(away_pmf[1:].sum()),   4),
        "prob_away_under05":round(float(away_pmf[0]),          4),
        "prob_away_over15": round(float(away_pmf[2:].sum()),   4),
        "prob_away_under15":round(float(away_pmf[:2].sum()),   4),
        "prob_exact_0":     round(p_exact[0],          4),
        "prob_exact_1":     round(p_exact[1],          4),
        "prob_exact_2":     round(p_exact[2],          4),
        "prob_exact_3":     round(p_exact[3],          4),
        "prob_exact_4plus": round(p_exact["4plus"],    4),
        "prob_home_and_btts": round(p_home_btts,       4),
        "prob_draw_and_btts": round(p_draw_btts,       4),
        "prob_away_and_btts": round(p_away_btts,       4),
        "prob_double_1x":   round(p_home + p_draw,     4),
        "prob_double_x2":   round(p_draw + p_away,     4),
        "prob_double_12":   round(p_home + p_away,     4),
        "prob_ah_home_minus05": round(p_ah_home_minus05, 4),
        "prob_ah_away_minus05": round(p_ah_away_minus05, 4),
        "prob_ah_home_plus05":  round(p_ah_home_plus05,  4),
        "prob_ah_away_plus05":  round(p_ah_away_plus05,  4),
        "prob_ah_home_minus1":  round(p_ah_home_minus1,  4),
        "prob_ah_away_minus1":  round(p_ah_away_minus1,  4),
        "exp_home_goals":   round(mu,       2),
        "exp_away_goals":   round(lam,      2),
        "exp_total_goals":  round(mu + lam, 2),
    }


def _compute_ah_prob_from_spread(spread_line: float,
                                  mu: float, lam: float) -> tuple[float, float]:
    """
    Calcula P(home cubre) y P(away cubre) para un spread real ESPN.
    """
    M   = _poisson_matrix(mu, lam)
    n   = MAX_GOALS
    threshold = -spread_line

    p_home_covers = float(sum(
        M[i][j] for i in range(n+1) for j in range(n+1)
        if (i - j) > threshold - 0.001
    ))
    is_integer_spread = abs(threshold - round(threshold)) < 0.01
    if is_integer_spread:
        p_push = float(sum(
            M[i][j] for i in range(n+1) for j in range(n+1)
            if abs((i - j) - threshold) < 0.01
        ))
        p_home_covers = float(sum(
            M[i][j] for i in range(n+1) for j in range(n+1)
            if (i - j) > threshold + 0.01
        )) + p_push * 0.5
    p_away_covers = 1.0 - p_home_covers

    return round(p_home_covers, 4), round(p_away_covers, 4)


# ─── CUOTAS HISTÓRICAS (Football-Data.co.uk) ─────────────────────────────────

def _get_odds_columns(df: pd.DataFrame) -> list[tuple]:
    candidates = [
        ("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A"), ("BWH", "BWD", "BWA"),
    ]
    return [(h, d, a) for h, d, a in candidates
            if all(c in df.columns for c in [h, d, a])]


def _extract_odds_from_row(row: pd.Series, odds_cols: list) -> dict | None:
    for col_h, col_d, col_a in odds_cols:
        try:
            h = float(row[col_h]); d = float(row[col_d]); a = float(row[col_a])
            if h > 1.0 and d > 1.0 and a > 1.0:
                over25  = None; under25 = None
                if "B365>2.5" in row.index:
                    try: over25  = float(row["B365>2.5"])
                    except: pass
                if "B365<2.5" in row.index:
                    try: under25 = float(row["B365<2.5"])
                    except: pass
                return {"home": h, "draw": d, "away": a,
                        "over25": over25, "under25": under25, "source": col_h[:3]}
        except (ValueError, TypeError, KeyError):
            continue
    return None


def get_current_season_odds(home_team: str, away_team: str,
                             historical: pd.DataFrame) -> dict | None:
    if historical is None or historical.empty:
        return None
    if "home_team_norm" not in historical.columns:
        return None

    h_norm    = normalize_team_name(home_team)
    a_norm    = normalize_team_name(away_team)
    odds_cols = _get_odds_columns(historical)
    if not odds_cols:
        return None

    mask   = (historical["home_team_norm"] == h_norm) & (historical["away_team_norm"] == a_norm)
    exact  = historical[mask].sort_values("match_date", ascending=False)
    if not exact.empty:
        result = _extract_odds_from_row(exact.iloc[0], odds_cols)
        if result:
            result["method"] = "exact_match"
            return result

    home_home = historical[historical["home_team_norm"] == h_norm]\
        .sort_values("match_date", ascending=False).head(_N_RECENT_CONTEXT)
    away_away = historical[historical["away_team_norm"] == a_norm]\
        .sort_values("match_date", ascending=False).head(_N_RECENT_CONTEXT)

    if home_home.empty or away_away.empty:
        return None

    hl, al = [], []
    for _, row in home_home.iterrows():
        o = _extract_odds_from_row(row, odds_cols)
        if o: hl.append(o["home"])
    for _, row in away_away.iterrows():
        o = _extract_odds_from_row(row, odds_cols)
        if o: al.append(o["away"])
    if not hl or not al:
        return None

    avg_h = float(np.mean(hl)); avg_a = float(np.mean(al))
    p_h = (1/avg_h)*0.95; p_a = (1/avg_a)*0.95
    p_d = max(0.05, 1.0 - p_h - p_a); total = p_h + p_d + p_a
    draw_odds = round(1 / (p_d / total), 3)

    over25_list, under25_list = [], []
    for _, row in home_home.iterrows():
        if "B365>2.5" in row.index:
            try: over25_list.append(float(row["B365>2.5"]))
            except: pass
        if "B365<2.5" in row.index:
            try: under25_list.append(float(row["B365<2.5"]))
            except: pass

    return {
        "home":    round(avg_h, 3), "draw":    draw_odds, "away":    round(avg_a, 3),
        "over25":  round(float(np.mean(over25_list)),  3) if over25_list  else None,
        "under25": round(float(np.mean(under25_list)), 3) if under25_list else None,
        "source":  "contextual", "method": "contextual_avg",
    }


# ─── CONSTRUCCIÓN DE CUOTAS ───────────────────────────────────────────────────

def _fair_to_market(prob: float, overround: float = 0.05) -> float:
    if prob <= 0: return 99.0
    return round(max(1.01, (1 / prob) * (1 - overround)), 3)


def _double_chance_odds(odds_a: float, odds_b: float,
                        overround: float = 0.95) -> float:
    if not odds_a or not odds_b or odds_a <= 1.0 or odds_b <= 1.0:
        return 1.10
    fair_prob = (1/odds_a) + (1/odds_b)
    if fair_prob >= 1.0:
        return round(overround, 2)
    return round((1/fair_prob) * overround, 3)


def build_odds_dict(home_team: str, away_team: str,
                    historical: pd.DataFrame,
                    fixture_row: pd.Series,
                    market_probs: dict) -> tuple[dict, bool, str]:
    """
    Construye cuotas para todos los mercados con 3 niveles.

    Nivel 1 — ESPN real (mejor):
        1a. 1X2 → moneyLine convertido a decimal
        1b. Over/Under → overOdds/underOdds sobre espn_total_line
        1c. AH/Spread → spreadOdds sobre espn_spread_line real

    Nivel 2 — Football-Data.co.uk (bueno):
        Solo 1X2 y Over/Under 2.5.

    Nivel 3 — model_implied (solo para mercados nicho):
        Cuota fair derivada del modelo con overround 5%.
        Mercados estándar bloqueados.

    Retorna (odds_dict, tiene_cuotas_reales_1x2, método).
    """
    # FIX v7: compute_all_market_probs genera "prob_btts" (sin _si).
    # El mercado interno se llama "btts_si" en ALL_MARKETS/evaluate_bet,
    # pero la clave en market_probs es "prob_btts". Se mapea explícitamente
    # antes de entrar al loop para que _fair_to_market reciba la prob correcta.
    _btts_prob = market_probs.get("prob_btts", 0)
    model_odds = {
        mkt: _fair_to_market(market_probs.get(f"prob_{mkt}", 0))
        for mkt in [
            "home_win", "draw", "away_win",
            "btts_no",
            "over05", "under05", "over15", "under15",
            "over25", "under25", "over35", "under35", "over45", "under45",
            "home_over05", "home_under05", "home_over15", "home_under15",
            "away_over05", "away_under05", "away_over15", "away_under15",
            "exact_0", "exact_1", "exact_2", "exact_3", "exact_4plus",
            "home_and_btts", "draw_and_btts", "away_and_btts",
            "double_1x", "double_x2", "double_12",
            "ah_home_minus05", "ah_away_minus05",
            "ah_home_plus05",  "ah_away_plus05",
            "ah_home_minus1",  "ah_away_minus1",
        ]
    }
    # Asignar btts_si con la prob correcta (prob_btts, no prob_btts_si)
    model_odds["btts_si"] = _fair_to_market(_btts_prob)

    # ── Nivel 1: ESPN real ────────────────────────────────────────────────
    if fixture_row.get("espn_odds_available"):
        h_odd = fixture_row.get("espn_odds_home")
        d_odd = fixture_row.get("espn_odds_draw")
        a_odd = fixture_row.get("espn_odds_away")

        if h_odd and a_odd and float(h_odd) > 1.0:
            odds = model_odds.copy()

            odds["home_win"] = float(h_odd)
            odds["draw"]     = float(d_odd) if d_odd else model_odds["draw"]
            odds["away_win"] = float(a_odd)
            odds["double_1x"] = _double_chance_odds(float(h_odd), odds["draw"])
            odds["double_x2"] = _double_chance_odds(odds["draw"], float(a_odd))
            odds["double_12"] = _double_chance_odds(float(h_odd), float(a_odd))

            total_line = fixture_row.get("espn_total_line")
            over_dec   = fixture_row.get("espn_over_odds")
            under_dec  = fixture_row.get("espn_under_odds")

            if total_line is not None and over_dec and under_dec:
                line = float(total_line)
                line_to_mkt = {0.5: ("over05", "under05"), 1.5: ("over15", "under15"),
                               2.5: ("over25", "under25"), 3.5: ("over35", "under35"),
                               4.5: ("over45", "under45")}
                mkt_over, mkt_under = line_to_mkt.get(line, ("over25", "under25"))
                odds[mkt_over]  = float(over_dec)
                odds[mkt_under] = float(under_dec)
            elif over_dec and under_dec:
                odds["over25"]  = float(over_dec)
                odds["under25"] = float(under_dec)

            spread_line = fixture_row.get("espn_spread_line")
            sh_odds     = fixture_row.get("espn_spread_home_odds")
            sa_odds     = fixture_row.get("espn_spread_away_odds")

            if spread_line is not None and sh_odds and sa_odds:
                sl = float(spread_line)
                if abs(sl - (-0.5)) < 0.3:
                    odds["ah_home_minus05"] = float(sh_odds)
                    odds["ah_away_plus05"]  = float(sa_odds)
                elif abs(sl - 0.5) < 0.3:
                    odds["ah_home_plus05"]  = float(sh_odds)
                    odds["ah_away_minus05"] = float(sa_odds)
                elif sl <= -0.9:
                    odds["ah_home_minus1"]  = float(sh_odds)
                    odds["ah_away_minus1"]  = float(sa_odds)

            return odds, True, "espn_live"

    # ── Nivel 2: Football-Data.co.uk ──────────────────────────────────────
    real = None
    if historical is not None and not historical.empty:
        try:
            real = get_current_season_odds(home_team, away_team, historical)
        except Exception:
            pass

    if real:
        odds = model_odds.copy()
        odds["home_win"] = real["home"]
        odds["draw"]     = real["draw"]
        odds["away_win"] = real["away"]
        odds["double_1x"] = _double_chance_odds(real["home"], real["draw"])
        odds["double_x2"] = _double_chance_odds(real["draw"], real["away"])
        odds["double_12"] = _double_chance_odds(real["home"], real["away"])
        if real.get("over25"):  odds["over25"]  = real["over25"]
        if real.get("under25"): odds["under25"] = real["under25"]
        return odds, True, real.get("method", "fd_historical")

    # ── Nivel 3: model_implied ────────────────────────────────────────────
    return model_odds, False, "model_implied"


# ─── CLASIFICACIÓN DE CONFIANZA ───────────────────────────────────────────────

def classify_confidence(edge: float, model_prob: float,
                        n_home: int, n_away: int,
                        odds_method: str,
                        market: str) -> str | None:
    """
    Clasifica confianza en tres niveles: alta, media, baja.

    ALTA:  edge >= 8%,  prob >= 62%, n >= 30. Señal fuerte.
    MEDIA: edge >= 4%,  prob >= 55%, n >= 15. Señal moderada.
    BAJA:  edge >= 2%,  prob >= 52%, n >= 6.
           Solo con cuotas reales (nunca model_implied).
           Solo mercados estándar (_MERCADOS_NIVEL_BAJA).

    Con model_implied:
      - Mercados estándar: BLOQUEADOS siempre.
      - Mercados nicho: umbral elevado (+4pp) y prob >= 65%.
    """
    is_real = odds_method in ("espn_live", "exact_match", "contextual_avg", "fd_historical")

    if odds_method == "model_implied":
        if market in _MERCADOS_BLOQUEADOS_MODEL_IMPLIED:
            return None
        edge_alta  = _EDGE_NICHO_ALTA
        edge_media = _EDGE_NICHO_MEDIA
        prob_min   = _PROB_NICHO
        # Sin nivel baja para model_implied
        if (edge >= edge_alta and model_prob >= prob_min and
            n_home >= MIN_PARTIDOS_BAJA and n_away >= MIN_PARTIDOS_BAJA):
            return "alta"
        if (edge >= edge_media and model_prob >= prob_min and
            n_home >= MIN_PARTIDOS_BAJA and n_away >= MIN_PARTIDOS_BAJA):
            return "media"
        return None

    elif is_real:
        edge_alta  = UMBRAL_EDGE_ALTA
        edge_media = UMBRAL_EDGE_MEDIA
        prob_alta  = UMBRAL_PROB_ALTA
        prob_media = UMBRAL_PROB_MEDIA
    else:
        # ESPN sin cuotas reales (casos edge)
        edge_alta  = UMBRAL_EDGE_ALTA_ESPN
        edge_media = UMBRAL_EDGE_MEDIA_ESPN
        prob_alta  = UMBRAL_PROB_ALTA
        prob_media = UMBRAL_PROB_MEDIA

    # ── ALTA ─────────────────────────────────────────────────────────────
    if (edge >= edge_alta and model_prob >= prob_alta and
        n_home >= MIN_PARTIDOS_ALTA and n_away >= MIN_PARTIDOS_ALTA):
        return "alta"

    # ── MEDIA ─────────────────────────────────────────────────────────────
    if (edge >= edge_media and model_prob >= prob_media and
        n_home >= MIN_PARTIDOS_MEDIA and n_away >= MIN_PARTIDOS_MEDIA):
        return "media"

    # ── BAJA — solo con cuotas reales y mercados estándar ─────────────────
    if (is_real and
        market in _MERCADOS_NIVEL_BAJA and
        edge >= UMBRAL_EDGE_BAJA and
        model_prob >= UMBRAL_PROB_BAJA and
        n_home >= MIN_PARTIDOS_BAJA and
        n_away >= MIN_PARTIDOS_BAJA):
        return "baja"

    return None


# ─── CÁLCULOS CORE ───────────────────────────────────────────────────────────

def calculate_edge(model_prob: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0 or model_prob <= 0:
        return -999.0
    return round((model_prob * decimal_odds - 1) * 100, 2)


def kelly_fraction(model_prob: float, decimal_odds: float,
                   fraction: float = KELLY_FRACCION,
                   confidence: str = "media") -> float:
    """
    Calcula el Kelly criterion fraccionado.
    El nivel 'baja' usa el 50% de la fracción normal para reflejar
    la menor certeza estadística.
    """
    if decimal_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b     = decimal_odds - 1
    kelly = (model_prob * b - (1 - model_prob)) / b
    # Reducir fracción para nivel baja
    effective_fraction = fraction * 0.5 if confidence == "baja" else fraction
    return round(max(kelly * effective_fraction, 0.0) * 100, 2)


def get_model_prob_for_market(market: str, market_probs: dict) -> float:
    return market_probs.get(f"prob_{market}", 0.0)


def build_explanation(market: str, fixture_row: dict,
                      market_probs: dict, top_features: list) -> str:
    mu  = market_probs.get("exp_home_goals", 0)
    lam = market_probs.get("exp_away_goals", 0)

    base = {
        "over05":        f"modelo espera {mu+lam:.1f} goles totales",
        "over15":        f">{1.5} goles — modelo espera {mu+lam:.1f}",
        "over25":        f"modelo espera {mu+lam:.1f} goles ({mu:.1f}+{lam:.1f})",
        "under25":       f"solo {mu+lam:.1f} goles esperados en total",
        "over35":        f">{3.5} goles — modelo espera {mu+lam:.1f}",
        "over45":        f">{4.5} goles — modelo espera {mu+lam:.1f}",
        "btts_si":       f"local {mu:.1f} esp. | visitante {lam:.1f} esp.",
        "btts_no":       f"uno de los dos equipos podría no marcar",
        "home_over05":   f"local espera {mu:.1f} goles",
        "away_over05":   f"visitante espera {lam:.1f} goles",
        "home_over15":   f"local necesita 2+ — espera {mu:.1f}",
        "away_over15":   f"visitante necesita 2+ — espera {lam:.1f}",
        "home_under05":  f"local espera solo {mu:.1f} goles",
        "away_under05":  f"visitante espera solo {lam:.1f} goles",
        "exact_0":       f"0-0 — {mu+lam:.1f} goles esperados es bajo",
        "exact_1":       f"exactamente 1 gol — prob. calculada desde DC",
        "exact_2":       f"exactamente 2 goles — prob. calculada desde DC",
        "exact_3":       f"exactamente 3 goles — prob. calculada desde DC",
        "exact_4plus":   f"4+ goles — modelo espera {mu+lam:.1f}",
        "home_and_btts": f"local gana Y ambos marcan — {mu:.1f} vs {lam:.1f} esp.",
        "away_and_btts": f"visitante gana Y ambos marcan — {lam:.1f} vs {mu:.1f} esp.",
        "draw_and_btts": f"empate Y ambos marcan — {mu:.1f} vs {lam:.1f} esp.",
        "ah_home_minus05": f"local gana (AH -0.5) — espera {mu:.1f} goles",
        "ah_away_minus05": f"visitante gana (AH -0.5) — espera {lam:.1f} goles",
        "ah_home_plus05":  f"local gana o empata (AH +0.5)",
        "ah_away_plus05":  f"visitante gana o empata (AH +0.5)",
        "ah_home_minus1":  f"local gana por 2+ (AH -1) — espera {mu:.1f}",
    }

    templates = {
        "home_forma":       lambda v: f"forma local {v:.1f}pts/PJ",
        "away_forma":       lambda v: f"forma visitante {v:.1f}pts/PJ",
        "elo_diff":         lambda v: f"ventaja ELO {abs(v):.0f}pts {'(L)' if v>0 else '(V)'}",
        "h2h_avg_goals":    lambda v: f"H2H promedio {v:.1f} goles",
        "h2h_btts_rate":    lambda v: f"BTTS en {v*100:.0f}% de H2H",
        "home_over25_rate": lambda v: f"local Over2.5 en {v*100:.0f}%",
        "away_over25_rate": lambda v: f"visit. Over2.5 en {v*100:.0f}%",
        "home_days_rest":   lambda v: f"local {v:.0f}d descanso",
        "away_days_rest":   lambda v: f"visitante {v:.0f}d descanso",
        "fatiga_flag":      lambda v: "fatiga (<4d)" if v else "",
    }

    parts = [base.get(market, "")]
    for feat in (top_features or [])[:2]:
        val = fixture_row.get(feat)
        if val is not None and feat in templates:
            try:
                txt = templates[feat](float(val))
                if txt and txt not in parts:
                    parts.append(txt)
            except Exception:
                pass
    return " | ".join(p for p in parts if p) or "análisis estadístico"


# ─── TODOS LOS MERCADOS ───────────────────────────────────────────────────────

ALL_MARKETS = [
    "home_win", "draw", "away_win",
    "btts_si", "btts_no",
    "over15", "under15", "over25", "under25", "over35", "under35", "over45",
    "home_over05", "home_under05", "home_over15", "home_under15",
    "away_over05", "away_under05", "away_over15", "away_under15",
    "exact_0", "exact_1", "exact_2", "exact_3", "exact_4plus",
    "home_and_btts", "away_and_btts", "draw_and_btts",
    "double_1x", "double_x2", "double_12",
    "ah_home_minus05", "ah_away_minus05",
    "ah_home_plus05",  "ah_away_plus05",
    "ah_home_minus1",  "ah_away_minus1",
]

MARKET_DISPLAY = {
    "home_win":         lambda h, a: f"Victoria {h}",
    "draw":             lambda h, a: "Empate",
    "away_win":         lambda h, a: f"Victoria {a}",
    "btts_si":          lambda h, a: "Ambos marcan — SÍ",
    "btts_no":          lambda h, a: "Ambos marcan — NO",
    "over15":           lambda h, a: "Over 1.5 goles",
    "under15":          lambda h, a: "Under 1.5 goles",
    "over25":           lambda h, a: "Over 2.5 goles",
    "under25":          lambda h, a: "Under 2.5 goles",
    "over35":           lambda h, a: "Over 3.5 goles",
    "under35":          lambda h, a: "Under 3.5 goles",
    "over45":           lambda h, a: "Over 4.5 goles",
    "home_over05":      lambda h, a: f"{h} marca 1+ goles",
    "home_under05":     lambda h, a: f"{h} no marca",
    "home_over15":      lambda h, a: f"{h} marca 2+ goles",
    "home_under15":     lambda h, a: f"{h} marca 0 ó 1 gol",
    "away_over05":      lambda h, a: f"{a} marca 1+ goles",
    "away_under05":     lambda h, a: f"{a} no marca",
    "away_over15":      lambda h, a: f"{a} marca 2+ goles",
    "away_under15":     lambda h, a: f"{a} marca 0 ó 1 gol",
    "exact_0":          lambda h, a: "Goles exactos: 0 (0-0)",
    "exact_1":          lambda h, a: "Goles exactos: 1",
    "exact_2":          lambda h, a: "Goles exactos: 2",
    "exact_3":          lambda h, a: "Goles exactos: 3",
    "exact_4plus":      lambda h, a: "4 o más goles en total",
    "home_and_btts":    lambda h, a: f"{h} gana + ambos marcan",
    "draw_and_btts":    lambda h, a: "Empate + ambos marcan",
    "away_and_btts":    lambda h, a: f"{a} gana + ambos marcan",
    "double_1x":        lambda h, a: f"Doble: {h} o Empate",
    "double_x2":        lambda h, a: f"Doble: Empate o {a}",
    "double_12":        lambda h, a: f"Doble: {h} o {a}",
    "ah_home_minus05":  lambda h, a: f"AH: {h} -0.5",
    "ah_away_minus05":  lambda h, a: f"AH: {a} -0.5",
    "ah_home_plus05":   lambda h, a: f"AH: {h} +0.5",
    "ah_away_plus05":   lambda h, a: f"AH: {a} +0.5",
    "ah_home_minus1":   lambda h, a: f"AH: {h} -1 (gana por 2+)",
    "ah_away_minus1":   lambda h, a: f"AH: {a} -1 (gana por 2+)",
}


# ─── ANALIZADOR PRINCIPAL ────────────────────────────────────────────────────

def analyze_fixture(fixture_row: pd.Series, predictions: dict,
                    historical: pd.DataFrame = None) -> list:
    home   = fixture_row.get("home_team", "")
    away   = fixture_row.get("away_team", "")
    n_home = int(fixture_row.get("n_home_matches", 0))
    n_away = int(fixture_row.get("n_away_matches", 0))

    mu  = predictions.get("dc_exp_home_goals", 1.4)
    lam = predictions.get("dc_exp_away_goals", 1.1)

    market_probs = compute_all_market_probs(mu, lam)

    for blend_key in ["prob_home_win", "prob_draw", "prob_away_win",
                       "prob_btts", "prob_over25", "prob_under25"]:
        if blend_key in predictions:
            market_probs[blend_key] = predictions[blend_key]

    spread_line = fixture_row.get("espn_spread_line")
    if spread_line is not None:
        p_h_covers, p_a_covers = _compute_ah_prob_from_spread(
            float(spread_line), mu, lam
        )
        sl = float(spread_line)
        if abs(sl - (-0.5)) < 0.3:
            market_probs["prob_ah_home_minus05"] = p_h_covers
            market_probs["prob_ah_away_plus05"]  = p_a_covers
        elif abs(sl - 0.5) < 0.3:
            market_probs["prob_ah_home_plus05"]  = p_h_covers
            market_probs["prob_ah_away_minus05"] = p_a_covers
        elif sl <= -0.9:
            market_probs["prob_ah_home_minus1"]  = p_h_covers
            market_probs["prob_ah_away_minus1"]  = p_a_covers

    reference_odds, odds_are_real, odds_method = build_odds_dict(
        home, away, historical, fixture_row, market_probs
    )

    top_features = predictions.get("top_features", {})

    bets = []
    for market in ALL_MARKETS:
        model_prob = get_model_prob_for_market(market, market_probs)
        if model_prob <= 0.01:
            continue

        odds = reference_odds.get(market)
        if odds is None or odds <= 1.0:
            continue

        edge       = calculate_edge(model_prob, odds)
        confidence = classify_confidence(
            edge, model_prob, n_home, n_away, odds_method, market
        )
        if confidence is None:
            continue

        # Kelly reducido al 50% para nivel baja
        kelly       = kelly_fraction(model_prob, odds, confidence=confidence)
        market_type = market.split("_")[0] if "_" in market else market
        feats       = top_features.get(market_type, [])
        explanation = build_explanation(
            market, fixture_row.to_dict(), market_probs, feats
        )
        display = MARKET_DISPLAY.get(market, lambda h, a: market)(home, away)

        bets.append({
            "home_team":        home,
            "away_team":        away,
            "league":           fixture_row.get("league_name", ""),
            "match_date":       fixture_row.get("match_date", str(date.today())),
            "market":           market,
            "market_display":   display,
            "model_prob":       round(model_prob, 4),
            "model_prob_pct":   f"{model_prob*100:.1f}%",
            "reference_odds":   round(odds, 3),
            "odds_source":      odds_method,
            "odds_are_real":    odds_are_real,
            "edge_pct":         edge,
            "kelly_pct":        kelly,
            "confidence":       confidence,
            "explanation":      explanation,
            "exp_home_goals":   round(mu,       2),
            "exp_away_goals":   round(lam,      2),
            "exp_total_goals":  round(mu + lam, 2),
            "n_home_matches":   n_home,
            "n_away_matches":   n_away,
        })

    bets.sort(key=lambda x: (
        {"alta": 0, "media": 1, "baja": 2}.get(x["confidence"], 9),
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
            log.info(f"{home} vs {away}: {len(bets)} bet(s) — "
                     f"{[b['market_display'] for b in bets]}")
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
        return {"total": 0, "alta": 0, "media": 0, "baja": 0}
    result = {
        "total":    len(bets_df),
        "alta":     len(bets_df[bets_df["confidence"] == "alta"]),
        "media":    len(bets_df[bets_df["confidence"] == "media"]),
        "baja":     len(bets_df[bets_df["confidence"] == "baja"]),
        "edge_max": round(bets_df["edge_pct"].max(), 2),
        "edge_avg": round(bets_df["edge_pct"].mean(), 2),
    }
    if "odds_source" in bets_df.columns:
        for method in ["espn_live", "exact_match", "contextual_avg",
                       "model_implied", "fd_historical"]:
            count = len(bets_df[bets_df["odds_source"] == method])
            if count:
                result[f"odds_{method}"] = count
    if "market" in bets_df.columns:
        result["top_mercados"] = bets_df["market"].value_counts().head(5).to_dict()
    return result