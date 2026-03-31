"""
_05_result_updater.py — v2
Cierra las predicciones del día anterior con el resultado real.

CAMBIOS v2:
  1. evaluate_bet — cubre TODOS los mercados del sistema:
       over/under 0.5–4.5, por equipo (home/away over/under 0.5–1.5),
       goles exactos (0–4plus), combinadas (resultado+btts),
       Asian Handicap ±0.5 y -1.
  2. compute_model_stats — añade desglose por nivel "baja" + stats por mercado.
  3. create_supabase_tables_sql — CHECK incluye 'baja' (DDL v2).
  4. get_results_from_api — añade fallback ESPN para resultados del día.

FUENTES de resultados (en orden de prioridad):
  1. football-data.org  (get_results_fdorg)  — free permanente
  2. ESPN API           (get_results_espn)   — sin key, cualquier liga
  3. API-Football       (fallback)            — requiere key
"""

import os
import sys
import logging
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    API_FOOTBALL_KEY, SUPABASE_URL, SUPABASE_KEY,
    DATA_PROCESSED, LIGAS,
    LIGAS_ESPN, LIGAS_ESPN_ACTIVAS,
    ESPN_SITE_V2,
)

try:
    from src._01_data_collector import get_results_fdorg
    _HAS_FDORG = True
except ImportError as e:
    _HAS_FDORG = False
    logging.getLogger(__name__).warning(f"get_results_fdorg no disponible: {e}")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── SUPABASE ────────────────────────────────────────────────────────────────

def init_supabase():
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        log.error(f"Error conectando Supabase: {e}")
        return None


# ─── FUENTES DE RESULTADOS ───────────────────────────────────────────────────

def get_results_espn(target_date: str) -> dict:
    """
    Obtiene resultados del día desde ESPN API (sin key).
    Cubre todas las ligas en LIGAS_ESPN_ACTIVAS.
    Retorna dict {(home_team, away_team): {home_goals, away_goals, status}}
    """
    results = {}
    date_compact = target_date.replace("-", "")   # YYYYMMDD

    for slug in LIGAS_ESPN_ACTIVAS:
        url = f"{ESPN_SITE_V2}/{slug}/scoreboard"
        try:
            resp = requests.get(
                url,
                params={"dates": date_compact},
                timeout=10,
                headers={"Accept-Encoding": "gzip"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"ESPN results [{slug}] error: {e}")
            continue

        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]

            status = (comp.get("status", {})
                          .get("type", {})
                          .get("name", ""))
            if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home_c or not away_c:
                continue

            def _parse_score(c):
                s = c.get("score", "0")
                if isinstance(s, dict):
                    s = s.get("displayValue", "0")
                try:
                    return int(str(s).replace(",", ".").split(".")[0])
                except (ValueError, TypeError):
                    return 0

            home_name = home_c.get("team", {}).get("displayName", "")
            away_name = away_c.get("team", {}).get("displayName", "")
            hg = _parse_score(home_c)
            ag = _parse_score(away_c)

            if home_name and away_name:
                results[(home_name, away_name)] = {
                    "home_goals": hg,
                    "away_goals": ag,
                    "status": "FT",
                }

    log.info(f"Resultados ESPN: {len(results)} partidos de {len(LIGAS_ESPN_ACTIVAS)} ligas")
    return results


def get_results_from_api(target_date: str) -> dict:
    """
    Obtiene resultados del día.
    Prioridad: football-data.org → ESPN → API-Football (fallback).
    """
    # 1. football-data.org (fuente principal para ligas EU)
    if _HAS_FDORG:
        try:
            results = get_results_fdorg(target_date)
            if results:
                log.info(f"Resultados fd.org: {len(results)} partidos")
                return results
        except Exception as e:
            log.warning(f"fd.org falló: {e}. Intentando ESPN...")

    # 2. ESPN (sin key — cubre LATAM + Champions)
    espn_results = get_results_espn(target_date)
    if espn_results:
        return espn_results

    # 3. API-Football (fallback con key)
    if not API_FOOTBALL_KEY:
        log.warning("Sin resultados disponibles — ni fd.org, ni ESPN, ni API_FOOTBALL_KEY.")
        return {}

    results = {}
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key":  API_FOOTBALL_KEY,
    }
    for league_id, (league_name, _, __) in LIGAS.items():
        try:
            resp = requests.get(
                url, headers=headers,
                params={"date": target_date, "league": league_id,
                        "season": 2024, "status": "FT"},
                timeout=10,
            )
            resp.raise_for_status()
            for f in resp.json().get("response", []):
                teams = f["teams"]
                goals = f["goals"]
                hg = goals.get("home", 0) or 0
                ag = goals.get("away", 0) or 0
                results[(teams["home"]["name"], teams["away"]["name"])] = {
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "status": "FT",
                }
        except Exception as e:
            log.warning(f"Error API-Football {league_name}: {e}")

    log.info(f"Resultados API-Football: {len(results)} partidos")
    return results


# ─── EVALUATE BET — TODOS LOS MERCADOS ───────────────────────────────────────

def evaluate_bet(market: str, home_goals: int, away_goals: int) -> bool:
    """
    Evalúa si una apuesta ganó dado el resultado real.

    Mercados soportados (alineados con _04_value_detector.py):
      1X2            : home_win, draw, away_win
      Doble oport.   : double_1x, double_x2, double_12
      Over/Under     : over05–over45, under05–under45
      BTTS           : btts_si, btts_no
      Por equipo     : home_over05, home_under05, home_over15, home_under15
                       away_over05, away_under05, away_over15, away_under15
      Goles exactos  : exact_0, exact_1, exact_2, exact_3, exact_4plus
      Combinadas     : home_and_btts, draw_and_btts, away_and_btts
      Asian Handicap : ah_home_minus05, ah_away_minus05
                       ah_home_plus05,  ah_away_plus05
                       ah_home_minus1,  ah_away_minus1

    Retorna False (no explota) para mercados desconocidos.
    """
    total = home_goals + away_goals
    btts  = home_goals > 0 and away_goals > 0

    evaluators = {
        # ── 1X2 ──────────────────────────────────────────────────────────
        "home_win":  lambda: home_goals > away_goals,
        "draw":      lambda: home_goals == away_goals,
        "away_win":  lambda: home_goals < away_goals,

        # ── Doble oportunidad ─────────────────────────────────────────────
        "double_1x": lambda: home_goals >= away_goals,
        "double_x2": lambda: away_goals >= home_goals,
        "double_12": lambda: home_goals != away_goals,

        # ── BTTS ─────────────────────────────────────────────────────────
        "btts_si":   lambda: btts,
        "btts_no":   lambda: not btts,

        # ── Over/Under totales ────────────────────────────────────────────
        "over05":    lambda: total > 0.5,
        "under05":   lambda: total < 0.5,
        "over15":    lambda: total > 1.5,
        "under15":   lambda: total < 1.5,
        "over25":    lambda: total > 2.5,
        "under25":   lambda: total < 2.5,
        "over35":    lambda: total > 3.5,
        "under35":   lambda: total < 3.5,
        "over45":    lambda: total > 4.5,
        "under45":   lambda: total < 4.5,

        # ── Por equipo ────────────────────────────────────────────────────
        "home_over05":  lambda: home_goals > 0.5,
        "home_under05": lambda: home_goals < 0.5,
        "home_over15":  lambda: home_goals > 1.5,
        "home_under15": lambda: home_goals < 1.5,
        "away_over05":  lambda: away_goals > 0.5,
        "away_under05": lambda: away_goals < 0.5,
        "away_over15":  lambda: away_goals > 1.5,
        "away_under15": lambda: away_goals < 1.5,

        # ── Goles exactos ─────────────────────────────────────────────────
        "exact_0":    lambda: total == 0,
        "exact_1":    lambda: total == 1,
        "exact_2":    lambda: total == 2,
        "exact_3":    lambda: total == 3,
        "exact_4plus": lambda: total >= 4,

        # ── Combinadas (resultado + ambos marcan) ─────────────────────────
        "home_and_btts": lambda: (home_goals > away_goals) and btts,
        "draw_and_btts": lambda: (home_goals == away_goals) and btts,
        "away_and_btts": lambda: (home_goals < away_goals) and btts,

        # ── Asian Handicap ────────────────────────────────────────────────
        # ±0.5: sin empate posible
        "ah_home_minus05": lambda: home_goals > away_goals,          # local cubre -0.5
        "ah_away_minus05": lambda: away_goals > home_goals,          # visitante cubre -0.5
        "ah_home_plus05":  lambda: home_goals >= away_goals,         # local cubre +0.5
        "ah_away_plus05":  lambda: away_goals >= home_goals,         # visitante cubre +0.5
        # -1: local gana por 2+ / visitante no pierde por 2+
        "ah_home_minus1":  lambda: (home_goals - away_goals) >= 2,
        "ah_away_minus1":  lambda: (home_goals - away_goals) < 2,
    }

    evaluator = evaluators.get(market)
    if evaluator is None:
        return False
    return bool(evaluator())


# ─── SUPABASE: guardar / actualizar predicciones ─────────────────────────────

def save_predictions_to_supabase(bets_df: pd.DataFrame, sb) -> int:
    if bets_df.empty or sb is None:
        return 0
    inserted = 0
    for _, bet in bets_df.iterrows():
        try:
            record = {
                "partido":          f"{bet['home_team']} vs {bet['away_team']}",
                "home_team":        bet["home_team"],
                "away_team":        bet["away_team"],
                "liga":             bet.get("league", ""),
                "fecha_partido":    str(bet.get("match_date", date.today())),
                "mercado":          bet["market"],
                "mercado_display":  bet["market_display"],
                "prediccion":       bet["market_display"],
                "prob_modelo":      float(bet["model_prob"]),
                "cuota_referencia": float(bet["reference_odds"]),
                "edge_pct":         float(bet["edge_pct"]),
                "kelly_pct":        float(bet["kelly_pct"]),
                "confianza":        bet["confidence"],
                "explicacion":      bet.get("explanation", ""),
                "resultado":        None,
                "ganada":           None,
                "creado_en":        datetime.utcnow().isoformat(),
                "modelo_version":   date.today().strftime("%Y%m%d"),
            }
            sb.table("predicciones").insert(record).execute()
            inserted += 1
        except Exception as e:
            log.warning(f"Error insertando en Supabase: {e}")
    log.info(f"Predicciones guardadas en Supabase: {inserted}")
    return inserted


def update_results_in_supabase(target_date: str, real_results: dict, sb) -> int:
    if sb is None:
        return 0
    try:
        response = (
            sb.table("predicciones")
            .select("*")
            .eq("fecha_partido", target_date)
            .is_("ganada", "null")
            .execute()
        )
        pending = response.data
    except Exception as e:
        log.error(f"Error consultando Supabase: {e}")
        return 0

    updated = 0
    for pred in pending:
        home   = pred.get("home_team", "")
        away   = pred.get("away_team", "")
        result = real_results.get((home, away))

        # Fuzzy fallback por substring
        if not result:
            for (h, a), r in real_results.items():
                if (home.lower() in h.lower() or h.lower() in home.lower()) and \
                   (away.lower() in a.lower() or a.lower() in away.lower()):
                    result = r
                    break

        if not result:
            continue

        hg  = result["home_goals"]
        ag  = result["away_goals"]
        won = evaluate_bet(pred["mercado"], hg, ag)

        try:
            sb.table("predicciones").update({
                "resultado":  f"{hg}-{ag}",
                "home_goals": hg,
                "away_goals": ag,
                "ganada":     won,
                "cerrado_en": datetime.utcnow().isoformat(),
            }).eq("id", pred["id"]).execute()
            updated += 1
            log.info(
                f"{home} vs {away} | {pred['mercado']} → "
                f"{hg}-{ag} → {'GANADA ✅' if won else 'perdida ❌'}"
            )
        except Exception as e:
            log.warning(f"Error actualizando Supabase: {e}")

    return updated


# ─── ESTADÍSTICAS DEL MODELO ─────────────────────────────────────────────────

def compute_model_stats(sb) -> dict:
    """
    Calcula ROI y tasa de acierto total + por nivel (alta/media/baja)
    + por mercado. Guarda snapshot en tabla estadisticas_modelo.
    """
    if sb is None:
        return {}
    try:
        response = (
            sb.table("predicciones")
            .select("*")
            .not_.is_("ganada", "null")
            .execute()
        )
        df = pd.DataFrame(response.data)
    except Exception as e:
        log.error(f"Error consultando estadísticas: {e}")
        return {}

    if df.empty:
        return {"total": 0}

    df["ganada"]     = df["ganada"].astype(bool)
    df["roi_unidad"] = df.apply(
        lambda r: (float(r["cuota_referencia"]) - 1) if r["ganada"] else -1.0,
        axis=1,
    )

    total   = len(df)
    ganadas = int(df["ganada"].sum())
    tasa    = ganadas / total if total > 0 else 0
    roi     = df["roi_unidad"].mean() * 100

    stats: dict = {
        "total":    total,
        "ganadas":  ganadas,
        "tasa_pct": round(tasa * 100, 1),
        "roi_pct":  round(roi, 2),
    }

    # ── Desglose por nivel: alta / media / baja ───────────────────────────
    for nivel in ("alta", "media", "baja"):
        sub = df[df["confianza"] == nivel]
        if sub.empty:
            continue
        stats[f"n_{nivel}"]         = len(sub)
        stats[f"ganadas_{nivel}"]   = int(sub["ganada"].sum())
        stats[f"tasa_{nivel}_pct"]  = round(sub["ganada"].mean() * 100, 1)
        stats[f"roi_{nivel}_pct"]   = round(sub["roi_unidad"].mean() * 100, 2)

    # ── Desglose por mercado ──────────────────────────────────────────────
    for market, group in df.groupby("mercado"):
        n = len(group)
        stats[f"n_{market}"]         = n
        stats[f"tasa_{market}_pct"]  = round(group["ganada"].mean() * 100, 1)
        stats[f"roi_{market}_pct"]   = round(group["roi_unidad"].mean() * 100, 2)

    log.info(
        f"Stats → Total: {total} | Tasa: {stats['tasa_pct']}% | ROI: {stats['roi_pct']}%"
    )
    for nivel in ("alta", "media", "baja"):
        if f"tasa_{nivel}_pct" in stats:
            log.info(
                f"  {nivel.upper():5s}: n={stats.get(f'n_{nivel}', 0):3d} | "
                f"tasa={stats[f'tasa_{nivel}_pct']}% | "
                f"ROI={stats[f'roi_{nivel}_pct']:+.1f}%"
            )

    # ── Guardar snapshot ──────────────────────────────────────────────────
    try:
        sb.table("estadisticas_modelo").insert({
            "fecha":               str(date.today()),
            "total_predicciones":  total,
            "tasa_acierto":        round(tasa, 4),
            "roi_acumulado":       round(roi / 100, 4),
            "calculado_en":        datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"Error guardando stats en Supabase: {e}")

    return stats


# ─── DDL SUPABASE (v2) ───────────────────────────────────────────────────────

def create_supabase_tables_sql() -> str:
    """
    Genera el DDL completo para Supabase.

    v2 respecto al DDL original:
      - CHECK confianza incluye 'baja'
      - columna home_goals / away_goals en predicciones
      - índice adicional por mercado
      - vista roi_por_mercado con desglose por nivel
    """
    return """
-- ═══════════════════════════════════════════════════════════════════════
-- FOOTBOT — DDL v2
-- Ejecutar en Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS predicciones (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    partido          TEXT NOT NULL,
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    liga             TEXT,
    fecha_partido    DATE NOT NULL,
    mercado          TEXT NOT NULL,
    mercado_display  TEXT,
    prediccion       TEXT,
    prob_modelo      NUMERIC(6,4),
    cuota_referencia NUMERIC(6,3),
    edge_pct         NUMERIC(7,2),
    kelly_pct        NUMERIC(6,2),
    confianza        TEXT CHECK (confianza IN ('alta','media','baja')),
    explicacion      TEXT,
    resultado        TEXT,
    home_goals       INTEGER,
    away_goals       INTEGER,
    ganada           BOOLEAN,
    creado_en        TIMESTAMPTZ DEFAULT NOW(),
    cerrado_en       TIMESTAMPTZ,
    modelo_version   TEXT
);

CREATE TABLE IF NOT EXISTS estadisticas_modelo (
    id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    fecha               DATE NOT NULL,
    total_predicciones  INTEGER,
    tasa_acierto        NUMERIC(6,4),
    roi_acumulado       NUMERIC(8,4),
    calculado_en        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_fecha     ON predicciones(fecha_partido);
CREATE INDEX IF NOT EXISTS idx_pred_confianza ON predicciones(confianza);
CREATE INDEX IF NOT EXISTS idx_pred_ganada    ON predicciones(ganada);
CREATE INDEX IF NOT EXISTS idx_pred_mercado   ON predicciones(mercado);

-- Vista: ROI por mercado y nivel de confianza
CREATE OR REPLACE VIEW roi_por_mercado AS
SELECT
    mercado,
    confianza,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN ganada THEN 1 ELSE 0 END)                          AS ganadas,
    ROUND(AVG(CASE WHEN ganada THEN 1.0 ELSE 0.0 END) * 100, 1)     AS tasa_acierto_pct,
    ROUND(AVG(CASE WHEN ganada THEN cuota_referencia - 1 ELSE -1 END) * 100, 2) AS roi_pct
FROM predicciones
WHERE ganada IS NOT NULL
GROUP BY mercado, confianza
ORDER BY roi_pct DESC;

-- Vista: resumen diario
CREATE OR REPLACE VIEW resumen_diario AS
SELECT
    fecha_partido,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN ganada THEN 1 ELSE 0 END)                          AS ganadas,
    ROUND(AVG(CASE WHEN ganada THEN 1.0 ELSE 0.0 END) * 100, 1)     AS tasa_pct,
    ROUND(AVG(CASE WHEN ganada THEN cuota_referencia - 1 ELSE -1 END) * 100, 2) AS roi_pct
FROM predicciones
WHERE ganada IS NOT NULL
GROUP BY fecha_partido
ORDER BY fecha_partido DESC;
"""


# ─── PIPELINE PRINCIPAL ──────────────────────────────────────────────────────

def run():
    log.info("═══ FOOTBOT · Actualización de resultados ═══")
    sb        = init_supabase()
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"Obteniendo resultados del {yesterday}...")

    real_results = get_results_from_api(yesterday)
    if real_results:
        updated = update_results_in_supabase(yesterday, real_results, sb)
        log.info(f"Predicciones cerradas: {updated}")
    else:
        log.warning("No se encontraron resultados para ayer.")

    stats = compute_model_stats(sb)
    if stats:
        log.info(f"Estadísticas actualizadas: {stats.get('total', 0)} predicciones históricas")
    log.info("═══ Actualización completada ═══")
    return stats


if __name__ == "__main__":
    import sys
    if "--ddl" in sys.argv:
        print(create_supabase_tables_sql())
    else:
        run()