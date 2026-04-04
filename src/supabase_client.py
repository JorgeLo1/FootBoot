"""
FOOTBOT — src/supabase_client.py
Cliente Supabase v3

CAMBIOS v3:
  - FIX 1: "fecha" → "fecha_partido" en guardar_prediccion() para alinearse
    con el DDL v3 y con save_predictions_to_supabase() de _05_result_updater.
  - FIX 2: "cuota" → "cuota_referencia" en guardar_prediccion() para alinearse
    con el nombre de columna real en la BD y con compute_model_stats()
    (que hace df["cuota_referencia"]).
  - FIX 3: obtener_predicciones_abiertas() filtraba por "fecha" (campo
    inexistente post-fix) → corregido a "fecha_partido".
  - FIX 4: partido (campo display) añadido al insert para consistencia
    con save_predictions_to_supabase().

Requiere: pip install supabase==2.3.0
"""

from __future__ import annotations

import os
import logging
from datetime import date
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Conexión lazy (se inicializa solo cuando se necesita) ─────────────────────

_client = None

def get_client():
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()

    if not url or not key or key == "TU_ANON_KEY_AQUI":
        raise EnvironmentError(
            "SUPABASE_URL y SUPABASE_KEY deben estar configurados en .env"
        )

    try:
        from supabase import create_client, Client
        _client = create_client(url, key)
        logger.info("Supabase conectado: %s", url)
        return _client
    except Exception as e:
        logger.error("Error conectando a Supabase: %s", e)
        raise


# ── Guardar predicción ────────────────────────────────────────────────────────

def guardar_prediccion(
    fecha: date,
    liga: str,
    league_id: int,
    home_team: str,
    away_team: str,
    mercado: str,
    prob_modelo: float,
    cuota: float,
    edge_pct: float,
    kelly_pct: float,
    confianza: str,
    odds_source: Optional[str] = None,
    odds_provider: Optional[str] = None,
    dc_exp_home: Optional[float] = None,
    dc_exp_away: Optional[float] = None,
) -> Optional[str]:
    """
    Inserta una predicción en Supabase.
    Retorna el UUID generado, o None si falla.
    """
    supabase = get_client()

    data = {
        # FIX 1: era "fecha" → columna correcta en BD es "fecha_partido"
        "fecha_partido":  str(fecha),
        "partido":        f"{home_team} vs {away_team}",   # FIX 4: campo display
        "liga":           liga,
        "league_id":      league_id,
        "home_team":      home_team,
        "away_team":      away_team,
        "mercado":        mercado,
        "prob_modelo":    round(float(prob_modelo), 4),
        # FIX 2: era "cuota" → columna correcta en BD es "cuota_referencia"
        "cuota_referencia": round(float(cuota), 3),
        "edge_pct":       round(float(edge_pct), 3),
        "kelly_pct":      round(float(kelly_pct), 4) if kelly_pct else None,
        "confianza":      confianza,
        "odds_source":    odds_source,
        "odds_provider":  odds_provider,
        "dc_exp_home":    round(float(dc_exp_home), 3) if dc_exp_home else None,
        "dc_exp_away":    round(float(dc_exp_away), 3) if dc_exp_away else None,
    }

    try:
        res = supabase.table("predicciones").insert(data).execute()
        pred_id = res.data[0]["id"] if res.data else None
        logger.info(
            "Predicción guardada id=%s | %s vs %s | %s | %s",
            pred_id, home_team, away_team, mercado, confianza
        )
        return pred_id
    except Exception as e:
        logger.error(
            "Error guardando predicción %s vs %s [%s]: %s",
            home_team, away_team, mercado, e
        )
        return None


# ── Cerrar predicción con resultado real ──────────────────────────────────────

def cerrar_prediccion(
    prediccion_id: str,
    fecha: date,
    home_team: str,
    away_team: str,
    goles_home: int,
    goles_away: int,
    mercado: str,
    resultado_bool: bool,
    profit_units: float,
) -> bool:
    """
    Inserta el resultado real de una predicción en la tabla `resultados`.
    Retorna True si se guardó correctamente.
    """
    supabase = get_client()

    data = {
        "prediccion_id":  prediccion_id,
        "fecha":          str(fecha),
        "home_team":      home_team,
        "away_team":      away_team,
        "goles_home":     goles_home,
        "goles_away":     goles_away,
        "mercado":        mercado,
        "resultado_bool": resultado_bool,
        "profit_units":   round(float(profit_units), 4),
    }

    try:
        supabase.table("resultados").insert(data).execute()
        logger.info(
            "Resultado cerrado | pred_id=%s | %s-%s | %s=%s | profit=%.3f",
            prediccion_id, goles_home, goles_away, mercado, resultado_bool, profit_units
        )
        return True
    except Exception as e:
        logger.error("Error cerrando predicción id=%s: %s", prediccion_id, e)
        return False


# ── Obtener predicciones abiertas (sin resultado) ─────────────────────────────

def obtener_predicciones_abiertas(fecha: Optional[date] = None) -> list[dict]:
    """
    Retorna predicciones que aún no tienen resultado registrado en `resultados`.
    Si se pasa fecha, filtra por esa fecha.
    """
    supabase = get_client()

    try:
        query = (
            supabase.table("predicciones")
            .select("*, resultados(id)")
            .is_("resultados.id", "null")
        )
        if fecha:
            # FIX 3: era .eq("fecha", ...) → columna correcta es "fecha_partido"
            query = query.eq("fecha_partido", str(fecha))

        res = query.order("fecha_partido", desc=True).execute()
        return res.data or []
    except Exception as e:
        logger.error("Error obteniendo predicciones abiertas: %s", e)
        return []


# ── Test de conexión ──────────────────────────────────────────────────────────

def test_conexion() -> bool:
    """Verifica que la conexión a Supabase funciona."""
    try:
        client = get_client()
        client.table("predicciones").select("id").limit(1).execute()
        print("✅ Supabase conectado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error de conexión Supabase: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_conexion()