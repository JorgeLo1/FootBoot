"""
test_result_updater.py
Tests para _05_result_updater.py

Cubre:
  - evaluate_bet: TODOS los mercados implementados (parametrizado)
  - evaluate_bet: mercado desconocido → False (no explota)
  - create_supabase_tables_sql: BUG CONOCIDO — la constraint omite 'baja'

BUG DOCUMENTADO:
  test_supabase_ddl_includes_baja FALLARÁ hasta que se corrija el DDL.
  El CHECK de la tabla predicciones solo permite ('alta','media') pero
  la v6 emite apuestas de nivel 'baja', que Supabase rechazará silenciosamente.

  Fix requerido en create_supabase_tables_sql():
    CHECK (confianza IN ('alta','media','baja'))
"""

import pytest
from src._05_result_updater import evaluate_bet, create_supabase_tables_sql


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_bet — todos los mercados
# ══════════════════════════════════════════════════════════════════════════════

# Formato: (market, home_goals, away_goals, expected_result, descripcion)
EVALUATE_BET_CASES = [
    # ── 1X2 ──────────────────────────────────────────────────────────────────
    ("home_win",  2, 0, True,  "Local gana 2-0"),
    ("home_win",  1, 1, False, "Empate, no gana local"),
    ("home_win",  0, 2, False, "Visitante gana, no local"),
    ("draw",      1, 1, True,  "Empate 1-1"),
    ("draw",      2, 0, False, "Local gana, no empate"),
    ("draw",      0, 0, True,  "Empate 0-0"),
    ("away_win",  0, 1, True,  "Visitante gana 0-1"),
    ("away_win",  1, 0, False, "Local gana, no visitante"),
    ("away_win",  2, 2, False, "Empate, no gana visitante"),
    # ── BTTS ─────────────────────────────────────────────────────────────────
    ("btts_si",   1, 1, True,  "Ambos marcan 1-1"),
    ("btts_si",   1, 0, False, "Solo local marca 1-0"),
    ("btts_si",   0, 1, False, "Solo visitante marca 0-1"),
    ("btts_si",   0, 0, False, "Nadie marca 0-0"),
    ("btts_no",   1, 0, True,  "Solo marca local → btts_no OK"),
    ("btts_no",   0, 0, True,  "Nadie marca → btts_no OK"),
    ("btts_no",   1, 1, False, "Ambos marcan → btts_no falla"),
    # ── Over/Under ────────────────────────────────────────────────────────────
    ("over25",    2, 1, True,  "3 goles > 2.5"),
    ("over25",    1, 1, False, "2 goles no > 2.5"),
    ("over25",    0, 0, False, "0 goles no > 2.5"),
    ("under25",   1, 0, True,  "1 gol < 2.5"),
    ("under25",   1, 1, True,  "2 goles < 2.5 (1+1=2, y 2 < 2.5 es True)"),
    ("under25",   2, 1, False, "3 goles no < 2.5"),
    ("over35",    2, 2, True,  "4 goles > 3.5"),
    ("over35",    2, 1, False, "3 goles no > 3.5"),
    # ── Doble oportunidad ────────────────────────────────────────────────────
    ("double_1x", 2, 0, True,  "Local gana → 1X OK"),
    ("double_1x", 0, 0, True,  "Empate → 1X OK"),
    ("double_1x", 0, 2, False, "Visitante gana → 1X falla"),
    ("double_x2", 0, 1, True,  "Visitante gana → X2 OK"),
    ("double_x2", 0, 0, True,  "Empate → X2 OK"),
    ("double_x2", 1, 0, False, "Local gana → X2 falla"),
    ("double_12", 2, 0, True,  "Local gana → 12 OK"),
    ("double_12", 0, 2, True,  "Visitante gana → 12 OK"),
    ("double_12", 1, 1, False, "Empate → 12 falla"),
]


@pytest.mark.parametrize("market, hg, ag, expected, description",
                         EVALUATE_BET_CASES,
                         ids=[f"{m}_{hg}-{ag}" for m, hg, ag, _, _ in EVALUATE_BET_CASES])
def test_evaluate_bet_parametrized(market, hg, ag, expected, description):
    result = evaluate_bet(market, hg, ag)
    assert result == expected, (
        f"evaluate_bet('{market}', {hg}, {ag}): "
        f"esperado={expected}, obtenido={result} — {description}"
    )


def test_evaluate_bet_unknown_market_returns_false():
    """Mercado desconocido no debe explotar — retorna False."""
    result = evaluate_bet("mercado_inventado_xyz", 2, 1)
    assert result is False


def test_evaluate_bet_returns_bool():
    """Todos los mercados conocidos deben devolver bool puro, no int u objeto."""
    for market, hg, ag, _, _ in EVALUATE_BET_CASES:
        result = evaluate_bet(market, hg, ag)
        assert isinstance(result, bool), f"'{market}' devolvió {type(result)}, esperado bool"


def test_evaluate_bet_0_0_draw():
    """Caso borde: 0-0 es empate y también btts_no."""
    assert evaluate_bet("draw",   0, 0) is True
    assert evaluate_bet("btts_no", 0, 0) is True
    assert evaluate_bet("btts_si", 0, 0) is False
    assert evaluate_bet("home_win", 0, 0) is False
    assert evaluate_bet("away_win", 0, 0) is False


def test_evaluate_bet_high_scoring_over35():
    """Partido de muchos goles — varios mercados activos simultáneamente."""
    hg, ag = 4, 2  # 6 goles totales
    assert evaluate_bet("over25", hg, ag) is True
    assert evaluate_bet("over35", hg, ag) is True
    assert evaluate_bet("btts_si", hg, ag) is True
    assert evaluate_bet("home_win", hg, ag) is True
    assert evaluate_bet("under25", hg, ag) is False


# ══════════════════════════════════════════════════════════════════════════════
#  create_supabase_tables_sql  — BUG CONOCIDO
# ══════════════════════════════════════════════════════════════════════════════

def test_supabase_ddl_includes_baja():
    """
    BUG: Este test FALLARÁ hasta que se corrija el DDL.

    La v6 de FOOTBOT emite apuestas de nivel 'baja', pero la constraint
    en la tabla 'predicciones' solo acepta ('alta','media').
    Resultado: todos los INSERT con confianza='baja' son rechazados por
    Supabase silenciosamente (error 23514 CHECK constraint violation).

    Fix requerido:
        CHECK (confianza IN ('alta', 'media', 'baja'))
    """
    sql = create_supabase_tables_sql()
    assert "'baja'" in sql, (
        "BUG CONFIRMADO: La constraint de Supabase no incluye 'baja'.\n"
        "Corregir en create_supabase_tables_sql():\n"
        "  CHECK (confianza IN ('alta','media'))  →  CHECK (confianza IN ('alta','media','baja'))"
    )


def test_supabase_ddl_creates_predicciones_table():
    sql = create_supabase_tables_sql()
    assert "CREATE TABLE IF NOT EXISTS predicciones" in sql


def test_supabase_ddl_creates_estadisticas_table():
    sql = create_supabase_tables_sql()
    assert "CREATE TABLE IF NOT EXISTS estadisticas_modelo" in sql


def test_supabase_ddl_creates_indexes():
    sql = create_supabase_tables_sql()
    assert "CREATE INDEX" in sql


def test_supabase_ddl_creates_roi_view():
    sql = create_supabase_tables_sql()
    assert "roi_por_mercado" in sql


def test_supabase_ddl_has_required_columns():
    sql = create_supabase_tables_sql()
    for col in ["home_team", "away_team", "mercado", "confianza",
                "edge_pct", "kelly_pct", "ganada", "fecha_partido"]:
        assert col in sql, f"Columna '{col}' ausente en el DDL"


def test_supabase_ddl_returns_string():
    sql = create_supabase_tables_sql()
    assert isinstance(sql, str)
    assert len(sql) > 100