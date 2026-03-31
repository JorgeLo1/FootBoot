"""
test_result_updater.py — v2
Tests para _05_result_updater.py

CAMBIOS v2:
  - evaluate_bet: cubre TODOS los mercados expandidos
  - Tests de compute_model_stats con los 3 niveles (alta/media/baja)
  - Tests de get_results_espn (mock)
  - DDL: verifica 'baja' en CHECK, vista resumen_diario

Ejecución:
    pytest tests/test_result_updater.py -v
    pytest tests/test_result_updater.py -v -k "evaluate_bet"
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import date

from src._05_result_updater import (
    evaluate_bet,
    create_supabase_tables_sql,
    compute_model_stats,
    save_predictions_to_supabase,
    update_results_in_supabase,
)


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_bet — mercados base (regresión)
# ══════════════════════════════════════════════════════════════════════════════

BASE_CASES = [
    # (market, home_goals, away_goals, expected, descripcion)
    # ── 1X2 ──────────────────────────────────────────────────────────────────
    ("home_win",  2, 0, True,  "Local gana 2-0"),
    ("home_win",  1, 1, False, "Empate"),
    ("home_win",  0, 2, False, "Visitante gana"),
    ("draw",      1, 1, True,  "Empate 1-1"),
    ("draw",      0, 0, True,  "Empate 0-0"),
    ("draw",      2, 0, False, "Local gana, no empate"),
    ("away_win",  0, 1, True,  "Visitante gana 0-1"),
    ("away_win",  1, 0, False, "Local gana"),
    ("away_win",  2, 2, False, "Empate"),
    # ── Doble oportunidad ─────────────────────────────────────────────────────
    ("double_1x", 2, 0, True,  "Local gana → 1X OK"),
    ("double_1x", 0, 0, True,  "Empate → 1X OK"),
    ("double_1x", 0, 2, False, "Visitante gana → 1X falla"),
    ("double_x2", 0, 1, True,  "Visitante gana → X2 OK"),
    ("double_x2", 0, 0, True,  "Empate → X2 OK"),
    ("double_x2", 1, 0, False, "Local gana → X2 falla"),
    ("double_12", 2, 0, True,  "Local gana → 12 OK"),
    ("double_12", 0, 2, True,  "Visitante gana → 12 OK"),
    ("double_12", 1, 1, False, "Empate → 12 falla"),
    # ── BTTS ─────────────────────────────────────────────────────────────────
    ("btts_si",   1, 1, True,  "Ambos marcan"),
    ("btts_si",   1, 0, False, "Solo local"),
    ("btts_si",   0, 0, False, "Nadie marca"),
    ("btts_no",   1, 0, True,  "Solo local → btts_no OK"),
    ("btts_no",   0, 0, True,  "Nadie marca → btts_no OK"),
    ("btts_no",   1, 1, False, "Ambos marcan → btts_no falla"),
    # ── Over/Under 2.5 ───────────────────────────────────────────────────────
    ("over25",    2, 1, True,  "3 goles > 2.5"),
    ("over25",    1, 1, False, "2 goles no > 2.5"),
    ("under25",   1, 0, True,  "1 gol < 2.5"),
    ("under25",   2, 1, False, "3 goles no < 2.5"),
    # ── Over/Under 3.5 ───────────────────────────────────────────────────────
    ("over35",    2, 2, True,  "4 goles > 3.5"),
    ("over35",    2, 1, False, "3 goles no > 3.5"),
]

EXPANDED_CASES = [
    # ── Over/Under 0.5 ───────────────────────────────────────────────────────
    ("over05",    1, 0, True,  "1 gol > 0.5"),
    ("over05",    0, 0, False, "0 goles no > 0.5"),
    ("under05",   0, 0, True,  "0 goles < 0.5"),
    ("under05",   1, 0, False, "1 gol no < 0.5"),
    # ── Over/Under 1.5 ───────────────────────────────────────────────────────
    ("over15",    1, 1, True,  "2 goles > 1.5"),
    ("over15",    1, 0, False, "1 gol no > 1.5"),
    ("under15",   1, 0, True,  "1 gol < 1.5"),
    ("under15",   1, 1, False, "2 goles no < 1.5"),
    # ── Over/Under 4.5 ───────────────────────────────────────────────────────
    ("over45",    3, 2, True,  "5 goles > 4.5"),
    ("over45",    2, 2, False, "4 goles no > 4.5"),
    ("under45",   2, 2, True,  "4 goles < 4.5"),
    ("under45",   3, 2, False, "5 goles no < 4.5"),
    # ── Por equipo ───────────────────────────────────────────────────────────
    ("home_over05",  1, 0, True,  "Local marca 1 > 0.5"),
    ("home_over05",  0, 2, False, "Local no marca"),
    ("home_under05", 0, 2, True,  "Local no marca → under05 OK"),
    ("home_under05", 1, 0, False, "Local marca → under05 falla"),
    ("home_over15",  2, 0, True,  "Local marca 2 > 1.5"),
    ("home_over15",  1, 3, False, "Local marca 1 no > 1.5"),
    ("home_under15", 1, 3, True,  "Local marca 1 < 1.5"),
    ("home_under15", 2, 0, False, "Local marca 2 no < 1.5"),
    ("away_over05",  0, 1, True,  "Visitante marca 1 > 0.5"),
    ("away_over05",  2, 0, False, "Visitante no marca"),
    ("away_under05", 2, 0, True,  "Visitante no marca → under05 OK"),
    ("away_under05", 0, 1, False, "Visitante marca → under05 falla"),
    ("away_over15",  0, 2, True,  "Visitante marca 2 > 1.5"),
    ("away_over15",  3, 1, False, "Visitante marca 1 no > 1.5"),
    ("away_under15", 3, 1, True,  "Visitante marca 1 < 1.5"),
    ("away_under15", 0, 2, False, "Visitante marca 2 no < 1.5"),
    # ── Goles exactos ────────────────────────────────────────────────────────
    ("exact_0",    0, 0, True,  "0-0 → exact_0"),
    ("exact_0",    1, 0, False, "1 gol ≠ 0"),
    ("exact_1",    1, 0, True,  "1 gol total"),
    ("exact_1",    0, 1, True,  "1 gol total (visitante)"),
    ("exact_1",    1, 1, False, "2 goles ≠ 1"),
    ("exact_2",    1, 1, True,  "2 goles totales"),
    ("exact_2",    2, 0, True,  "2 goles totales (local)"),
    ("exact_2",    2, 1, False, "3 goles ≠ 2"),
    ("exact_3",    2, 1, True,  "3 goles totales"),
    ("exact_3",    1, 1, False, "2 goles ≠ 3"),
    ("exact_4plus", 2, 2, True,  "4 goles >= 4"),
    ("exact_4plus", 3, 2, True,  "5 goles >= 4"),
    ("exact_4plus", 2, 1, False, "3 goles < 4"),
    # ── Combinadas ───────────────────────────────────────────────────────────
    ("home_and_btts", 2, 1, True,  "Local gana y ambos marcan"),
    ("home_and_btts", 1, 0, False, "Local gana pero solo él marca"),
    ("home_and_btts", 1, 2, False, "Visitante gana"),
    ("home_and_btts", 0, 0, False, "Empate 0-0"),
    ("draw_and_btts", 1, 1, True,  "Empate y ambos marcan"),
    ("draw_and_btts", 0, 0, False, "Empate pero nadie marca"),
    ("draw_and_btts", 2, 1, False, "No hay empate"),
    ("away_and_btts", 1, 2, True,  "Visitante gana y ambos marcan"),
    ("away_and_btts", 0, 1, False, "Visitante gana pero solo él marca"),
    ("away_and_btts", 2, 1, False, "Local gana"),
    # ── Asian Handicap ────────────────────────────────────────────────────────
    ("ah_home_minus05", 1, 0, True,  "Local gana → cubre -0.5"),
    ("ah_home_minus05", 0, 0, False, "Empate → no cubre -0.5"),
    ("ah_home_minus05", 0, 1, False, "Visitante gana → no cubre"),
    ("ah_away_minus05", 0, 1, True,  "Visitante gana → cubre -0.5"),
    ("ah_away_minus05", 1, 1, False, "Empate → no cubre"),
    ("ah_home_plus05",  1, 0, True,  "Local gana → cubre +0.5"),
    ("ah_home_plus05",  0, 0, True,  "Empate → cubre +0.5"),
    ("ah_home_plus05",  0, 1, False, "Visitante gana → no cubre +0.5"),
    ("ah_away_plus05",  0, 1, True,  "Visitante gana → cubre +0.5"),
    ("ah_away_plus05",  0, 0, True,  "Empate → cubre +0.5"),
    ("ah_away_plus05",  1, 0, False, "Local gana → no cubre +0.5"),
    ("ah_home_minus1",  2, 0, True,  "Local gana por 2 → cubre -1"),
    ("ah_home_minus1",  1, 0, False, "Local gana por 1 → no cubre -1"),
    ("ah_home_minus1",  3, 1, True,  "Local gana por 2 → cubre -1"),
    ("ah_away_minus1",  1, 0, True,  "Local gana por 1 → visitante cubre -1"),
    ("ah_away_minus1",  2, 0, False, "Local gana por 2 → visitante no cubre"),
    ("ah_away_minus1",  0, 2, True,  "Visitante gana → cubre -1"),
]

ALL_CASES = BASE_CASES + EXPANDED_CASES


@pytest.mark.parametrize(
    "market, hg, ag, expected, description",
    ALL_CASES,
    ids=[f"{m}_{hg}-{ag}" for m, hg, ag, _, _ in ALL_CASES],
)
def test_evaluate_bet_all_markets(market, hg, ag, expected, description):
    result = evaluate_bet(market, hg, ag)
    assert result == expected, (
        f"evaluate_bet('{market}', {hg}, {ag}): "
        f"esperado={expected}, obtenido={result} — {description}"
    )


def test_evaluate_bet_unknown_market_returns_false():
    assert evaluate_bet("mercado_inventado_xyz", 2, 1) is False


def test_evaluate_bet_returns_bool():
    for market, hg, ag, _, _ in ALL_CASES:
        result = evaluate_bet(market, hg, ag)
        assert isinstance(result, bool), (
            f"'{market}' devolvió {type(result).__name__}, esperado bool"
        )


def test_evaluate_bet_0_0_consistency():
    """0-0 activa btts_no, exact_0, draw, under15, under25, under35, under45."""
    assert evaluate_bet("draw",     0, 0) is True
    assert evaluate_bet("btts_no",  0, 0) is True
    assert evaluate_bet("btts_si",  0, 0) is False
    assert evaluate_bet("exact_0",  0, 0) is True
    assert evaluate_bet("under05",  0, 0) is True
    assert evaluate_bet("under15",  0, 0) is True
    assert evaluate_bet("under25",  0, 0) is True
    assert evaluate_bet("over05",   0, 0) is False
    assert evaluate_bet("home_win", 0, 0) is False
    assert evaluate_bet("away_win", 0, 0) is False


def test_evaluate_bet_high_scoring_4_2():
    """4-2: activa over35, over45, btts, exact_4plus, home_win, home_and_btts."""
    hg, ag = 4, 2
    assert evaluate_bet("over25",       hg, ag) is True
    assert evaluate_bet("over35",       hg, ag) is True
    assert evaluate_bet("over45",       hg, ag) is True
    assert evaluate_bet("btts_si",      hg, ag) is True
    assert evaluate_bet("home_win",     hg, ag) is True
    assert evaluate_bet("home_and_btts",hg, ag) is True
    assert evaluate_bet("away_win",     hg, ag) is False
    assert evaluate_bet("under25",      hg, ag) is False
    assert evaluate_bet("exact_0",      hg, ag) is False


def test_evaluate_bet_1_1_combinadas():
    """1-1: draw_and_btts True, home/away_and_btts False."""
    assert evaluate_bet("draw_and_btts", 1, 1) is True
    assert evaluate_bet("home_and_btts", 1, 1) is False
    assert evaluate_bet("away_and_btts", 1, 1) is False
    assert evaluate_bet("btts_si",       1, 1) is True
    assert evaluate_bet("draw",          1, 1) is True


def test_evaluate_bet_ah_symmetry():
    """minus05 y plus05 son complementarios (sin empate posible)."""
    for hg, ag in [(1,0), (0,1), (2,1), (1,2)]:
        h_m = evaluate_bet("ah_home_minus05", hg, ag)
        a_m = evaluate_bet("ah_away_minus05", hg, ag)
        # En 1X2 sin empate siempre gana uno — minus05 son complementarios
        assert h_m != a_m, f"ah_minus05 no son complementarios para {hg}-{ag}"

    # Con empate: plus05 ambos True
    assert evaluate_bet("ah_home_plus05", 0, 0) is True
    assert evaluate_bet("ah_away_plus05", 0, 0) is True


# ══════════════════════════════════════════════════════════════════════════════
#  DDL Supabase v2
# ══════════════════════════════════════════════════════════════════════════════

def test_supabase_ddl_includes_baja():
    sql = create_supabase_tables_sql()
    assert "'baja'" in sql, (
        "El CHECK de confianza debe incluir 'baja'.\n"
        "CHECK (confianza IN ('alta','media','baja'))"
    )


def test_supabase_ddl_check_completo():
    sql = create_supabase_tables_sql()
    assert "'alta'" in sql
    assert "'media'" in sql
    assert "'baja'" in sql


def test_supabase_ddl_creates_predicciones_table():
    assert "CREATE TABLE IF NOT EXISTS predicciones" in create_supabase_tables_sql()


def test_supabase_ddl_creates_estadisticas_table():
    assert "CREATE TABLE IF NOT EXISTS estadisticas_modelo" in create_supabase_tables_sql()


def test_supabase_ddl_creates_indexes():
    assert "CREATE INDEX" in create_supabase_tables_sql()


def test_supabase_ddl_creates_roi_view():
    assert "roi_por_mercado" in create_supabase_tables_sql()


def test_supabase_ddl_creates_resumen_diario_view():
    """v2: nueva vista resumen_diario."""
    assert "resumen_diario" in create_supabase_tables_sql()


def test_supabase_ddl_has_required_columns():
    sql = create_supabase_tables_sql()
    for col in ["home_team", "away_team", "mercado", "confianza",
                "edge_pct", "kelly_pct", "ganada", "fecha_partido",
                "home_goals", "away_goals"]:
        assert col in sql, f"Columna '{col}' ausente en el DDL"


def test_supabase_ddl_returns_string():
    sql = create_supabase_tables_sql()
    assert isinstance(sql, str)
    assert len(sql) > 100


# ══════════════════════════════════════════════════════════════════════════════
#  compute_model_stats — mock Supabase
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_sb(rows: list[dict]):
    """Crea un mock de Supabase que devuelve rows al hacer .execute()."""
    mock_sb = MagicMock()
    mock_response = MagicMock()
    mock_response.data = rows
    (mock_sb.table.return_value
             .select.return_value
             .not_.return_value
             .is_.return_value
             .execute.return_value) = mock_response
    # También silenciar el insert de estadisticas_modelo
    mock_sb.table.return_value.insert.return_value.execute.return_value = MagicMock()
    return mock_sb


def _sample_row(confianza: str, ganada: bool, cuota: float = 2.0) -> dict:
    return {
        "confianza":        confianza,
        "ganada":           ganada,
        "cuota_referencia": cuota,
        "mercado":          "home_win",
    }


def test_compute_model_stats_empty_returns_zero():
    sb = _make_mock_sb([])
    stats = compute_model_stats(sb)
    assert stats.get("total", 0) == 0


def test_compute_model_stats_none_sb_returns_empty():
    stats = compute_model_stats(None)
    assert stats == {}


def test_compute_model_stats_total():
    rows = [
        _sample_row("alta",  True,  2.5),
        _sample_row("media", False, 1.8),
        _sample_row("baja",  True,  1.9),
    ]
    sb = _make_mock_sb(rows)
    stats = compute_model_stats(sb)
    assert stats["total"] == 3
    assert stats["ganadas"] == 2


def test_compute_model_stats_desglose_niveles():
    """Verifica que se calculen stats para los 3 niveles."""
    rows = (
        [_sample_row("alta",  True,  2.5)] * 5 +
        [_sample_row("alta",  False, 2.5)] * 5 +
        [_sample_row("media", True,  2.0)] * 3 +
        [_sample_row("media", False, 2.0)] * 7 +
        [_sample_row("baja",  True,  1.8)] * 2 +
        [_sample_row("baja",  False, 1.8)] * 8
    )
    sb = _make_mock_sb(rows)
    stats = compute_model_stats(sb)

    assert "tasa_alta_pct" in stats,  "Falta stats nivel alta"
    assert "tasa_media_pct" in stats, "Falta stats nivel media"
    assert "tasa_baja_pct" in stats,  "Falta stats nivel baja"

    assert stats["tasa_alta_pct"]  == 50.0
    assert stats["tasa_media_pct"] == 30.0
    assert stats["tasa_baja_pct"]  == 20.0


def test_compute_model_stats_roi_negativo_apuesta_perdida():
    """Una apuesta perdida con cualquier cuota → roi_unidad = -1."""
    rows = [_sample_row("alta", False, 3.0)]
    sb = _make_mock_sb(rows)
    stats = compute_model_stats(sb)
    assert stats["roi_pct"] == pytest.approx(-100.0, abs=0.1)


def test_compute_model_stats_roi_positivo_apuesta_ganada():
    """Una apuesta ganada a cuota 2.0 → roi_unidad = +1 → ROI = 100%."""
    rows = [_sample_row("alta", True, 2.0)]
    sb = _make_mock_sb(rows)
    stats = compute_model_stats(sb)
    assert stats["roi_pct"] == pytest.approx(100.0, abs=0.1)


def test_compute_model_stats_por_mercado():
    """Stats por mercado deben estar presentes."""
    rows = [
        {"confianza": "alta", "ganada": True,  "cuota_referencia": 2.0, "mercado": "home_win"},
        {"confianza": "alta", "ganada": False, "cuota_referencia": 1.9, "mercado": "over25"},
        {"confianza": "media","ganada": True,  "cuota_referencia": 1.85,"mercado": "over25"},
    ]
    sb = _make_mock_sb(rows)
    stats = compute_model_stats(sb)

    assert "tasa_home_win_pct" in stats
    assert "tasa_over25_pct" in stats
    assert stats["tasa_home_win_pct"] == 100.0
    assert stats["tasa_over25_pct"]   == pytest.approx(50.0, abs=0.1)


# ══════════════════════════════════════════════════════════════════════════════
#  update_results_in_supabase — cobertura mercados expandidos
# ══════════════════════════════════════════════════════════════════════════════

def _make_sb_for_update(pending_rows: list[dict], real_results: dict):
    """
    Mock Supabase para update_results_in_supabase.
    pending_rows: predicciones sin resultado.
    """
    sb = MagicMock()

    # select pendientes
    select_resp = MagicMock()
    select_resp.data = pending_rows
    (sb.table.return_value
       .select.return_value
       .eq.return_value
       .is_.return_value
       .execute.return_value) = select_resp

    # update
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

    return sb


def test_update_marks_correct_winner():
    """Un partido real cierra la predicción correctamente."""
    pred = {
        "id":         "uuid-1",
        "home_team":  "Nacional",
        "away_team":  "Santa Fe",
        "mercado":    "over25",
    }
    real = {("Nacional", "Santa Fe"): {"home_goals": 2, "away_goals": 1}}
    sb = _make_sb_for_update([pred], real)
    updated = update_results_in_supabase("2026-03-28", real, sb)
    assert updated == 1


def test_update_marks_loser():
    """Under25 pierde si hay 3 goles."""
    pred = {
        "id": "uuid-2",
        "home_team": "A", "away_team": "B",
        "mercado": "under25",
    }
    real = {("A", "B"): {"home_goals": 2, "away_goals": 1}}
    sb = _make_sb_for_update([pred], real)
    updated = update_results_in_supabase("2026-03-28", real, sb)
    assert updated == 1
    # Verificar que se llamó update con ganada=False
    call_args = sb.table.return_value.update.call_args_list[0][0][0]
    assert call_args["ganada"] is False


def test_update_no_match_skips():
    """Predicción sin resultado real → no se actualiza."""
    pred = {
        "id": "uuid-3",
        "home_team": "X", "away_team": "Y",
        "mercado": "home_win",
    }
    real = {}  # sin resultados
    sb = _make_sb_for_update([pred], real)
    updated = update_results_in_supabase("2026-03-28", real, sb)
    assert updated == 0


def test_update_fuzzy_name_match():
    """Nombre levemente diferente se resuelve por substring."""
    pred = {
        "id": "uuid-4",
        "home_team": "Atletico Nacional",
        "away_team": "Santa Fe",
        "mercado": "btts_si",
    }
    # Nombre ESPN ligeramente diferente
    real = {("Club Atletico Nacional", "Independiente Santa Fe"): {
        "home_goals": 1, "away_goals": 1,
    }}
    sb = _make_sb_for_update([pred], real)
    updated = update_results_in_supabase("2026-03-28", real, sb)
    assert updated == 1


# ══════════════════════════════════════════════════════════════════════════════
#  save_predictions_to_supabase
# ══════════════════════════════════════════════════════════════════════════════

def test_save_predictions_empty_df(sample_bets_df):
    sb = MagicMock()
    result = save_predictions_to_supabase(pd.DataFrame(), sb)
    assert result == 0


def test_save_predictions_inserts_rows(sample_bets_df):
    sb = MagicMock()
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock()
    inserted = save_predictions_to_supabase(sample_bets_df, sb)
    assert inserted == len(sample_bets_df)


def test_save_predictions_none_sb(sample_bets_df):
    result = save_predictions_to_supabase(sample_bets_df, None)
    assert result == 0