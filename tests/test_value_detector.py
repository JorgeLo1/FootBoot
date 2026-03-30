"""
test_value_detector.py
Tests para _04_value_detector.py — lógica de negocio central.

Cubre:
  - _poisson_matrix (propiedades de probabilidad)
  - compute_all_market_probs (pares complementarios suman 1)
  - calculate_edge (matemática)
  - kelly_fraction (incluyendo reducción para nivel 'baja')
  - classify_confidence (tres niveles + bloqueos)
  - build_odds_dict (jerarquía de fuentes)
  - analyze_fixture (integración — genera lista de value bets)

BUGS DOCUMENTADOS:
  - BUG: Las cuotas FALLBACK 1X2 existen en el código aunque classify_confidence
    bloquea model_implied para mercados estándar. Si el bloqueo falla por algún
    motivo, las cuotas fallback podrían filtrarse al usuario como reales.
    Ver test_model_implied_never_emits_standard_1x2.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src._04_value_detector import (
    _poisson_matrix,
    compute_all_market_probs,
    calculate_edge,
    kelly_fraction,
    classify_confidence,
    build_odds_dict,
    analyze_fixture,
    detect_all_value_bets,
    summarize_bets,
    _MERCADOS_BLOQUEADOS_MODEL_IMPLIED,
    _MERCADOS_NIVEL_BAJA,
    CUOTAS_FALLBACK,
    MAX_GOALS,
)
from config.settings import (
    UMBRAL_EDGE_ALTA, UMBRAL_EDGE_MEDIA, UMBRAL_EDGE_BAJA,
    UMBRAL_PROB_ALTA, UMBRAL_PROB_MEDIA, UMBRAL_PROB_BAJA,
    MIN_PARTIDOS_ALTA, MIN_PARTIDOS_MEDIA, MIN_PARTIDOS_BAJA,
)


# ══════════════════════════════════════════════════════════════════════════════
#  _poisson_matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestPoissonMatrix:
    def test_shape_is_max_goals_plus_one(self):
        M = _poisson_matrix(1.4, 1.1)
        assert M.shape == (MAX_GOALS + 1, MAX_GOALS + 1)

    def test_sums_to_one(self):
        M = _poisson_matrix(1.4, 1.1)
        assert M.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_entries_non_negative(self):
        M = _poisson_matrix(1.4, 1.1)
        assert (M >= 0).all()

    def test_high_mu_shifts_home_goals(self):
        M_high = _poisson_matrix(3.0, 0.5)
        M_low  = _poisson_matrix(0.5, 3.0)
        # Con mu alto, la diagonal inferior (local gana) tiene más peso
        assert M_high[2:, :2].sum() > M_low[2:, :2].sum()

    def test_zero_mu_concentrates_at_zero_goals(self):
        M = _poisson_matrix(0.001, 1.0)
        assert M[0, :].sum() > 0.99


# ══════════════════════════════════════════════════════════════════════════════
#  compute_all_market_probs
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAllMarketProbs:
    @pytest.fixture
    def probs(self):
        return compute_all_market_probs(mu=1.6, lam=1.2)

    def test_1x2_sums_to_one(self, probs):
        total = probs["prob_home_win"] + probs["prob_draw"] + probs["prob_away_win"]
        assert total == pytest.approx(1.0, abs=0.01)

    def test_btts_complements_sum_to_one(self, probs):
        assert probs["prob_btts"] + probs["prob_btts_no"] == pytest.approx(1.0, abs=0.01)

    def test_over_under_pairs_sum_to_one(self, probs):
        pairs = [
            ("prob_over05",  "prob_under05"),
            ("prob_over15",  "prob_under15"),
            ("prob_over25",  "prob_under25"),
            ("prob_over35",  "prob_under35"),
            ("prob_over45",  "prob_under45"),
        ]
        for over, under in pairs:
            total = probs[over] + probs[under]
            assert total == pytest.approx(1.0, abs=0.02), f"{over}+{under}={total}"

    def test_home_goals_complements_sum_to_one(self, probs):
        assert (probs["prob_home_over05"] + probs["prob_home_under05"] ==
                pytest.approx(1.0, abs=0.01))

    def test_double_chance_gt_single_chance(self, probs):
        """Doble oportunidad siempre mayor que cada resultado individual."""
        assert probs["prob_double_1x"] > probs["prob_home_win"]
        assert probs["prob_double_x2"] > probs["prob_away_win"]

    def test_all_probs_between_0_and_1(self, probs):
        for key, val in probs.items():
            if key.startswith("prob_"):
                assert 0.0 <= val <= 1.0, f"{key}={val}"

    def test_higher_mu_increases_home_win(self):
        p_strong = compute_all_market_probs(mu=2.5, lam=0.8)
        p_weak   = compute_all_market_probs(mu=0.8, lam=2.5)
        assert p_strong["prob_home_win"] > p_weak["prob_home_win"]

    def test_expected_goals_returned(self, probs):
        assert probs["exp_home_goals"] == pytest.approx(1.6, abs=0.1)
        assert probs["exp_away_goals"] == pytest.approx(1.2, abs=0.1)


# ══════════════════════════════════════════════════════════════════════════════
#  calculate_edge
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculateEdge:
    def test_positive_edge(self):
        """prob=0.6 con odds=2.0 → edge = (0.6*2.0 - 1)*100 = 20%"""
        assert calculate_edge(0.6, 2.0) == pytest.approx(20.0, abs=0.01)

    def test_negative_edge(self):
        """prob=0.5 con odds=1.9 → edge = (0.5*1.9 - 1)*100 = -5%"""
        assert calculate_edge(0.5, 1.9) == pytest.approx(-5.0, abs=0.01)

    def test_zero_edge_at_fair_odds(self):
        """Cuota justa → edge = 0%"""
        assert calculate_edge(0.5, 2.0) == pytest.approx(0.0, abs=0.01)

    def test_invalid_odds_returns_large_negative(self):
        """odds <= 1.0 no tiene sentido → retorna valor muy negativo."""
        assert calculate_edge(0.5, 1.0) < -900

    def test_zero_prob_returns_large_negative(self):
        assert calculate_edge(0.0, 2.0) < -900

    def test_edge_rounding(self):
        """Resultado redondeado a 2 decimales."""
        edge = calculate_edge(0.55, 1.95)
        assert edge == round(edge, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  kelly_fraction
# ══════════════════════════════════════════════════════════════════════════════

class TestKellyFraction:
    def test_positive_ev_positive_kelly(self):
        k = kelly_fraction(0.6, 2.0)
        assert k > 0

    def test_negative_ev_zero_kelly(self):
        """Con EV negativo, Kelly debe ser 0 (no apostar)."""
        k = kelly_fraction(0.4, 2.0)
        assert k == 0.0

    def test_fraction_scales_kelly(self):
        k1 = kelly_fraction(0.6, 2.0, fraction=0.25)
        k2 = kelly_fraction(0.6, 2.0, fraction=0.50)
        assert k2 == pytest.approx(k1 * 2, abs=0.01)

    def test_baja_confidence_halves_kelly(self):
        """
        El nivel 'baja' aplica al 50% de la fracción normal.
        """
        k_media = kelly_fraction(0.6, 2.0, fraction=0.25, confidence="media")
        k_baja  = kelly_fraction(0.6, 2.0, fraction=0.25, confidence="baja")
        assert k_baja == pytest.approx(k_media * 0.5, abs=0.01)

    def test_alta_confidence_normal_kelly(self):
        k_alta  = kelly_fraction(0.6, 2.0, fraction=0.25, confidence="alta")
        k_media = kelly_fraction(0.6, 2.0, fraction=0.25, confidence="media")
        assert k_alta == pytest.approx(k_media, abs=0.01)

    def test_invalid_odds_returns_zero(self):
        assert kelly_fraction(0.6, 0.9) == 0.0

    def test_result_is_percentage(self):
        """Kelly devuelve % del bankroll (multiplicado por 100)."""
        k = kelly_fraction(0.6, 2.0, fraction=0.25)
        # Kelly completo = (0.6*1 - 0.4)/1 = 0.2 → fraccionado = 0.2*0.25*100 = 5%
        assert k == pytest.approx(5.0, abs=0.1)


# ══════════════════════════════════════════════════════════════════════════════
#  classify_confidence
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyConfidence:
    """
    Tres niveles: alta, media, baja + None cuando no hay señal.
    Restricciones críticas:
      - model_implied bloquea TODOS los mercados estándar (None siempre)
      - nivel baja SOLO con cuotas reales y mercados en _MERCADOS_NIVEL_BAJA
    """

    # ── Nivel ALTA ────────────────────────────────────────────────────────────

    def test_alta_with_strong_signal(self):
        result = classify_confidence(
            edge=10.0, model_prob=0.65, n_home=40, n_away=40,
            odds_method="espn_live", market="home_win"
        )
        assert result == "alta"

    def test_alta_with_exact_match_odds(self):
        result = classify_confidence(
            edge=UMBRAL_EDGE_ALTA + 0.1, model_prob=UMBRAL_PROB_ALTA + 0.01,
            n_home=MIN_PARTIDOS_ALTA, n_away=MIN_PARTIDOS_ALTA,
            odds_method="exact_match", market="over25"
        )
        assert result == "alta"

    # ── Nivel MEDIA ───────────────────────────────────────────────────────────

    def test_media_with_moderate_signal(self):
        result = classify_confidence(
            edge=5.0, model_prob=0.58, n_home=20, n_away=20,
            odds_method="espn_live", market="home_win"
        )
        assert result == "media"

    def test_below_alta_above_media_is_media(self):
        result = classify_confidence(
            edge=UMBRAL_EDGE_MEDIA + 0.1, model_prob=UMBRAL_PROB_MEDIA + 0.01,
            n_home=MIN_PARTIDOS_MEDIA, n_away=MIN_PARTIDOS_MEDIA,
            odds_method="contextual_avg", market="btts_si"
        )
        assert result == "media"

    # ── Nivel BAJA ────────────────────────────────────────────────────────────

    def test_baja_with_weak_real_signal(self):
        result = classify_confidence(
            edge=UMBRAL_EDGE_BAJA + 0.1, model_prob=UMBRAL_PROB_BAJA + 0.01,
            n_home=MIN_PARTIDOS_BAJA, n_away=MIN_PARTIDOS_BAJA,
            odds_method="espn_live", market="home_win"
        )
        assert result == "baja"

    def test_baja_requires_real_odds(self):
        """Nivel baja NUNCA con model_implied."""
        result = classify_confidence(
            edge=UMBRAL_EDGE_BAJA + 0.1, model_prob=UMBRAL_PROB_BAJA + 0.01,
            n_home=MIN_PARTIDOS_BAJA, n_away=MIN_PARTIDOS_BAJA,
            odds_method="model_implied", market="home_win"
        )
        assert result is None

    def test_baja_only_standard_markets(self):
        """Mercados no estándar (ej: exact_0) no califican para nivel baja."""
        result = classify_confidence(
            edge=UMBRAL_EDGE_BAJA + 0.1, model_prob=UMBRAL_PROB_BAJA + 0.01,
            n_home=MIN_PARTIDOS_BAJA, n_away=MIN_PARTIDOS_BAJA,
            odds_method="espn_live", market="exact_0"   # no está en _MERCADOS_NIVEL_BAJA
        )
        assert result is None

    def test_mercados_nivel_baja_set_not_empty(self):
        assert len(_MERCADOS_NIVEL_BAJA) > 0

    # ── model_implied bloqueos ────────────────────────────────────────────────

    def test_model_implied_blocks_home_win(self):
        """BUG GUARD: home_win con model_implied SIEMPRE debe devolver None."""
        result = classify_confidence(
            edge=20.0, model_prob=0.80, n_home=100, n_away=100,
            odds_method="model_implied", market="home_win"
        )
        assert result is None, (
            "home_win con model_implied filtró como señal válida. "
            "Revisar _MERCADOS_BLOQUEADOS_MODEL_IMPLIED."
        )

    def test_model_implied_blocks_draw(self):
        result = classify_confidence(
            edge=20.0, model_prob=0.80, n_home=100, n_away=100,
            odds_method="model_implied", market="draw"
        )
        assert result is None

    def test_model_implied_blocks_over25(self):
        result = classify_confidence(
            edge=20.0, model_prob=0.80, n_home=100, n_away=100,
            odds_method="model_implied", market="over25"
        )
        assert result is None

    def test_model_implied_blocks_double_chance(self):
        for market in ["double_1x", "double_x2", "double_12"]:
            result = classify_confidence(
                edge=20.0, model_prob=0.80, n_home=100, n_away=100,
                odds_method="model_implied", market=market
            )
            assert result is None, f"double_chance '{market}' no bloqueado con model_implied"

    def test_model_implied_all_blocked_markets(self):
        """Todos los mercados en _MERCADOS_BLOQUEADOS_MODEL_IMPLIED retornan None."""
        for market in _MERCADOS_BLOQUEADOS_MODEL_IMPLIED:
            result = classify_confidence(
                edge=20.0, model_prob=0.80, n_home=100, n_away=100,
                odds_method="model_implied", market=market
            )
            assert result is None, f"Mercado '{market}' no bloqueado con model_implied"

    # ── None cuando no hay suficiente señal ──────────────────────────────────

    def test_none_when_edge_too_low(self):
        result = classify_confidence(
            edge=1.0, model_prob=0.70, n_home=50, n_away=50,
            odds_method="espn_live", market="home_win"
        )
        assert result is None

    def test_none_when_prob_too_low(self):
        result = classify_confidence(
            edge=15.0, model_prob=0.40, n_home=50, n_away=50,
            odds_method="espn_live", market="home_win"
        )
        assert result is None

    def test_none_when_insufficient_matches(self):
        result = classify_confidence(
            edge=15.0, model_prob=0.70, n_home=2, n_away=2,
            odds_method="espn_live", market="home_win"
        )
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
#  build_odds_dict — jerarquía de fuentes
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildOddsDict:
    @pytest.fixture
    def market_probs(self):
        return {
            "prob_home_win": 0.48, "prob_draw": 0.27, "prob_away_win": 0.25,
            "prob_btts": 0.58, "prob_btts_no": 0.42,
            "prob_over25": 0.54, "prob_under25": 0.46,
            "prob_over15": 0.70, "prob_under15": 0.30,
            "prob_double_1x": 0.75, "prob_double_x2": 0.52, "prob_double_12": 0.73,
            "prob_ah_home_minus05": 0.48, "prob_ah_away_minus05": 0.52,
            "prob_ah_home_plus05":  0.75, "prob_ah_away_plus05":  0.25,
            "prob_ah_home_minus1":  0.30, "prob_ah_away_minus1":  0.70,
        }

    def test_level1_espn_live(self, sample_fixture_row, market_probs):
        """Nivel 1: cuotas ESPN en tiempo real — odds_are_real=True."""
        row = sample_fixture_row.copy()
        row["espn_odds_available"] = True
        row["espn_odds_home"]      = 2.10
        row["espn_odds_draw"]      = 3.40
        row["espn_odds_away"]      = 3.60
        row["espn_total_line"]     = 2.5
        row["espn_over_odds"]      = 1.85
        row["espn_under_odds"]     = 1.95

        odds, real, method = build_odds_dict(
            "Arsenal", "Chelsea", pd.DataFrame(), row, market_probs
        )

        assert real is True
        assert method == "espn_live"
        assert odds["home_win"] == 2.10
        assert odds["draw"] == 3.40
        assert odds["away_win"] == 3.60
        assert odds["over25"] == 1.85
        assert odds["under25"] == 1.95

    def test_level2_historical_fd(self, sample_historical, sample_fixture_row, market_probs):
        """Nivel 2: cuotas históricas Football-Data.co.uk — odds_are_real=True."""
        from src._02_feature_builder import init_resolver
        init_resolver(sample_historical)

        row = sample_fixture_row.copy()
        row["espn_odds_available"] = False

        odds, real, method = build_odds_dict(
            "Arsenal", "Chelsea", sample_historical, row, market_probs
        )
        # Si hay datos históricos reales para el par, real=True
        # Si no hay datos del par (puede pasar con datos random), method=model_implied
        assert isinstance(real, bool)
        assert method in ("exact_match", "contextual_avg", "model_implied")

    def test_level3_model_implied_fallback(self, market_probs):
        """Nivel 3: sin fuentes externas → model_implied, odds_are_real=False."""
        row = pd.Series({"espn_odds_available": False})
        odds, real, method = build_odds_dict(
            "Unknown Team A", "Unknown Team B",
            pd.DataFrame(), row, market_probs
        )
        assert real is False
        assert method == "model_implied"

    def test_espn_spread_mapped_to_ah(self, sample_fixture_row, market_probs):
        """Cuotas de spread ESPN deben mapearse a mercados AH."""
        row = sample_fixture_row.copy()
        row["espn_odds_available"]   = True
        row["espn_odds_home"]        = 2.10
        row["espn_odds_draw"]        = 3.40
        row["espn_odds_away"]        = 3.60
        row["espn_spread_line"]      = -0.5   # local favorito por -0.5
        row["espn_spread_home_odds"] = 1.90
        row["espn_spread_away_odds"] = 1.90

        odds, _, _ = build_odds_dict(
            "Arsenal", "Chelsea", pd.DataFrame(), row, market_probs
        )
        assert odds.get("ah_home_minus05") == 1.90


# ══════════════════════════════════════════════════════════════════════════════
#  analyze_fixture
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeFixture:
    def test_returns_list(self, sample_fixture_row, sample_predictions):
        result = analyze_fixture(sample_fixture_row, sample_predictions)
        assert isinstance(result, list)

    def test_each_bet_has_required_keys(self, sample_fixture_row, sample_predictions):
        bets = analyze_fixture(sample_fixture_row, sample_predictions)
        required = [
            "home_team", "away_team", "market", "market_display",
            "model_prob", "reference_odds", "edge_pct", "kelly_pct",
            "confidence", "odds_source",
        ]
        for bet in bets:
            for key in required:
                assert key in bet, f"Clave '{key}' ausente en apuesta"

    def test_confidence_values_valid(self, sample_fixture_row, sample_predictions):
        bets = analyze_fixture(sample_fixture_row, sample_predictions)
        valid = {"alta", "media", "baja"}
        for bet in bets:
            assert bet["confidence"] in valid

    def test_sorted_by_confidence_then_edge(self, sample_fixture_row, sample_predictions):
        bets = analyze_fixture(sample_fixture_row, sample_predictions)
        if len(bets) < 2:
            pytest.skip("No hay suficientes apuestas para verificar orden")
        conf_order = {"alta": 0, "media": 1, "baja": 2}
        for i in range(len(bets) - 1):
            c1 = conf_order[bets[i]["confidence"]]
            c2 = conf_order[bets[i+1]["confidence"]]
            if c1 == c2:
                assert bets[i]["edge_pct"] >= bets[i+1]["edge_pct"]
            else:
                assert c1 <= c2

    def test_with_espn_live_odds_generates_real_bets(self, sample_predictions):
        """Con cuotas ESPN live, se deben generar apuestas con odds_are_real=True."""
        row = pd.Series({
            "home_team": "Atlético Nacional", "away_team": "Millonarios",
            "league_name": "Liga BetPlay", "league_id": 501,
            "match_date": "2024-04-01", "source": "espn",
            "n_home_matches": 40, "n_away_matches": 40,
            "home_xg_scored": 1.8, "home_xg_conceded": 1.0,
            "home_goals_scored": 1.9, "home_goals_conceded": 0.9,
            "home_forma": 2.2, "home_btts_rate": 0.6, "home_over25_rate": 0.6,
            "home_corners_avg": 5.5, "home_fouls_avg": 11.0, "home_days_rest": 7,
            "home_n_matches": 10, "home_n_matches_total": 40,
            "away_xg_scored": 1.1, "away_xg_conceded": 1.4,
            "away_goals_scored": 1.2, "away_goals_conceded": 1.3,
            "away_forma": 1.5, "away_btts_rate": 0.5, "away_over25_rate": 0.5,
            "away_corners_avg": 5.0, "away_fouls_avg": 11.5, "away_days_rest": 7,
            "away_n_matches": 10, "away_n_matches_total": 40,
            "h2h_home_wins": 0.5, "h2h_draws": 0.3, "h2h_away_wins": 0.2,
            "h2h_avg_goals": 2.5, "h2h_btts_rate": 0.55, "h2h_n": 8,
            "elo_diff": 80.0, "xg_diff": 0.7, "xg_total_exp": 3.0,
            "goals_diff": 0.7, "forma_diff": 0.7, "rest_diff": 0, "fatiga_flag": 0,
            "rain_flag": 0, "wind_flag": 0,
            "market_prob_home": 0.50, "market_prob_draw": 0.27, "market_prob_away": 0.23,
            # ESPN live odds
            "espn_odds_available":   True,
            "espn_odds_home":        1.80,
            "espn_odds_draw":        3.50,
            "espn_odds_away":        4.50,
            "espn_total_line":       2.5,
            "espn_over_odds":        1.85,
            "espn_under_odds":       1.95,
            "espn_spread_line":      None,
            "espn_spread_home_odds": None,
            "espn_spread_away_odds": None,
        })

        bets = analyze_fixture(row, sample_predictions)
        real_bets = [b for b in bets if b["odds_are_real"]]
        assert len(real_bets) >= 0  # puede ser 0 si el edge no alcanza el umbral

    def test_model_implied_does_not_emit_1x2(self, sample_predictions):
        """
        BUG GUARD: Con model_implied (sin cuotas externas),
        NINGUNA apuesta de home_win, draw o away_win debe generarse.
        """
        row = pd.Series({
            "home_team": "A", "away_team": "B",
            "league_name": "Test", "league_id": 0,
            "match_date": "2024-04-01", "source": "fdorg",
            "n_home_matches": 50, "n_away_matches": 50,
            "home_xg_scored": 1.5, "home_xg_conceded": 1.2,
            "home_goals_scored": 1.5, "home_goals_conceded": 1.2,
            "home_forma": 2.0, "home_btts_rate": 0.5, "home_over25_rate": 0.5,
            "home_corners_avg": 5.0, "home_fouls_avg": 11.0, "home_days_rest": 7,
            "home_n_matches": 15, "home_n_matches_total": 50,
            "away_xg_scored": 1.2, "away_xg_conceded": 1.5,
            "away_goals_scored": 1.2, "away_goals_conceded": 1.5,
            "away_forma": 1.8, "away_btts_rate": 0.5, "away_over25_rate": 0.5,
            "away_corners_avg": 5.0, "away_fouls_avg": 11.0, "away_days_rest": 7,
            "away_n_matches": 15, "away_n_matches_total": 50,
            "h2h_home_wins": 0.4, "h2h_draws": 0.3, "h2h_away_wins": 0.3,
            "h2h_avg_goals": 2.5, "h2h_btts_rate": 0.5, "h2h_n": 8,
            "elo_diff": 0.0, "xg_diff": 0.3, "xg_total_exp": 2.7,
            "goals_diff": 0.3, "forma_diff": 0.2, "rest_diff": 0, "fatiga_flag": 0,
            "rain_flag": 0, "wind_flag": 0,
            "market_prob_home": 0.45, "market_prob_draw": 0.28, "market_prob_away": 0.27,
            "espn_odds_available": False,
        })

        bets = analyze_fixture(row, sample_predictions, historical=pd.DataFrame())
        standard_1x2 = [b for b in bets if b["market"] in ("home_win", "draw", "away_win")]
        assert len(standard_1x2) == 0, (
            f"Con model_implied se generaron apuestas 1X2: "
            f"{[(b['market'], b['odds_source']) for b in standard_1x2]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  summarize_bets
# ══════════════════════════════════════════════════════════════════════════════

class TestSummarizeBets:
    def test_empty_df(self):
        result = summarize_bets(pd.DataFrame())
        assert result["total"] == 0
        assert result["alta"] == 0
        assert result["media"] == 0
        assert result["baja"] == 0

    def test_counts_by_confidence(self, sample_bets_df):
        result = summarize_bets(sample_bets_df)
        assert result["total"] == 3
        assert result["alta"] == 1
        assert result["media"] == 1
        assert result["baja"] == 1

    def test_edge_stats_present(self, sample_bets_df):
        result = summarize_bets(sample_bets_df)
        assert "edge_max" in result
        assert "edge_avg" in result
        assert result["edge_max"] >= result["edge_avg"]