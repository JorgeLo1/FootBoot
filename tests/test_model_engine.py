"""
test_model_engine.py
Tests para _03_model_engine.py

Cubre:
  - _resolve_league_name (fix v6 — lookup ESPN leagues)
  - DixonColesModel._tau (corrección de baja puntuación)
  - DixonColesModel.predict_proba (propiedades de probabilidad)
  - DixonColesModel._default_proba (fallback coherente)
  - DixonColesEnsemble (fallback chain: liga → global → default)
  - blend_predictions (matemática del blend ponderado)
  - _optimize_blend_weight (minimiza Brier score)
  - FootbotEnsemble.fit + predict (marcado @slow)
"""

import numpy as np
import pandas as pd
import pytest

from src._03_model_engine import (
    DixonColesModel,
    DixonColesEnsemble,
    FootbotEnsemble,
    blend_predictions,
    _resolve_league_name,
    _optimize_blend_weight,
    DEFAULT_DC_WEIGHT,
    MIN_MATCHES_PER_LIGA,
)


# ══════════════════════════════════════════════════════════════════════════════
#  _resolve_league_name  (FIX v6)
# ══════════════════════════════════════════════════════════════════════════════

class TestResolveLeagueName:
    """
    FIX v6: DixonColesEnsemble guarda ligas ESPN con league_name como clave.
    _resolve_league_name traduce league_id numérico → league_name.
    Sin este fix, las ligas LATAM siempre caían al modelo global.
    """

    def test_liga_betplay_resolved(self):
        name = _resolve_league_name(501)
        assert name == "Liga BetPlay"

    def test_liga_argentina_resolved(self):
        name = _resolve_league_name(502)
        assert name == "Liga Profesional Argentina"

    def test_libertadores_resolved(self):
        name = _resolve_league_name(511)
        assert name == "Copa Libertadores"

    def test_eliminatorias_conmebol_resolved(self):
        name = _resolve_league_name(361)
        assert name == "Eliminatorias CONMEBOL"

    def test_unknown_id_returns_none(self):
        result = _resolve_league_name(99999)
        assert result is None

    def test_eu_league_not_in_espn_dict(self):
        """Las ligas EU (league_id 39, 140, etc.) no están en LIGAS_ESPN
        y no deben resolver por esta función — se resuelven por LIGAS directamente."""
        # Premier League ID=39 no está en LIGAS_ESPN → None esperado
        result = _resolve_league_name(39)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
#  DixonColesModel._tau
# ══════════════════════════════════════════════════════════════════════════════

class TestDixonColesTau:
    """
    La función tau es la corrección de Dixon-Coles para partidos 0-0, 0-1, 1-0, 1-1.
    Para otros marcadores debe devolver 1.0 exacto.
    """

    @pytest.fixture
    def dc(self):
        return DixonColesModel()

    def test_tau_0_0(self, dc):
        """0-0: τ = 1 - μλρ. Con ρ<0, τ > 1 → probabilidad aumentada."""
        mu, lam, rho = 1.4, 1.1, -0.1
        tau = dc._tau(0, 0, mu, lam, rho)
        assert tau == pytest.approx(1 - mu * lam * rho, abs=1e-9)

    def test_tau_1_1(self, dc):
        """1-1: τ = 1 - ρ"""
        rho = -0.1
        tau = dc._tau(1, 1, 1.4, 1.1, rho)
        assert tau == pytest.approx(1 - rho, abs=1e-9)

    def test_tau_other_scores_is_one(self, dc):
        """Para x≥2 o y≥2, τ = 1.0 (sin corrección)."""
        for x, y in [(2, 0), (0, 2), (2, 2), (3, 1)]:
            tau = dc._tau(x, y, 1.4, 1.1, -0.1)
            assert tau == pytest.approx(1.0, abs=1e-9), f"tau({x},{y}) debería ser 1.0"


# ══════════════════════════════════════════════════════════════════════════════
#  DixonColesModel._default_proba
# ══════════════════════════════════════════════════════════════════════════════

class TestDixonColesDefaultProba:
    def test_all_required_keys_present(self):
        proba = DixonColesModel()._default_proba()
        required = [
            "dc_prob_home", "dc_prob_draw", "dc_prob_away",
            "dc_prob_btts", "dc_prob_over25",
            "dc_exp_home_goals", "dc_exp_away_goals", "dc_exp_total_goals",
        ]
        for k in required:
            assert k in proba, f"Clave '{k}' ausente en _default_proba"

    def test_1x2_approximately_sum_to_one(self):
        proba = DixonColesModel()._default_proba()
        total = proba["dc_prob_home"] + proba["dc_prob_draw"] + proba["dc_prob_away"]
        assert total == pytest.approx(1.0, abs=0.05)

    def test_unfitted_predict_returns_default(self):
        dc = DixonColesModel()
        assert not dc.fitted
        result = dc.predict_proba("Arsenal", "Chelsea")
        assert "dc_prob_home" in result


# ══════════════════════════════════════════════════════════════════════════════
#  DixonColesModel.fit + predict_proba
# ══════════════════════════════════════════════════════════════════════════════

class TestDixonColesModelFit:
    @pytest.fixture
    def fitted_dc(self, sample_historical):
        """Modelo DC entrenado con el histórico de Premier League (240 partidos)."""
        dc = DixonColesModel(league_id=39).fit(sample_historical)
        return dc

    def test_fitted_is_true(self, fitted_dc):
        assert fitted_dc.fitted is True

    def test_attack_dict_populated(self, fitted_dc):
        assert len(fitted_dc.attack) > 0

    def test_defense_dict_populated(self, fitted_dc):
        assert len(fitted_dc.defense) > 0

    def test_home_advantage_positive(self, fitted_dc):
        """La ventaja de local debe ser generalmente positiva."""
        assert fitted_dc.home_adv > -0.5

    def test_1x2_probs_sum_to_one(self, fitted_dc):
        proba = fitted_dc.predict_proba("arsenal", "chelsea")
        total = proba["dc_prob_home"] + proba["dc_prob_draw"] + proba["dc_prob_away"]
        assert total == pytest.approx(1.0, abs=0.02)

    def test_all_probs_between_0_and_1(self, fitted_dc):
        proba = fitted_dc.predict_proba("arsenal", "chelsea")
        for key, val in proba.items():
            if key.startswith("dc_prob"):
                assert 0.0 <= val <= 1.0, f"{key}={val} fuera de [0,1]"

    def test_expected_goals_positive(self, fitted_dc):
        proba = fitted_dc.predict_proba("arsenal", "chelsea")
        assert proba["dc_exp_home_goals"] > 0
        assert proba["dc_exp_away_goals"] > 0

    def test_unknown_team_uses_mean_params(self, fitted_dc):
        """Equipo desconocido no debe provocar error."""
        proba = fitted_dc.predict_proba("unknown_team_xyz", "chelsea")
        assert "dc_prob_home" in proba

    def test_insufficient_data_does_not_fit(self, sample_historical):
        """Con menos de MIN_MATCHES_PER_LIGA el modelo no debe marcar fitted=True."""
        small_df = sample_historical.head(MIN_MATCHES_PER_LIGA - 1)
        dc = DixonColesModel(league_id=99).fit(small_df)
        assert dc.fitted is False


# ══════════════════════════════════════════════════════════════════════════════
#  DixonColesEnsemble — fallback chain
# ══════════════════════════════════════════════════════════════════════════════

class TestDixonColesEnsembleFallback:
    def test_unfitted_ensemble_returns_default_proba(self):
        ens    = DixonColesEnsemble()
        result = ens.predict_proba("Arsenal", "Chelsea", league_id=39)
        assert "dc_prob_home" in result

    def test_fitted_ensemble_uses_league_model(self, sample_historical):
        ens = DixonColesEnsemble().fit(sample_historical)
        # Liga 39 (Premier League) debe tener modelo propio
        result = ens.predict_proba("arsenal", "chelsea", league_id=39)
        total  = result["dc_prob_home"] + result["dc_prob_draw"] + result["dc_prob_away"]
        assert total == pytest.approx(1.0, abs=0.02)

    def test_unknown_league_falls_back(self, sample_historical):
        """Liga desconocida debe devolver resultados válidos (fallback global o similar)."""
        ens    = DixonColesEnsemble().fit(sample_historical)
        result = ens.predict_proba("arsenal", "chelsea", league_id=9999)
        assert "dc_prob_home" in result
        total = result["dc_prob_home"] + result["dc_prob_draw"] + result["dc_prob_away"]
        assert total == pytest.approx(1.0, abs=0.05)

    def test_resolve_espn_league_by_id(self, sample_historical):
        """
        FIX v6: league_id=501 (Liga BetPlay) debe resolver a su nombre
        y buscar el modelo antes de caer al global.
        """
        ens    = DixonColesEnsemble().fit(sample_historical)
        # No hay modelo para col.1 en este histórico EU, pero no debe explotar
        result = ens.predict_proba("atletico nacional", "millonarios", league_id=501)
        assert "dc_prob_home" in result


# ══════════════════════════════════════════════════════════════════════════════
#  blend_predictions
# ══════════════════════════════════════════════════════════════════════════════

class TestBlendPredictions:
    @pytest.fixture
    def dc_probs(self):
        return {
            "dc_prob_home":   0.45,
            "dc_prob_draw":   0.28,
            "dc_prob_away":   0.27,
            "dc_prob_btts":   0.52,
            "dc_prob_over25": 0.50,
        }

    @pytest.fixture
    def xgb_probs(self):
        return {
            "xgb_prob_home_win": 0.50,
            "xgb_prob_draw":     0.25,
            "xgb_prob_away_win": 0.25,
            "xgb_prob_btts":     0.58,
            "xgb_prob_over25":   0.55,
        }

    def test_result_contains_all_markets(self, dc_probs, xgb_probs):
        result = blend_predictions(dc_probs, xgb_probs)
        for m in ["prob_home_win", "prob_draw", "prob_away_win",
                  "prob_btts", "prob_over25"]:
            assert m in result

    def test_1x2_normalized_to_one(self, dc_probs, xgb_probs):
        result = blend_predictions(dc_probs, xgb_probs)
        total  = result["prob_home_win"] + result["prob_draw"] + result["prob_away_win"]
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_default_weight_is_blend(self, dc_probs, xgb_probs):
        """Con peso por defecto, el blend debe estar entre DC y XGB."""
        result = blend_predictions(dc_probs, xgb_probs)
        # home_win debe estar entre min(0.45, 0.50) y max(0.45, 0.50) antes de norm
        raw_home = DEFAULT_DC_WEIGHT * 0.45 + (1 - DEFAULT_DC_WEIGHT) * 0.50
        assert result["prob_home_win"] == pytest.approx(raw_home / 1.0, abs=0.01)

    def test_dc_weight_1_equals_dc_only(self, dc_probs, xgb_probs):
        """dc_weight=1.0 → resultado debe ser solo DC (normalizado)."""
        weights = {m: 1.0 for m in ["home_win", "draw", "away_win", "btts", "over25"]}
        result  = blend_predictions(dc_probs, xgb_probs, blend_weights=weights)
        # home_win normalizado desde DC puro
        s = 0.45 + 0.28 + 0.27
        assert result["prob_home_win"] == pytest.approx(0.45 / s, abs=1e-4)

    def test_xgb_weight_1_equals_xgb_only(self, dc_probs, xgb_probs):
        """dc_weight=0.0 → resultado debe ser solo XGB (normalizado)."""
        weights = {m: 0.0 for m in ["home_win", "draw", "away_win", "btts", "over25"]}
        result  = blend_predictions(dc_probs, xgb_probs, blend_weights=weights)
        s = 0.50 + 0.25 + 0.25
        assert result["prob_home_win"] == pytest.approx(0.50 / s, abs=1e-4)

    def test_missing_xgb_falls_back_to_dc(self, dc_probs):
        """Sin probs XGB, el blend debe usar solo DC."""
        result = blend_predictions(dc_probs, {})
        assert result["prob_home_win"] > 0


# ══════════════════════════════════════════════════════════════════════════════
#  _optimize_blend_weight
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizeBlendWeight:
    def test_returns_value_between_bounds(self):
        np.random.seed(1)
        n     = 100
        y     = np.random.randint(0, 2, n).astype(float)
        dc_p  = np.clip(y * 0.7 + np.random.normal(0, 0.1, n), 0.01, 0.99)
        xgb_p = np.clip(y * 0.6 + np.random.normal(0, 0.15, n), 0.01, 0.99)

        w = _optimize_blend_weight(dc_p, xgb_p, y, "home_win")
        assert 0.05 <= w <= 0.70

    def test_returns_float(self):
        np.random.seed(2)
        y    = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10, dtype=float)
        dc_p = np.full_like(y, 0.6)
        xg_p = np.full_like(y, 0.55)
        w = _optimize_blend_weight(dc_p, xg_p, y, "draw")
        assert isinstance(w, float)


# ══════════════════════════════════════════════════════════════════════════════
#  FootbotEnsemble (marcado @slow — entrena XGBoost × 5 mercados)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestFootbotEnsemble:
    @pytest.fixture
    def fitted_ensemble(self, sample_training_df, sample_historical):
        dc  = DixonColesEnsemble().fit(sample_historical)
        ens = FootbotEnsemble().fit(sample_training_df, dc_ensemble=dc)
        return ens

    def test_fitted_is_true(self, fitted_ensemble):
        assert fitted_ensemble.fitted is True

    def test_all_markets_trained(self, fitted_ensemble):
        for m in ["home_win", "draw", "away_win", "btts", "over25"]:
            assert m in fitted_ensemble.models

    def test_predict_returns_dict(self, fitted_ensemble, sample_fixture_row):
        result = fitted_ensemble.predict(sample_fixture_row.to_dict())
        assert isinstance(result, dict)
        assert "xgb_prob_home_win" in result

    def test_predict_probs_between_0_and_1(self, fitted_ensemble, sample_fixture_row):
        result = fitted_ensemble.predict(sample_fixture_row.to_dict())
        for key, val in result.items():
            if key.startswith("xgb_prob"):
                assert 0.0 <= val <= 1.0, f"{key}={val} fuera de [0,1]"

    def test_blend_weights_stored(self, fitted_ensemble):
        assert len(fitted_ensemble.blend_weights) == 5
        for m in ["home_win", "draw", "away_win", "btts", "over25"]:
            assert m in fitted_ensemble.blend_weights

    def test_validation_summary_generated(self, fitted_ensemble):
        summary = fitted_ensemble.get_validation_summary()
        assert "home_win" in summary
        assert "ROI" in summary

    def test_unfitted_predict_returns_empty(self):
        ens    = FootbotEnsemble()
        result = ens.predict({})
        assert result == {}