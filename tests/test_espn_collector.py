"""
test_espn_collector.py
Tests para espn_collector.py — énfasis en los FIX documentados:

  FIX A1 — _parse_score: usa displayValue como fuente primaria
            (value puede tener coma decimal: "3,0" → falla sin el fix)
  FIX A2 — get_match_odds: itera providers hasta encontrar moneyLine válido
  FIX A3 — _to_decimal: normaliza coma decimal antes de float()
  FIX B1 — _safe_int: maneja "27,0" sin explotar
  FIX B2 — get_standings: usa _safe_int en campos numéricos
"""

import pytest
from unittest.mock import MagicMock, patch


# ─── Imports del módulo ───────────────────────────────────────────────────────

from src.espn_collector import (
    _parse_score,
    _to_decimal,
    _safe_int,
    _norm_status,
    _parse_fixture,
    ESPNClient,
    FINISHED_STATUSES,
    LIVE_STATUSES,
    SCHEDULED_STATUSES,
)


# ══════════════════════════════════════════════════════════════════════════════
#  _parse_score  (FIX A1)
# ══════════════════════════════════════════════════════════════════════════════

class TestParseScore:
    """
    FIX A1: _parse_score debe usar displayValue como fuente primaria
    cuando score es un dict (endpoint /schedule).

    Antes del fix, se usaba 'value' directamente; en entornos ES/COL
    'value' viene como string "3,0" que hace fallar float().
    """

    def test_none_returns_none(self):
        assert _parse_score(None) is None

    def test_string_integer(self):
        """Endpoint /scoreboard devuelve string plano."""
        assert _parse_score("3") == 3

    def test_string_zero(self):
        assert _parse_score("0") == 0

    def test_string_invalid_returns_none(self):
        assert _parse_score("invalid") is None

    # ── FIX A1: dict con displayValue ────────────────────────────────────────

    def test_dict_display_value_preferred(self):
        """displayValue="3" debe usarse aunque value tenga coma decimal."""
        score = {"displayValue": "3", "value": "3,0", "$ref": "http://..."}
        assert _parse_score(score) == 3

    def test_dict_display_value_zero(self):
        score = {"displayValue": "0", "value": "0,0"}
        assert _parse_score(score) == 0

    def test_dict_fallback_to_value_with_comma(self):
        """Sin displayValue, 'value' con coma decimal debe convertirse igual."""
        score = {"value": "3,0"}
        assert _parse_score(score) == 3

    def test_dict_fallback_to_value_normal(self):
        score = {"value": 2.0}
        assert _parse_score(score) == 2

    def test_dict_both_none_returns_none(self):
        assert _parse_score({"value": None, "displayValue": None}) is None

    def test_dict_empty_returns_none(self):
        assert _parse_score({}) is None


# ══════════════════════════════════════════════════════════════════════════════
#  _to_decimal  (FIX A3)
# ══════════════════════════════════════════════════════════════════════════════

class TestToDecimal:
    """
    FIX A3: _to_decimal normaliza coma decimal antes de float().
    DraftKings en col.1 devuelve moneylines como "270,0" en vez de "270".
    """

    def test_none_returns_none(self):
        assert _to_decimal(None) is None

    def test_positive_american_to_decimal(self):
        """+ 200 → 3.00"""
        assert _to_decimal(200) == pytest.approx(3.00, abs=0.01)

    def test_negative_american_to_decimal(self):
        """-150 → 1.67"""
        assert _to_decimal(-150) == pytest.approx(1.667, abs=0.01)

    def test_already_decimal(self):
        """Valores entre 1 y 100 se asumen ya en decimal."""
        assert _to_decimal(1.85) == pytest.approx(1.85, abs=0.01)

    # ── FIX A3: coma decimal ─────────────────────────────────────────────────

    def test_comma_decimal_positive_american(self):
        """'270,0' debe tratarse como +270 americano → 3.70."""
        result = _to_decimal("270,0")
        assert result == pytest.approx(3.70, abs=0.01)

    def test_comma_decimal_negative_american(self):
        """'-150,0' debe tratarse como -150 → 1.67."""
        result = _to_decimal("-150,0")
        assert result == pytest.approx(1.667, abs=0.01)

    def test_string_normal_positive(self):
        assert _to_decimal("200") == pytest.approx(3.00, abs=0.01)

    def test_invalid_string_returns_none(self):
        assert _to_decimal("abc") is None

    def test_small_value_returns_none(self):
        """Valores ≤ 1 no tienen sentido como cuota."""
        assert _to_decimal(0.5) is None


# ══════════════════════════════════════════════════════════════════════════════
#  _safe_int  (FIX B1)
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeInt:
    """
    FIX B1: ESPN standings devuelve stats como "27,0" en entornos ES/COL.
    int("27,0") explota — _safe_int lo resuelve.
    """

    def test_integer_passthrough(self):
        assert _safe_int(27) == 27

    def test_float_truncates(self):
        assert _safe_int(27.9) == 27

    def test_string_integer(self):
        assert _safe_int("27") == 27

    def test_string_with_comma_decimal(self):
        """FIX B1: '27,0' debe convertirse a 27 sin explotar."""
        assert _safe_int("27,0") == 27

    def test_string_with_dot_decimal(self):
        assert _safe_int("27.0") == 27

    def test_none_returns_default(self):
        assert _safe_int(None) == 0

    def test_none_custom_default(self):
        assert _safe_int(None, default=99) == 99

    def test_invalid_string_returns_default(self):
        assert _safe_int("invalid") == 0


# ══════════════════════════════════════════════════════════════════════════════
#  _norm_status
# ══════════════════════════════════════════════════════════════════════════════

class TestNormStatus:
    def test_scheduled(self):
        assert _norm_status("STATUS_SCHEDULED") == "NS"

    def test_full_time(self):
        assert _norm_status("STATUS_FULL_TIME") == "FT"

    def test_final(self):
        assert _norm_status("STATUS_FINAL") == "FT"

    def test_in_progress(self):
        assert _norm_status("STATUS_IN_PROGRESS") == "1H"

    def test_halftime(self):
        assert _norm_status("STATUS_HALFTIME") == "HT"

    def test_postponed(self):
        assert _norm_status("STATUS_POSTPONED") == "PST"

    def test_unknown_passthrough(self):
        """Statuses desconocidos se devuelven tal cual."""
        assert _norm_status("STATUS_UNKNOWN_XYZ") == "STATUS_UNKNOWN_XYZ"


# ══════════════════════════════════════════════════════════════════════════════
#  Status sets
# ══════════════════════════════════════════════════════════════════════════════

class TestStatusSets:
    def test_ft_is_finished(self):
        assert "FT" in FINISHED_STATUSES

    def test_aet_is_finished(self):
        assert "AET" in FINISHED_STATUSES

    def test_pen_is_finished(self):
        assert "PEN" in FINISHED_STATUSES

    def test_1h_is_live(self):
        assert "1H" in LIVE_STATUSES

    def test_ns_is_scheduled(self):
        assert "NS" in SCHEDULED_STATUSES

    def test_no_overlap_finished_live(self):
        assert FINISHED_STATUSES.isdisjoint(LIVE_STATUSES)

    def test_no_overlap_finished_scheduled(self):
        assert FINISHED_STATUSES.isdisjoint(SCHEDULED_STATUSES)


# ══════════════════════════════════════════════════════════════════════════════
#  _parse_fixture
# ══════════════════════════════════════════════════════════════════════════════

class TestParseFixture:
    """Verifica que el schema de salida sea consistente con el resto del pipeline."""

    def _make_event(self, home_score="2", away_score="1",
                    status="STATUS_FULL_TIME"):
        return {
            "id": "999",
            "date": "2024-03-15T20:00:00Z",
            "season": {"year": 2024},
            "competitions": [{
                "status": {
                    "type": {"name": status},
                    "displayClock": "90'",
                },
                "notes": [{"headline": "Matchday 28"}],
                "venue": {
                    "fullName": "Estadio Test",
                    "address": {"city": "Madrid"},
                },
                "competitors": [
                    {
                        "homeAway": "home",
                        "score":    home_score,
                        "team": {"displayName": "Real Madrid", "id": "1"},
                    },
                    {
                        "homeAway": "away",
                        "score":    away_score,
                        "team": {"displayName": "Barcelona", "id": "2"},
                    },
                ],
            }],
        }

    def test_basic_keys_present(self):
        event    = self._make_event()
        result   = _parse_fixture(event, league_id=140, league_name="La Liga")
        required = [
            "fixture_id", "date", "status", "league_id", "league_name",
            "home_team", "away_team", "home_goals", "away_goals",
        ]
        for key in required:
            assert key in result, f"Clave '{key}' no encontrada en _parse_fixture"

    def test_scores_parsed_correctly(self):
        event  = self._make_event(home_score="3", away_score="0")
        result = _parse_fixture(event, 140, "La Liga")
        assert result["home_goals"] == 3
        assert result["away_goals"] == 0

    def test_status_normalized(self):
        event  = self._make_event(status="STATUS_FULL_TIME")
        result = _parse_fixture(event, 140, "La Liga")
        assert result["status"] == "FT"

    def test_missing_competitors_returns_none(self):
        event = {"id": "1", "date": "", "season": {}, "competitions": [{}]}
        assert _parse_fixture(event, 140, "La Liga") is None

    def test_league_id_preserved(self):
        event  = self._make_event()
        result = _parse_fixture(event, league_id=39, league_name="Premier League")
        assert result["league_id"] == 39
        assert result["league_name"] == "Premier League"


# ══════════════════════════════════════════════════════════════════════════════
#  ESPNClient
# ══════════════════════════════════════════════════════════════════════════════

class TestESPNClient:
    def test_instantiation_no_key_needed(self):
        """ESPNClient no debe requerir ningún parámetro de API key."""
        client = ESPNClient(delay=0)
        assert client is not None

    def test_calls_counter_starts_zero(self):
        client = ESPNClient(delay=0)
        assert client.calls == 0

    def test_remaining_is_none(self):
        """La propiedad .remaining existe y devuelve None (sin límite oficial)."""
        client = ESPNClient(delay=0)
        assert client.remaining is None

    def test_status_string(self):
        client = ESPNClient(delay=0)
        assert "ESPN" in client.status()

    @patch("src.espn_collector._SESSION")
    def test_get_increments_counter(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"events": []}
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        client = ESPNClient(delay=0)
        client.get("https://fake.url/test")
        assert client.calls == 1

    @patch("src.espn_collector._SESSION")
    def test_get_returns_none_on_http_error(self, mock_session):
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error
        mock_session.get.return_value = mock_resp

        client = ESPNClient(delay=0)
        result = client.get("https://fake.url/not-found")
        assert result is None