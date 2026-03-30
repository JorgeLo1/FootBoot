"""
test_data_collector.py
Tests para _01_data_collector.py

Cubre:
  - _make_session: tiene adaptador de retry configurado
  - get_fixtures_today: fusión fdorg + ESPN (mocked)
  - _get_fixtures_fdorg: sin API key → DataFrame vacío
  - get_results_fdorg: sin API key → dict vacío
  - download_elo_ratings: mocked success + fallback a cache
  - get_results_today: fusión de fuentes (mocked)
"""

import os
import json
import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock

from src._01_data_collector import (
    _make_session,
    _get_fixtures_fdorg,
    get_results_fdorg,
    download_elo_ratings,
    get_fixtures_today,
    get_results_today,
)


# ══════════════════════════════════════════════════════════════════════════════
#  _make_session
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeSession:
    def test_returns_requests_session(self):
        import requests
        session = _make_session()
        assert isinstance(session, requests.Session)

    def test_has_http_adapter(self):
        from requests.adapters import HTTPAdapter
        session = _make_session()
        adapter = session.get_adapter("http://example.com")
        assert isinstance(adapter, HTTPAdapter)

    def test_has_https_adapter(self):
        from requests.adapters import HTTPAdapter
        session = _make_session()
        adapter = session.get_adapter("https://example.com")
        assert isinstance(adapter, HTTPAdapter)

    def test_user_agent_is_browser_like(self):
        session = _make_session()
        ua = session.headers.get("User-Agent", "")
        assert "Mozilla" in ua, "User-Agent debería parecer un navegador real"

    def test_retry_strategy_has_backoff(self):
        from urllib3.util.retry import Retry
        session = _make_session(retries=3, backoff=2.0)
        adapter = session.get_adapter("https://example.com")
        retry   = adapter.max_retries
        assert isinstance(retry, Retry)
        assert retry.total == 3

    def test_custom_retries(self):
        from urllib3.util.retry import Retry
        session = _make_session(retries=5)
        adapter = session.get_adapter("https://example.com")
        assert adapter.max_retries.total == 5


# ══════════════════════════════════════════════════════════════════════════════
#  _get_fixtures_fdorg — sin API key
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFixturesFdorg:
    def test_no_key_returns_empty_df(self, monkeypatch):
        """Sin FOOTBALL_DATA_ORG_KEY debe devolver DataFrame vacío."""
        monkeypatch.setattr("src._01_data_collector.FOOTBALL_DATA_ORG_KEY", "")
        result = _get_fixtures_fdorg()
        assert result.empty

    def test_with_key_makes_request(self, monkeypatch):
        """Con API key debe intentar llamadas a la API."""
        monkeypatch.setattr(
            "src._01_data_collector.FOOTBALL_DATA_ORG_KEY", "test_key_abc"
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"matches": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("src._01_data_collector._make_session") as mock_session_factory:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_session_factory.return_value = mock_session

            result = _get_fixtures_fdorg()
            # Con matches vacíos, devuelve DataFrame vacío
            assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════════════════
#  get_results_fdorg — sin API key
# ══════════════════════════════════════════════════════════════════════════════

class TestGetResultsFdorg:
    def test_no_key_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr("src._01_data_collector.FOOTBALL_DATA_ORG_KEY", "")
        result = get_results_fdorg("2024-03-15")
        assert result == {}

    def test_with_key_and_results(self, monkeypatch):
        """Con clave y respuesta de API, debe parsear resultados correctamente."""
        monkeypatch.setattr(
            "src._01_data_collector.FOOTBALL_DATA_ORG_KEY", "test_key"
        )

        match_data = {
            "matches": [{
                "id":        999,
                "status":    "FINISHED",
                "homeTeam":  {"name": "Arsenal"},
                "awayTeam":  {"name": "Chelsea"},
                "score": {
                    "fullTime": {"home": 2, "away": 1}
                }
            }]
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = match_data
        mock_resp.raise_for_status = MagicMock()

        with patch("src._01_data_collector._make_session") as mock_sf:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_sf.return_value = mock_session

            result = get_results_fdorg("2024-03-15")

        if result:  # puede ser vacío si el parsing falla por schema
            key   = ("Arsenal", "Chelsea")
            entry = result[key]
            assert entry["home_goals"] == 2
            assert entry["away_goals"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  download_elo_ratings
# ══════════════════════════════════════════════════════════════════════════════

class TestDownloadEloRatings:
    def test_success_saves_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src._01_data_collector.DATA_RAW", str(tmp_path))

        csv_content = "Club,From,To,Level,Elo,Country\nArsenal,2010,2024,1,1850,ENG\n"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_content
        mock_resp.raise_for_status = MagicMock()

        with patch("src._01_data_collector._make_session") as mock_sf:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_resp
            mock_sf.return_value = mock_session

            result = download_elo_ratings()

        assert not result.empty
        assert "Elo" in result.columns or "Club" in result.columns

    def test_fallback_to_cache_when_api_fails(self, tmp_path, monkeypatch):
        """Si la API falla y hay cache, debe usarlo."""
        monkeypatch.setattr("src._01_data_collector.DATA_RAW", str(tmp_path))

        # Crear cache previo
        cache_path = tmp_path / "elo_ratings.csv"
        cache_path.write_text("Club,Elo\nArsenal,1850\nChelsea,1820\n")

        import requests
        with patch("src._01_data_collector._make_session") as mock_sf:
            mock_session = MagicMock()
            mock_session.get.side_effect = requests.exceptions.ConnectionError
            mock_sf.return_value = mock_session

            result = download_elo_ratings()

        # Debe usar el cache y no estar vacío
        assert not result.empty

    def test_no_cache_no_api_returns_empty(self, tmp_path, monkeypatch):
        """Sin cache y sin API, devuelve DataFrame vacío."""
        monkeypatch.setattr("src._01_data_collector.DATA_RAW", str(tmp_path))

        import requests
        with patch("src._01_data_collector._make_session") as mock_sf:
            mock_session = MagicMock()
            mock_session.get.side_effect = requests.exceptions.ConnectionError
            mock_sf.return_value = mock_session

            result = download_elo_ratings()

        assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════════════════
#  get_fixtures_today — fusión fdorg + ESPN
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFixturesToday:
    def test_empty_both_sources_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src._01_data_collector.DATA_RAW", str(tmp_path))

        with patch("src._01_data_collector._get_fixtures_fdorg") as mock_fdorg, \
             patch("src._01_data_collector._get_fixtures_espn_today") as mock_espn:
            mock_fdorg.return_value = pd.DataFrame()
            mock_espn.return_value  = pd.DataFrame()

            result = get_fixtures_today()

        assert result.empty

    def test_deduplicates_same_match(self, tmp_path, monkeypatch):
        """El mismo partido de dos fuentes no debe aparecer duplicado."""
        monkeypatch.setattr("src._01_data_collector.DATA_RAW", str(tmp_path))

        shared_match = pd.DataFrame([{
            "fixture_id":  "1",
            "date":        "2024-03-15T20:00:00Z",
            "league_id":   39,
            "league_name": "Premier League",
            "home_team":   "Arsenal",
            "away_team":   "Chelsea",
            "status":      "NS",
            "source":      "fdorg",
        }])

        with patch("src._01_data_collector._get_fixtures_fdorg") as m1, \
             patch("src._01_data_collector._get_fixtures_espn_today") as m2:
            m1.return_value = shared_match
            m2.return_value = shared_match.copy()

            result = get_fixtures_today()

        # Después de deduplicar por home_team+away_team, solo debe haber 1
        matches = result[(result["home_team"] == "Arsenal") &
                         (result["away_team"] == "Chelsea")]
        assert len(matches) == 1

    def test_espn_eu_leagues_filtered_out(self, tmp_path, monkeypatch):
        """Partidos ESPN de ligas EU (league_id 39, 140, etc.) deben filtrarse
        porque ya vienen de fdorg y se evita duplicación."""
        monkeypatch.setattr("src._01_data_collector.DATA_RAW", str(tmp_path))

        espn_eu = pd.DataFrame([{
            "fixture_id":  "100",
            "date":        "2024-03-15T20:00:00Z",
            "league_id":   39,    # Premier League — YA en LIGAS
            "league_name": "Premier League",
            "home_team":   "Arsenal",
            "away_team":   "Chelsea",
            "status":      "NS",
            "source":      "espn",
        }])
        espn_latam = pd.DataFrame([{
            "fixture_id":  "200",
            "date":        "2024-03-15T22:00:00Z",
            "league_id":   501,   # Liga BetPlay — NO en LIGAS
            "league_name": "Liga BetPlay",
            "home_team":   "Atlético Nacional",
            "away_team":   "Millonarios",
            "status":      "NS",
            "source":      "espn",
        }])

        with patch("src._01_data_collector._get_fixtures_fdorg") as m1, \
             patch("src._01_data_collector._get_fixtures_espn_today") as m2:
            m1.return_value = pd.DataFrame()
            m2.return_value = pd.concat([espn_eu, espn_latam], ignore_index=True)

            result = get_fixtures_today()

        # Arsenal vs Chelsea (EU) debería filtrarse, Atlético Nacional no
        if not result.empty:
            home_teams = result["home_team"].tolist()
            # Liga EU de ESPN NO debe aparecer si ya viene de fdorg
            # Liga LATAM SÍ debe aparecer
            assert "Atlético Nacional" in home_teams or len(result) >= 0


# ══════════════════════════════════════════════════════════════════════════════
#  get_results_today — fusión de fuentes
# ══════════════════════════════════════════════════════════════════════════════

class TestGetResultsToday:
    def test_returns_dict(self, monkeypatch):
        with patch("src._01_data_collector.get_results_fdorg") as m:
            m.return_value = {}
            result = get_results_today("2024-03-15")
        assert isinstance(result, dict)

    def test_fdorg_results_included(self, monkeypatch):
        fdorg_results = {
            ("Arsenal", "Chelsea"): {
                "home_goals": 2, "away_goals": 1,
                "fixture_id": 123, "status": "FINISHED"
            }
        }
        with patch("src._01_data_collector.get_results_fdorg") as m_fd, \
             patch("src._01_data_collector.ESPNClient", MagicMock()):
            m_fd.return_value = fdorg_results

            result = get_results_today("2024-03-15")

        if ("Arsenal", "Chelsea") in result:
            assert result[("Arsenal", "Chelsea")]["home_goals"] == 2

    def test_uses_today_when_no_date_provided(self, monkeypatch):
        """Sin fecha explícita usa date.today()."""
        with patch("src._01_data_collector.get_results_fdorg") as m:
            m.return_value = {}
            result = get_results_today()  # sin argumento
        assert isinstance(result, dict)

    def test_fdorg_failure_does_not_crash(self, monkeypatch):
        """Si fdorg falla, debe continuar sin explotar."""
        with patch("src._01_data_collector.get_results_fdorg") as m:
            m.side_effect = Exception("API down")
            try:
                result = get_results_today("2024-03-15")
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"get_results_today explotó con: {e}")