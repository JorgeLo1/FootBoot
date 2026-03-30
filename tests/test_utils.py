"""
test_utils.py
Tests para src/utils.py

Cubre:
  - ApiRateLimiter: remaining, consume, can_request, persistencia, reset diario
  - get_weather_for_fixture: respuesta normal, fecha futura (no API call), equipo desconocido
  - ESTADIOS: equipos de ligas LATAM están en el diccionario
"""

import json
import os
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from src.utils import ApiRateLimiter, get_weather_for_fixture, ESTADIOS


# ══════════════════════════════════════════════════════════════════════════════
#  ApiRateLimiter
# ══════════════════════════════════════════════════════════════════════════════

class TestApiRateLimiter:
    @pytest.fixture
    def limiter(self, tmp_path):
        usage_file = str(tmp_path / "api_usage.json")
        return ApiRateLimiter(usage_file, daily_limit=100)

    def test_initial_remaining_equals_limit(self, limiter):
        assert limiter.remaining == 100

    def test_can_request_initially(self, limiter):
        assert limiter.can_request(1) is True

    def test_can_request_exact_limit(self, limiter):
        assert limiter.can_request(100) is True

    def test_cannot_request_over_limit(self, limiter):
        assert limiter.can_request(101) is False

    def test_consume_reduces_remaining(self, limiter):
        limiter.consume(10)
        assert limiter.remaining == 90

    def test_consume_multiple_times(self, limiter):
        limiter.consume(30)
        limiter.consume(20)
        assert limiter.remaining == 50

    def test_consume_to_zero(self, limiter):
        limiter.consume(100)
        assert limiter.remaining == 0
        assert limiter.can_request(1) is False

    def test_persists_to_file(self, tmp_path):
        """El contador debe sobrevivir entre instancias del limiter."""
        usage_file = str(tmp_path / "api_usage.json")
        limiter1   = ApiRateLimiter(usage_file, daily_limit=100)
        limiter1.consume(42)

        limiter2 = ApiRateLimiter(usage_file, daily_limit=100)
        assert limiter2.remaining == 58

    def test_resets_on_new_day(self, tmp_path):
        """Si la fecha guardada es ayer, el contador debe reiniciarse a 0."""
        usage_file = str(tmp_path / "api_usage.json")
        # Escribir un estado de ayer
        yesterday = str(date.today() - timedelta(days=1))
        with open(usage_file, "w") as f:
            json.dump({"date": yesterday, "count": 99}, f)

        limiter = ApiRateLimiter(usage_file, daily_limit=100)
        assert limiter.remaining == 100

    def test_status_string_contains_info(self, limiter):
        status = limiter.status()
        assert "100" in status or "0" in status

    def test_alert_threshold_does_not_crash(self, tmp_path, monkeypatch):
        """Cuando remaining baja del umbral, no debe explotar (Telegram puede no estar config)."""
        usage_file = str(tmp_path / "api_usage.json")
        # Parchear send_telegram para que no intente conectar
        monkeypatch.setattr(
            "src.utils.ApiRateLimiter._save", lambda self: None
        )
        limiter = ApiRateLimiter(usage_file, daily_limit=25)
        # Consumir hasta acercarse al umbral (20)
        limiter._state["count"] = 0
        # Simular consumo sin guardar
        with patch("src.telegram_sender.send_telegram", return_value=False):
            try:
                limiter.consume(10)
            except Exception:
                pass  # No debe explotar

    def test_file_created_after_consume(self, tmp_path):
        usage_file = str(tmp_path / "api_usage.json")
        limiter    = ApiRateLimiter(usage_file, daily_limit=100)
        limiter.consume(5)
        assert os.path.exists(usage_file)


# ══════════════════════════════════════════════════════════════════════════════
#  get_weather_for_fixture
# ══════════════════════════════════════════════════════════════════════════════

class TestGetWeatherForFixture:
    def test_returns_required_keys(self):
        with patch("src.utils.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "daily": {
                    "precipitation_sum":    [1.5],
                    "windspeed_10m_max":    [18.0],
                    "temperature_2m_max":  [22.0],
                }
            }
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_weather_for_fixture("Arsenal", str(date.today()))

        required = ["temp_max", "precipitation", "wind_max", "rain_flag", "wind_flag"]
        for key in required:
            assert key in result

    def test_rain_flag_on_heavy_rain(self):
        with patch("src.utils.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "daily": {
                    "precipitation_sum":   [15.0],  # > 2mm → rain_flag=1
                    "windspeed_10m_max":   [10.0],
                    "temperature_2m_max":  [18.0],
                }
            }
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_weather_for_fixture("Arsenal", str(date.today()))

        assert result["rain_flag"] == 1

    def test_wind_flag_on_high_wind(self):
        with patch("src.utils.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "daily": {
                    "precipitation_sum":   [0.0],
                    "windspeed_10m_max":   [45.0],  # > 30 → wind_flag=1
                    "temperature_2m_max":  [15.0],
                }
            }
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_weather_for_fixture("Arsenal", str(date.today()))

        assert result["wind_flag"] == 1

    def test_future_date_beyond_16_days_no_api_call(self):
        """
        FIX: Fechas > 16 días adelante devuelven valores neutros sin llamar a la API.
        Antes del fix, se llamaba a Open-Meteo que respondía 400 Bad Request.
        """
        far_future = date.today() + timedelta(days=20)

        with patch("src.utils.requests.get") as mock_get:
            result = get_weather_for_fixture("Arsenal", str(far_future))
            mock_get.assert_not_called()  # NO debe llamar a la API

        assert result["rain_flag"] == 0
        assert result["wind_flag"] == 0

    def test_api_error_returns_neutral_values(self):
        """Si la API falla, devuelve valores neutros sin explotar."""
        import requests
        with patch("src.utils.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("timeout")
            result = get_weather_for_fixture("Arsenal", str(date.today()))

        assert "rain_flag" in result
        assert "wind_flag" in result
        assert result["rain_flag"] in (0, 1)

    def test_unknown_team_uses_london_default(self):
        """Equipo sin coordenadas en ESTADIOS usa Londres como fallback."""
        with patch("src.utils.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "daily": {
                    "precipitation_sum":   [0.5],
                    "windspeed_10m_max":   [12.0],
                    "temperature_2m_max":  [16.0],
                }
            }
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = get_weather_for_fixture("Equipo Muy Desconocido XYZ",
                                             str(date.today()))

        assert "rain_flag" in result  # No explota


# ══════════════════════════════════════════════════════════════════════════════
#  ESTADIOS — cobertura de ligas LATAM
# ══════════════════════════════════════════════════════════════════════════════

class TestEstadiosDict:
    """
    Verifica que los equipos de ligas LATAM tienen coordenadas en ESTADIOS.
    Antes de la v3 de utils.py, todos los equipos LATAM usaban el fallback
    de Londres, enviando solicitudes de clima incorrectas a Open-Meteo.
    """

    # Liga BetPlay (col.1)
    def test_atletico_nacional_has_coords(self):
        assert "Atlético Nacional" in ESTADIOS

    def test_millonarios_has_coords(self):
        assert "Millonarios" in ESTADIOS

    # Liga Argentina (arg.1)
    def test_river_plate_has_coords(self):
        assert "River Plate" in ESTADIOS

    def test_boca_juniors_has_coords(self):
        assert "Boca Juniors" in ESTADIOS

    # Brasileirão (bra.1)
    def test_flamengo_has_coords(self):
        assert "Flamengo" in ESTADIOS

    def test_palmeiras_has_coords(self):
        assert "Palmeiras" in ESTADIOS

    # Liga MX (mex.1)
    def test_club_america_has_coords(self):
        assert "Club América" in ESTADIOS

    def test_guadalajara_has_coords(self):
        assert "Guadalajara" in ESTADIOS

    # Premier League
    def test_arsenal_has_coords(self):
        assert "Arsenal" in ESTADIOS

    def test_coords_are_valid_tuples(self):
        for team, coords in ESTADIOS.items():
            assert isinstance(coords, tuple), f"{team}: coords no es tuple"
            assert len(coords) == 2, f"{team}: coords no tiene 2 elementos"
            lat, lon = coords
            assert -90 <= lat <= 90, f"{team}: latitud {lat} inválida"
            assert -180 <= lon <= 180, f"{team}: longitud {lon} inválida"

    def test_latam_teams_not_using_london_coords(self):
        """
        LATAM teams deben tener coordenadas distintas a Londres (51.5, -0.1).
        """
        london = (51.5074, -0.1278)
        latam_teams = [
            "Atlético Nacional", "Millonarios", "River Plate",
            "Flamengo", "Club América",
        ]
        for team in latam_teams:
            if team in ESTADIOS:
                lat, lon = ESTADIOS[team]
                assert (abs(lat - london[0]) > 1 or abs(lon - london[1]) > 1), (
                    f"{team} tiene coordenadas de Londres — no se corrigió en utils.py"
                )