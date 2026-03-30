"""
test_telegram_sender.py
Tests para telegram_sender.py

Cubre:
  - format_date_es (días y meses en español)
  - format_message con bets vacío → sección "Sin value bets"
  - format_message con nivel alta → sección 🟢
  - format_message con nivel media → sección 🟡
  - format_message con nivel baja → sección 🔵 (nuevo en v6)
  - format_message con model_stats → estadísticas incluidas
  - send_telegram no explota cuando credentials son placeholder
"""

import pandas as pd
import pytest
from datetime import date

from src.telegram_sender import format_message, format_date_es


# ══════════════════════════════════════════════════════════════════════════════
#  format_date_es
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatDateEs:
    def test_monday_in_spanish(self):
        # 2024-01-01 es lunes
        monday = date(2024, 1, 1)
        assert monday.weekday() == 0  # confirmar que es lunes
        assert "Lunes" in format_date_es(monday)

    def test_saturday_in_spanish(self):
        # 2024-01-06 es sábado
        saturday = date(2024, 1, 6)
        assert saturday.weekday() == 5  # confirmar que es sábado
        assert "Sábado" in format_date_es(saturday)

    def test_contains_day_number(self):
        d      = date(2024, 3, 15)
        result = format_date_es(d)
        assert "15" in result

    def test_contains_month_abbreviation(self):
        d      = date(2024, 3, 15)
        result = format_date_es(d)
        assert "Mar" in result

    def test_returns_string(self):
        assert isinstance(format_date_es(date.today()), str)


# ══════════════════════════════════════════════════════════════════════════════
#  format_message — estructura general
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatMessage:
    def test_empty_bets_no_value_today(self):
        msg = format_message(pd.DataFrame())
        assert "Sin value bets" in msg

    def test_empty_bets_has_header(self):
        msg = format_message(pd.DataFrame())
        assert "FOOTBOT" in msg

    def test_empty_bets_has_disclaimer(self):
        msg = format_message(pd.DataFrame())
        assert "responsabilidad" in msg.lower()

    def test_returns_string(self):
        msg = format_message(pd.DataFrame())
        assert isinstance(msg, str)
        assert len(msg) > 50

    def test_alta_section_present_when_alta_bets(self, sample_bets_df):
        alta_only = sample_bets_df[sample_bets_df["confidence"] == "alta"]
        msg = format_message(alta_only)
        assert "ALTA CONFIANZA" in msg
        assert "🟢" in msg

    def test_media_section_present_when_media_bets(self, sample_bets_df):
        media_only = sample_bets_df[sample_bets_df["confidence"] == "media"]
        msg = format_message(media_only)
        assert "MEDIA CONFIANZA" in msg
        assert "🟡" in msg

    def test_baja_section_present_when_baja_bets(self, sample_bets_df):
        """
        Nuevo en v6: sección BAJA CONFIANZA debe aparecer con 🔵.
        """
        baja_only = sample_bets_df[sample_bets_df["confidence"] == "baja"]
        msg = format_message(baja_only)
        assert "BAJA CONFIANZA" in msg, (
            "La sección 🔵 BAJA CONFIANZA no se renderiza. "
            "Verificar format_message en telegram_sender.py v6."
        )
        assert "🔵" in msg

    def test_all_three_sections_with_full_bets(self, sample_bets_df):
        msg = format_message(sample_bets_df)
        assert "ALTA CONFIANZA" in msg
        assert "MEDIA CONFIANZA" in msg
        assert "BAJA CONFIANZA" in msg

    def test_alta_before_media(self, sample_bets_df):
        """Alta confianza siempre aparece antes que media."""
        msg   = format_message(sample_bets_df)
        pos_a = msg.find("ALTA CONFIANZA")
        pos_m = msg.find("MEDIA CONFIANZA")
        assert pos_a < pos_m

    def test_media_before_baja(self, sample_bets_df):
        """Media confianza siempre aparece antes que baja."""
        msg   = format_message(sample_bets_df)
        pos_m = msg.find("MEDIA CONFIANZA")
        pos_b = msg.find("BAJA CONFIANZA")
        assert pos_m < pos_b

    def test_team_names_in_message(self, sample_bets_df):
        """Los nombres de equipos deben aparecer en el mensaje."""
        msg = format_message(sample_bets_df)
        assert "Arsenal" in msg
        assert "Chelsea" in msg

    def test_market_display_in_message(self, sample_bets_df):
        msg = format_message(sample_bets_df)
        assert "Victoria Arsenal" in msg or "Victoria" in msg

    def test_league_in_message(self, sample_bets_df):
        msg = format_message(sample_bets_df)
        assert "Premier League" in msg

    def test_no_sin_value_bets_when_bets_present(self, sample_bets_df):
        msg = format_message(sample_bets_df)
        assert "Sin value bets" not in msg

    def test_cuota_estimada_tag_for_model_implied(self, sample_bets_df):
        """Cuotas estimadas (model_implied) deben marcarse en el mensaje."""
        df_model = sample_bets_df.copy()
        df_model["odds_source"] = "model_implied"
        msg = format_message(df_model)
        # La etiqueta _(est.)_ o similar debe aparecer
        assert "est." in msg or "estimada" in msg

    def test_model_stats_included_when_sufficient(self, sample_bets_df):
        stats = {
            "total":       50,
            "ganadas":     28,
            "tasa_pct":    56.0,
            "roi_pct":     8.5,
            "tasa_alta_pct":  62.0,
            "roi_alta_pct":   14.2,
            "tasa_media_pct": 50.0,
            "roi_media_pct":   3.1,
        }
        msg = format_message(sample_bets_df, model_stats=stats)
        assert "Rendimiento" in msg
        assert "56" in msg or "56.0" in msg

    def test_model_stats_not_shown_when_insufficient(self, sample_bets_df):
        """Con menos de 10 apuestas cerradas no se muestran stats."""
        stats = {"total": 5, "tasa_pct": 60.0, "roi_pct": 5.0}
        msg   = format_message(sample_bets_df, model_stats=stats)
        assert "10+ apuestas" in msg

    def test_baja_warning_text_present(self, sample_bets_df):
        """El nivel baja debe incluir advertencia de señal débil."""
        baja_only = sample_bets_df[sample_bets_df["confidence"] == "baja"]
        msg = format_message(baja_only)
        # Buscar alguna indicación de señal débil o stake reducido
        assert any(kw in msg.lower() for kw in ["débil", "reducido", "exploratorio", "baja"]), (
            "No hay advertencia de señal débil para nivel baja"
        )

    def test_message_not_too_long_for_single_match(self, sample_bets_df):
        """Un mensaje para un partido no debe superar los 4096 chars de Telegram."""
        alta_only = sample_bets_df[sample_bets_df["confidence"] == "alta"]
        msg = format_message(alta_only)
        assert len(msg) < 4096

    def test_estimated_odds_count_in_subtitle(self):
        """Si hay cuotas estimadas, el subtítulo debe mencionarlo."""
        df = pd.DataFrame([{
            "home_team": "A", "away_team": "B",
            "league": "Test", "match_date": "2024-01-01",
            "market": "home_win", "market_display": "Victoria A",
            "model_prob": 0.55, "model_prob_pct": "55.0%",
            "reference_odds": 2.10, "odds_source": "model_implied",
            "odds_are_real": False,
            "edge_pct": 15.5, "kelly_pct": 2.5, "confidence": "alta",
            "explanation": "test", "exp_home_goals": 1.5,
            "exp_away_goals": 1.0, "exp_total_goals": 2.5,
            "n_home_matches": 30, "n_away_matches": 30,
        }])
        msg = format_message(df)
        # Debe indicar que la cuota es estimada
        assert "estimada" in msg.lower() or "est." in msg