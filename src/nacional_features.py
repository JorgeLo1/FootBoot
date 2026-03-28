"""
nacional_features.py
Features para selecciones nacionales.

FIX v4: eliminado el import roto de 'nacional_collector' (módulo que no existe).
Las funciones que dependían de él (get_team_stats, get_h2h de la API externa)
se reemplazaron por versiones que usan solo datos históricos locales.
El client de ESPN se usa opcionalmente para enriquecer standings.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_RAW, DATA_PROCESSED, LAMBDA_DECAY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

VENTANA_FORMA_NACIONAL = 8

COMP_TIER = {
    "eliminatorias": 1,
    "conmebol":      1,
    "copa america":  1,
    "world cup":     1,
    "copa del mundo":1,
    "nations league":1,
    "friendlies":    2,
    "amistosos":     2,
    "friendly":      2,
}

TIER_1_LEAGUES = {361, 271, 1, 4, 5, 30, 600, 601}
TIER_2_LEAGUES = {10, 11, 29}


def _get_comp_tier(league_id: int, league_name: str) -> int:
    if league_id in TIER_1_LEAGUES:
        return 1
    name_lower = (league_name or "").lower()
    for keyword, tier in COMP_TIER.items():
        if keyword in name_lower:
            return tier
    return 2


def _exponential_weight(days_ago: float, lam: float = LAMBDA_DECAY) -> float:
    return np.exp(-lam * max(days_ago, 0))


# ─── RANKING FIFA ─────────────────────────────────────────────────────────────

def load_fifa_rankings() -> pd.DataFrame:
    path = os.path.join(DATA_RAW, "fifa_rankings.csv")
    if not os.path.exists(path):
        log.warning(
            "fifa_rankings.csv no encontrado. "
            "Usando ranking neutro (1000) para todos los equipos."
        )
        return pd.DataFrame()

    df = pd.read_csv(path)
    if not all(c in df.columns for c in ["team", "points"]):
        log.warning("fifa_rankings.csv debe tener columnas: team, points")
        return pd.DataFrame()

    df["team_lower"] = df["team"].str.lower().str.strip()
    log.info(f"FIFA rankings cargados: {len(df)} selecciones")
    return df


def get_fifa_diff(home_team: str, away_team: str,
                  fifa_df: pd.DataFrame) -> float:
    if fifa_df.empty:
        return 0.0
    h = home_team.lower().strip()
    a = away_team.lower().strip()
    home_row = fifa_df[fifa_df["team_lower"] == h]
    away_row = fifa_df[fifa_df["team_lower"] == a]
    hp = float(home_row["points"].values[0]) if not home_row.empty else 1000.0
    ap = float(away_row["points"].values[0]) if not away_row.empty else 1000.0
    return round(hp - ap, 1)


def get_fifa_rank_diff(home_team: str, away_team: str,
                       fifa_df: pd.DataFrame) -> float:
    if fifa_df.empty or "rank" not in fifa_df.columns:
        return 0.0
    h = home_team.lower().strip()
    a = away_team.lower().strip()
    home_row = fifa_df[fifa_df["team_lower"] == h]
    away_row = fifa_df[fifa_df["team_lower"] == a]
    hr = float(home_row["rank"].values[0]) if not home_row.empty else 50.0
    ar = float(away_row["rank"].values[0]) if not away_row.empty else 50.0
    return round(ar - hr, 1)


# ─── FORMA RECIENTE ──────────────────────────────────────────────────────────

def compute_national_team_stats(
    team: str,
    is_home: bool,
    historical: pd.DataFrame,
    reference_date: datetime,
    window: int = VENTANA_FORMA_NACIONAL,
    min_tier: int = 2,
) -> dict:
    p = "home" if is_home else "away"

    if historical.empty:
        return _empty_national_stats(is_home)

    team_lower = team.lower().strip()

    mask_home = historical["home_team"].str.lower().str.strip() == team_lower
    mask_away = historical["away_team"].str.lower().str.strip() == team_lower
    mask_date = historical["match_date"] < reference_date

    if "league_id" in historical.columns:
        tier_mask = historical.apply(
            lambda row: _get_comp_tier(
                row.get("league_id", 2),
                row.get("league_name", "")
            ) <= min_tier,
            axis=1,
        )
        df_all = historical[(mask_home | mask_away) & mask_date & tier_mask]
    else:
        df_all = historical[(mask_home | mask_away) & mask_date]

    n_total = len(df_all)
    df      = df_all.sort_values("match_date", ascending=False).head(window).copy()

    if df.empty:
        return _empty_national_stats(is_home)

    df["days_ago"] = (reference_date - df["match_date"]).dt.days
    df["weight"]   = df["days_ago"].apply(_exponential_weight)
    W = df["weight"].sum() or 1.0

    def goals_scored(row):
        return row["home_goals"] if row["home_team"].lower().strip() == team_lower \
            else row["away_goals"]

    def goals_conceded(row):
        return row["away_goals"] if row["home_team"].lower().strip() == team_lower \
            else row["home_goals"]

    def pts(row):
        gf = goals_scored(row)
        gc = goals_conceded(row)
        return 3 if gf > gc else (1 if gf == gc else 0)

    def was_home(row):
        return row["home_team"].lower().strip() == team_lower

    df["gf"]       = df.apply(goals_scored,   axis=1)
    df["gc"]       = df.apply(goals_conceded, axis=1)
    df["pts"]      = df.apply(pts,            axis=1)
    df["was_home"] = df.apply(was_home,       axis=1)

    avg_gf   = (df["gf"]  * df["weight"]).sum() / W
    avg_gc   = (df["gc"]  * df["weight"]).sum() / W
    forma    = (df["pts"] * df["weight"]).sum() / W

    df["btts"]   = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    df["over25"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)

    btts_rate   = (df["btts"]   * df["weight"]).sum() / W
    over25_rate = (df["over25"] * df["weight"]).sum() / W

    df_role = df[df["was_home"] == is_home]
    if not df_role.empty:
        W_role      = df_role["weight"].sum() or 1.0
        forma_role  = (df_role["pts"] * df_role["weight"]).sum() / W_role
        avg_gf_role = (df_role["gf"]  * df_role["weight"]).sum() / W_role
        avg_gc_role = (df_role["gc"]  * df_role["weight"]).sum() / W_role
    else:
        forma_role  = forma
        avg_gf_role = avg_gf
        avg_gc_role = avg_gc

    last_date = df["match_date"].max()
    days_rest = int((reference_date - last_date).days) if not pd.isna(last_date) else 30
    racha     = _compute_racha(df.head(5), team_lower)

    return {
        f"{p}_goals_scored":    round(avg_gf,       3),
        f"{p}_goals_conceded":  round(avg_gc,       3),
        f"{p}_forma":           round(forma,        3),
        f"{p}_forma_role":      round(forma_role,   3),
        f"{p}_gf_role":         round(avg_gf_role,  3),
        f"{p}_gc_role":         round(avg_gc_role,  3),
        f"{p}_btts_rate":       round(btts_rate,    3),
        f"{p}_over25_rate":     round(over25_rate,  3),
        f"{p}_days_rest":       days_rest,
        f"{p}_n_matches":       len(df),
        f"{p}_n_matches_total": n_total,
        f"{p}_racha":           racha,
    }


def _compute_racha(df: pd.DataFrame, team_lower: str) -> int:
    if df.empty:
        return 0
    racha       = 0
    last_result = None

    for _, row in df.iterrows():
        is_home_here = row["home_team"].lower().strip() == team_lower
        gf = row["home_goals"] if is_home_here else row["away_goals"]
        gc = row["away_goals"] if is_home_here else row["home_goals"]

        if gf > gc:    result = "W"
        elif gf == gc: result = "D"
        else:          result = "L"

        if last_result is None:
            last_result = result
            racha = 1 if result == "W" else (-1 if result == "L" else 0)
        elif result == last_result:
            if result == "W":   racha += 1
            elif result == "L": racha -= 1
        else:
            break

    return racha


def _empty_national_stats(is_home: bool) -> dict:
    p = "home" if is_home else "away"
    return {
        f"{p}_goals_scored":    1.2,
        f"{p}_goals_conceded":  1.2,
        f"{p}_forma":           1.2,
        f"{p}_forma_role":      1.2,
        f"{p}_gf_role":         1.2,
        f"{p}_gc_role":         1.2,
        f"{p}_btts_rate":       0.5,
        f"{p}_over25_rate":     0.5,
        f"{p}_days_rest":       30,
        f"{p}_n_matches":       0,
        f"{p}_n_matches_total": 0,
        f"{p}_racha":           0,
    }


# ─── H2H ─────────────────────────────────────────────────────────────────────

def compute_national_h2h(
    home_team: str,
    away_team: str,
    historical: pd.DataFrame,
    reference_date: datetime,
    last_n: int = 10,
    tier_1_only: bool = True,
) -> dict:
    default = {
        "h2h_home_wins":  0.40,
        "h2h_draws":      0.25,
        "h2h_away_wins":  0.35,
        "h2h_avg_goals":  2.3,
        "h2h_btts_rate":  0.50,
        "h2h_n":          0,
        "h2h_tier1_n":    0,
    }

    if historical.empty:
        return default

    h = home_team.lower().strip()
    a = away_team.lower().strip()

    mask = (
        (
            (historical["home_team"].str.lower().str.strip() == h) &
            (historical["away_team"].str.lower().str.strip() == a)
        ) | (
            (historical["home_team"].str.lower().str.strip() == a) &
            (historical["away_team"].str.lower().str.strip() == h)
        )
    )
    df = historical[mask & (historical["match_date"] < reference_date)]

    if tier_1_only and "league_id" in df.columns:
        df = df[df["league_id"].isin(TIER_1_LEAGUES)]

    df = df.sort_values("match_date", ascending=False).head(last_n)

    if df.empty:
        if tier_1_only:
            return compute_national_h2h(
                home_team, away_team, historical,
                reference_date, last_n, tier_1_only=False,
            )
        return default

    total     = len(df)
    h2h_tier1 = len(df[df["league_id"].isin(TIER_1_LEAGUES)]) \
        if "league_id" in df.columns else total

    hw = len(df[
        ((df["home_team"].str.lower().str.strip() == h) & (df["home_goals"] > df["away_goals"])) |
        ((df["away_team"].str.lower().str.strip() == h) & (df["away_goals"] > df["home_goals"]))
    ])
    aw = len(df[
        ((df["home_team"].str.lower().str.strip() == a) & (df["home_goals"] > df["away_goals"])) |
        ((df["away_team"].str.lower().str.strip() == a) & (df["away_goals"] > df["home_goals"]))
    ])
    draws = total - hw - aw

    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["btts"]        = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    return {
        "h2h_home_wins":  round(hw / total,              3),
        "h2h_draws":      round(draws / total,            3),
        "h2h_away_wins":  round(aw / total,              3),
        "h2h_avg_goals":  round(df["total_goals"].mean(), 3),
        "h2h_btts_rate":  round(df["btts"].mean(),        3),
        "h2h_n":          total,
        "h2h_tier1_n":    h2h_tier1,
    }


# ─── CONTEXTO DE COMPETICIÓN ─────────────────────────────────────────────────

def get_competition_context(fixture: dict, standings: dict) -> dict:
    league_id = fixture.get("league_id")
    ctx = {
        "competition_tier":  1 if league_id in TIER_1_LEAGUES else 2,
        "is_neutral_venue":  _is_neutral_venue(fixture),
        "round_str":         fixture.get("round", ""),
        "is_knockout":       _is_knockout_round(fixture.get("round", "")),
        "home_pts_standing": 0,
        "away_pts_standing": 0,
        "pts_diff_standing": 0,
        "home_pos_standing": 10,
        "away_pos_standing": 10,
        "pos_diff_standing": 0,
    }

    if not standings or league_id not in standings:
        return ctx

    rows      = standings[league_id]
    home_name = fixture.get("home_team", "").lower().strip()
    away_name = fixture.get("away_team", "").lower().strip()

    home_row = next((r for r in rows if r["team"].lower().strip() == home_name), None)
    away_row = next((r for r in rows if r["team"].lower().strip() == away_name), None)

    if home_row:
        ctx["home_pts_standing"] = home_row.get("points", 0)
        ctx["home_pos_standing"] = home_row.get("rank", 10)
    if away_row:
        ctx["away_pts_standing"] = away_row.get("points", 0)
        ctx["away_pos_standing"] = away_row.get("rank", 10)

    ctx["pts_diff_standing"] = ctx["home_pts_standing"] - ctx["away_pts_standing"]
    ctx["pos_diff_standing"] = ctx["away_pos_standing"] - ctx["home_pos_standing"]

    return ctx


def _is_neutral_venue(fixture: dict) -> int:
    league_id = fixture.get("league_id")
    if league_id in {271, 1}:
        return 1
    return 0


def _is_knockout_round(round_str: str) -> int:
    knockout_kw = [
        "final", "semi", "quarter", "round of",
        "knockout", "eliminacion", "cuartos", "semis",
    ]
    r = (round_str or "").lower()
    return int(any(kw in r for kw in knockout_kw))


# ─── BUILDER PRINCIPAL ───────────────────────────────────────────────────────

def build_nacional_features(
    fixtures: list[dict],
    historical: pd.DataFrame,
    standings: dict,
    client=None,
) -> pd.DataFrame:
    """
    Genera features para partidos de selecciones.

    FIX v4: eliminado el import de 'nacional_collector' (no existe).
    Los stats de API se obtienen directamente desde ESPN si se pasa client,
    pero sin depender de módulos externos inexistentes.
    """
    fifa_df = load_fifa_rankings()
    rows    = []

    for fixture in fixtures:
        home = fixture["home_team"]
        away = fixture["away_team"]
        log.info(f"Features nacionales: {home} vs {away}")

        # Normalizar fecha
        raw_date = fixture.get("date", str(datetime.now()))
        try:
            ref_date = pd.to_datetime(raw_date, utc=True).tz_localize(None)
            if not isinstance(ref_date, datetime):
                ref_date = datetime.now()
        except Exception:
            ref_date = datetime.now()

        home_stats = compute_national_team_stats(home, True,  historical, ref_date)
        away_stats = compute_national_team_stats(away, False, historical, ref_date)
        h2h        = compute_national_h2h(home, away, historical, ref_date)

        fifa_diff      = get_fifa_diff(home, away, fifa_df)
        fifa_rank_diff = get_fifa_rank_diff(home, away, fifa_df)

        ctx = get_competition_context(fixture, standings)

        goals_diff = (
            home_stats.get("home_goals_scored", 1.2) -
            away_stats.get("away_goals_scored", 1.2)
        )
        forma_diff = (
            home_stats.get("home_forma", 1.2) -
            away_stats.get("away_forma", 1.2)
        )
        xg_total = (
            home_stats.get("home_goals_scored", 1.2) +
            away_stats.get("away_goals_scored", 1.2)
        )

        rows.append({
            "fixture_id":     fixture.get("fixture_id",  0),
            "league_id":      fixture.get("league_id",   0),
            "league_name":    fixture.get("league_name", ""),
            "home_team":      home,
            "away_team":      away,
            "match_date":     fixture.get("date", str(date.today())),
            "round":          fixture.get("round", ""),
            "n_home_matches": home_stats.get("home_n_matches_total", 0),
            "n_away_matches": away_stats.get("away_n_matches_total", 0),
            "fifa_diff":      fifa_diff,
            "fifa_rank_diff": fifa_rank_diff,
            **home_stats,
            **away_stats,
            **h2h,
            **ctx,
            "goals_diff":     round(goals_diff, 3),
            "forma_diff":     round(forma_diff, 3),
            "xg_total_exp":   round(xg_total,   3),
            "rest_diff":      (
                away_stats.get("away_days_rest", 30) -
                home_stats.get("home_days_rest", 30)
            ),
            "fatiga_flag":    int(
                home_stats.get("home_days_rest", 30) < 4 or
                away_stats.get("away_days_rest", 30) < 4
            ),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        path = os.path.join(DATA_PROCESSED, f"nacional_features_{date.today()}.csv")
        df.to_csv(path, index=False)
        log.info(f"Features nacionales: {len(df)} partidos → {path}")

    return df


# ─── DATASET DE ENTRENAMIENTO ────────────────────────────────────────────────

def build_nacional_training_dataset(historical: pd.DataFrame) -> pd.DataFrame:
    log.info("Construyendo dataset walk-forward nacional...")

    fifa_df    = load_fifa_rankings()
    historical = historical.sort_values("match_date").reset_index(drop=True)
    start_idx  = max(100, len(historical) // 5)

    rows  = []
    total = len(historical) - start_idx

    for idx in range(start_idx, len(historical)):
        match    = historical.iloc[idx]
        ref_date = match["match_date"]
        past     = historical.iloc[:idx]

        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        if not home or not away:
            continue

        home_stats = compute_national_team_stats(home, True,  past, ref_date)
        away_stats = compute_national_team_stats(away, False, past, ref_date)
        h2h        = compute_national_h2h(home, away, past, ref_date)
        fifa_diff  = get_fifa_diff(home, away, fifa_df)

        hg = int(match.get("home_goals", 0) or 0)
        ag = int(match.get("away_goals", 0) or 0)

        rows.append({
            "home_team":         home,
            "away_team":         away,
            "match_date":        ref_date,
            "league_id":         match.get("league_id",   0),
            "league_name":       match.get("league_name", ""),
            "fifa_diff":         fifa_diff,
            "n_home_matches":    home_stats.get("home_n_matches_total", 0),
            "n_away_matches":    away_stats.get("away_n_matches_total", 0),
            **home_stats,
            **away_stats,
            **h2h,
            "goals_diff":        round(home_stats.get("home_goals_scored",1.2) -
                                       away_stats.get("away_goals_scored",1.2), 3),
            "forma_diff":        round(home_stats.get("home_forma",1.2) -
                                       away_stats.get("away_forma",1.2), 3),
            "xg_total_exp":      round(home_stats.get("home_goals_scored",1.2) +
                                       away_stats.get("away_goals_scored",1.2), 3),
            "target_home_win":   int(hg > ag),
            "target_draw":       int(hg == ag),
            "target_away_win":   int(hg < ag),
            "target_btts":       int(hg > 0 and ag > 0),
            "target_over25":     int(hg + ag > 2.5),
            "home_goals_actual": hg,
            "away_goals_actual": ag,
        })

        if (idx - start_idx) % 50 == 0:
            pct = (idx - start_idx) / total * 100
            log.info(f"  Training nacional: {pct:.0f}%")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_team_norm"] = df["home_team"].str.lower().str.strip()
        df["away_team_norm"] = df["away_team"].str.lower().str.strip()
        df["home_goals"]     = df["home_goals_actual"]
        df["away_goals"]     = df["away_goals_actual"]
        path = os.path.join(DATA_PROCESSED, "nacional_training_dataset.csv")
        df.to_csv(path, index=False)
        log.info(f"Dataset entrenamiento nacional: {len(df)} partidos → {path}")

    return df