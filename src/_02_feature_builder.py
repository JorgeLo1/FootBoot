"""
_02_feature_builder.py — v7
Construye todas las variables predictivas para cada partido.

CAMBIOS v4:
  1. load_historical_results fusiona Football-Data.co.uk (EU, con cuotas)
     con ESPN histórico (ligas sin cobertura EU, sin cuotas).
  2. normalize_team_name usa rapidfuzz para matching robusto.
  3. TeamNameResolver construye vocabulario canónico desde datos históricos
     de AMBAS fuentes.
  4. build_training_dataset excluye market_prob_* (data leakage fix).
  5. build_features_for_fixtures enriquece con cuotas ESPN en tiempo real.

CAMBIOS v5:
  6. build_training_dataset O(n²) → O(n): _precompute_rolling_cache.

CAMBIOS v6:
  7. load_elo() fusiona ClubElo + elo_espn.csv (ELO propio LATAM).

CAMBIOS v7 (nuevo):
  8. build_features_for_fixtures enriquece con lesiones ESPN si
     ESPN_INJURIES_ENABLED=true. Features añadidas:
       home_injured_count, home_injury_score, home_out_count
       away_injured_count, away_injury_score, away_out_count
  9. build_features_for_fixtures enriquece con ESPN BPI si
     ESPN_BPI_ENABLED=true. Features añadidas:
       espn_bpi_home_prob, espn_bpi_away_prob, espn_bpi_available
  10. build_training_dataset incluye columnas de lesiones/BPI como NaN
      (no disponibles en histórico) para mantener schema consistente.
  11. FEATURE_COLS actualizado para incluir injury_score y bpi cuando
      están disponibles (acceso condicional seguro).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_RAW, DATA_STATSBOMB, DATA_PROCESSED,
    VENTANA_FORMA, LAMBDA_DECAY, LIGAS,
    ESPN_INJURIES_ENABLED, ESPN_BPI_ENABLED,
    ESPN_STANDINGS_FEATURES,
)
from src.utils import get_weather_for_fixture  # noqa: F401

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── FUZZY MATCHING ───────────────────────────────────────────────────────────

try:
    from rapidfuzz import process as fuzz_process, fuzz
    _HAS_RAPIDFUZZ = True
    log.debug("rapidfuzz disponible — usando fuzzy matching.")
except ImportError:
    _HAS_RAPIDFUZZ = False
    log.warning(
        "rapidfuzz no instalado. Usando normalización básica. "
        "Instala con: pip install rapidfuzz"
    )

_STRIP_TOKENS = [
    "fc", "cf", "ac", "sc", "rc", "cd", "sd", "ud", "rcd", "real",
    "atletico", "athletic", "sporting", "deportivo",
    "united", "city", "town", "rovers", "wanderers", "hotspur",
    "saint", "st", "borussia", "bayer", "rb", "ss", "as",
]

_FUZZY_THRESHOLD = 82


def _clean_name(name: str) -> str:
    import re
    s = name.lower().strip()
    s = re.sub(r"['\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    if len(tokens) > 1:
        tokens = [t for t in tokens if t not in _STRIP_TOKENS]
    return " ".join(tokens) if tokens else s


class TeamNameResolver:
    """
    Resuelve nombres de equipos de distintas fuentes (fd.org, ESPN, StatsBomb)
    contra un vocabulario canónico construido desde los datos históricos.
    """

    def __init__(self):
        self._canonical: list[str] = []
        self._raw_to_canonical: dict[str, str] = {}
        self._built = False

    def build_from_historical(self, historical: pd.DataFrame):
        if historical.empty:
            log.warning("TeamNameResolver: histórico vacío, sin vocabulario.")
            return

        teams = set()
        for col in ["home_team", "away_team"]:
            if col in historical.columns:
                teams.update(historical[col].dropna().unique())

        self._canonical = sorted({_clean_name(t) for t in teams})
        self._raw_to_canonical = {}
        self._built = True
        log.info(f"TeamNameResolver: {len(self._canonical)} equipos en vocabulario.")

    def resolve(self, name: str) -> str:
        if not name:
            return ""

        if name in self._raw_to_canonical:
            return self._raw_to_canonical[name]

        cleaned = _clean_name(name)

        if cleaned in self._canonical:
            self._raw_to_canonical[name] = cleaned
            return cleaned

        if _HAS_RAPIDFUZZ and self._built and self._canonical:
            result = fuzz_process.extractOne(
                cleaned,
                self._canonical,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=_FUZZY_THRESHOLD,
            )
            if result:
                matched, score, _ = result
                log.debug(f"Fuzzy match: '{name}' → '{matched}' (score={score})")
                self._raw_to_canonical[name] = matched
                return matched

        log.debug(f"Sin match para '{name}', usando nombre limpiado: '{cleaned}'")
        self._raw_to_canonical[name] = cleaned
        return cleaned

    def resolve_series(self, series: pd.Series) -> pd.Series:
        return series.apply(self.resolve)


_resolver = TeamNameResolver()


def normalize_team_name(name: str) -> str:
    if _resolver._built:
        return _resolver.resolve(name)
    return _clean_name(name)


def init_resolver(historical: pd.DataFrame):
    _resolver.build_from_historical(historical)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def exponential_weight(days_ago: float, lam: float = LAMBDA_DECAY) -> float:
    return np.exp(-lam * max(days_ago, 0))


# ─── CARGA DE DATOS HISTÓRICOS ───────────────────────────────────────────────

def load_historical_results() -> pd.DataFrame:
    """
    Carga y fusiona datos históricos de todas las fuentes:
      1. Football-Data.co.uk (7 ligas EU) — incluye cuotas B365, Pinnacle
      2. ESPN API (ligas activas) — goles, sin cuotas históricas
    """
    frames_eu = []

    for _, (league_name, fd_code, _) in LIGAS.items():
        path = os.path.join(DATA_RAW, f"fd_{fd_code}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            df["fd_code"]     = fd_code
            df["league_name"] = league_name
            df["source"]      = "fd_uk"
            frames_eu.append(df)

    if not frames_eu:
        log.warning("Sin datos Football-Data.co.uk. Ejecuta primero _01_data_collector.")

    raw_eu = pd.DataFrame()
    if frames_eu:
        raw_eu = pd.concat(frames_eu, ignore_index=True)
        raw_eu = raw_eu.rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team",
            "FTHG":     "home_goals", "FTAG":    "away_goals",
            "FTR":      "result",     "Date":    "date_str",
        })

        def parse_date(d):
            for fmt in ("%d/%m/%y", "%d/%m/%Y"):
                try:
                    return datetime.strptime(str(d), fmt)
                except Exception:
                    continue
            return pd.NaT

        raw_eu["match_date"] = raw_eu["date_str"].apply(parse_date)
        raw_eu = raw_eu.dropna(subset=["match_date", "home_goals", "away_goals"])
        raw_eu["home_goals"] = pd.to_numeric(raw_eu["home_goals"], errors="coerce")
        raw_eu["away_goals"] = pd.to_numeric(raw_eu["away_goals"], errors="coerce")
        raw_eu = raw_eu.dropna(subset=["home_goals", "away_goals"])

        for col in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                    "B365>2.5", "B365<2.5"]:
            if col in raw_eu.columns:
                raw_eu[col] = pd.to_numeric(raw_eu[col], errors="coerce")

    raw_espn = pd.DataFrame()
    try:
        from src._01_data_collector import load_espn_historical
        raw_espn = load_espn_historical()

        if not raw_espn.empty:
            for col in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                        "B365>2.5", "B365<2.5"]:
                if col not in raw_espn.columns:
                    raw_espn[col] = np.nan
            if "date_str" not in raw_espn.columns:
                raw_espn["date_str"] = raw_espn["match_date"].astype(str)

            log.info(f"ESPN histórico disponible: {len(raw_espn)} partidos")
    except Exception as e:
        log.warning(f"No se pudo cargar ESPN histórico: {e}")

    if not raw_eu.empty and not raw_espn.empty:
        common_cols = ["home_team", "away_team", "home_goals", "away_goals",
                       "match_date", "league_name", "source",
                       "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                       "B365>2.5", "B365<2.5"]
        for col in common_cols:
            if col not in raw_eu.columns:
                raw_eu[col] = np.nan
            if col not in raw_espn.columns:
                raw_espn[col] = np.nan

        raw = pd.concat(
            [raw_eu[common_cols + [c for c in raw_eu.columns if c not in common_cols]],
             raw_espn[common_cols + [c for c in raw_espn.columns if c not in common_cols]]],
            ignore_index=True
        )
    elif not raw_eu.empty:
        raw = raw_eu
    elif not raw_espn.empty:
        raw = raw_espn
    else:
        log.error("Sin datos históricos de ninguna fuente.")
        return pd.DataFrame()

    raw["match_date"] = pd.to_datetime(raw["match_date"], errors="coerce")
    raw = raw.dropna(subset=["match_date", "home_goals", "away_goals"])
    raw = raw.sort_values("match_date").reset_index(drop=True)

    init_resolver(raw)

    raw["home_team_norm"] = raw["home_team"].apply(normalize_team_name)
    raw["away_team_norm"] = raw["away_team"].apply(normalize_team_name)

    log.info(
        f"Histórico total: {len(raw)} partidos "
        f"(EU: {len(raw_eu) if not raw_eu.empty else 0} | "
        f"ESPN: {len(raw_espn) if not raw_espn.empty else 0})"
    )
    return raw


def load_xg_data() -> pd.DataFrame:
    path = os.path.join(DATA_STATSBOMB, "shots_xg.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["team_norm"] = df["team"].apply(normalize_team_name)
    if "is_home_team" not in df.columns:
        df["is_home_team"] = df["team"] == df["home_team"]
    return df


def load_match_summary() -> pd.DataFrame:
    path = os.path.join(DATA_STATSBOMB, "match_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["home_team_norm"] = df["home_team"].apply(normalize_team_name)
    df["away_team_norm"] = df["away_team"].apply(normalize_team_name)
    return df


def load_elo() -> pd.DataFrame:
    """
    Carga ELO ratings fusionando ClubElo (EU) + elo_espn.csv (LATAM).
    Prioriza ESPN en equipos duplicados.
    """
    frames = []

    path_clubelo = os.path.join(DATA_RAW, "elo_ratings.csv")
    if os.path.exists(path_clubelo):
        df_clubelo = pd.read_csv(path_clubelo)
        if "Club" in df_clubelo.columns:
            df_clubelo["team_norm"] = df_clubelo["Club"].apply(normalize_team_name)
        df_clubelo["_source"] = "clubelo"
        frames.append(df_clubelo)
        log.info(f"ELO ClubElo cargado: {len(df_clubelo)} equipos")
    else:
        log.warning("elo_ratings.csv no encontrado — solo se usará elo_espn.csv")

    path_espn = os.path.join(DATA_RAW, "elo_espn.csv")
    if os.path.exists(path_espn):
        df_espn = pd.read_csv(path_espn)
        if "Club" in df_espn.columns:
            df_espn["team_norm"] = df_espn["Club"].apply(normalize_team_name)
        df_espn["_source"] = "espn"
        frames.append(df_espn)
        log.info(f"ELO ESPN cargado: {len(df_espn)} equipos")
    else:
        log.warning(
            "elo_espn.csv no encontrado. Ejecuta compute_elo_espn() en _01_data_collector.py."
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    source_order = {"clubelo": 0, "espn": 1}
    combined["_source_order"] = combined["_source"].map(source_order).fillna(0)
    combined = combined.sort_values("_source_order")
    combined = combined.drop_duplicates(subset="team_norm", keep="last")
    combined = combined.drop(columns=["_source", "_source_order"])

    log.info(f"ELO total (fusionado): {len(combined)} equipos únicos")
    return combined


# ─── xG ──────────────────────────────────────────────────────────────────────

def _compute_xg(team_norm: str, xg_data: pd.DataFrame,
                is_home: bool, mode: str) -> float:
    if xg_data.empty or "shot_statsbomb_xg" not in xg_data.columns:
        return 0.0

    team_shots = xg_data[xg_data["team_norm"] == team_norm]
    if team_shots.empty:
        return 0.0

    if mode == "attack":
        n = team_shots["match_id"].nunique()
        return float(team_shots["shot_statsbomb_xg"].sum() / max(n, 1))

    team_match_ids = team_shots["match_id"].unique()
    if "is_home_team" in xg_data.columns:
        rival_is_home = not is_home
        rival_shots = xg_data[
            xg_data["match_id"].isin(team_match_ids) &
            (xg_data["team_norm"] != team_norm) &
            (xg_data["is_home_team"] == rival_is_home)
        ]
    else:
        rival_shots = xg_data[
            xg_data["match_id"].isin(team_match_ids) &
            (xg_data["team_norm"] != team_norm)
        ]

    if rival_shots.empty:
        return 0.0

    n = rival_shots["match_id"].nunique()
    return float(rival_shots["shot_statsbomb_xg"].sum() / max(n, 1))


# ─── ESTADÍSTICAS POR EQUIPO ─────────────────────────────────────────────────

def _empty_team_stats(is_home: bool) -> dict:
    p = "home" if is_home else "away"
    return {
        f"{p}_xg_scored":       1.2,
        f"{p}_xg_conceded":     1.2,
        f"{p}_goals_scored":    1.2,
        f"{p}_goals_conceded":  1.2,
        f"{p}_forma":           1.2,
        f"{p}_btts_rate":       0.5,
        f"{p}_over25_rate":     0.5,
        f"{p}_corners_avg":     5.0,
        f"{p}_fouls_avg":       11.0,
        f"{p}_days_rest":       7,
        f"{p}_n_matches":       0,
        f"{p}_n_matches_total": 0,
    }


def _compute_corners_fouls(team_norm: str, match_summary: pd.DataFrame,
                            is_home: bool) -> tuple[float, float]:
    if match_summary.empty:
        return 5.0, 11.0
    col_team = "home_team_norm" if is_home else "away_team_norm"
    c_col    = "corners_home"   if is_home else "corners_away"
    f_col    = "fouls_home"     if is_home else "fouls_away"
    df = match_summary[match_summary[col_team] == team_norm]
    if df.empty:
        return 5.0, 11.0
    c = df[c_col].mean() if c_col in df.columns else 5.0
    f = df[f_col].mean() if f_col in df.columns else 11.0
    return float(c), float(f)


def compute_team_stats(team: str, is_home: bool,
                       historical: pd.DataFrame,
                       xg_data: pd.DataFrame,
                       match_summary: pd.DataFrame,
                       reference_date: datetime,
                       window: int = VENTANA_FORMA) -> dict:
    team_norm  = normalize_team_name(team)
    p          = "home" if is_home else "away"
    col_team   = "home_team_norm" if is_home else "away_team_norm"
    col_scored = "home_goals"     if is_home else "away_goals"
    col_recv   = "away_goals"     if is_home else "home_goals"

    df_all = historical[
        (historical[col_team] == team_norm) &
        (historical["match_date"] < reference_date)
    ]
    n_total = len(df_all)

    df = df_all.sort_values("match_date", ascending=False).head(window).copy()

    if df.empty:
        return _empty_team_stats(is_home)

    df["days_ago"] = (reference_date - df["match_date"]).dt.days
    df["weight"]   = df["days_ago"].apply(exponential_weight)
    W = df["weight"].sum() or 1.0

    goals_scored   = (df[col_scored] * df["weight"]).sum() / W
    goals_conceded = (df[col_recv]   * df["weight"]).sum() / W

    xg_scored   = _compute_xg(team_norm, xg_data, is_home, "attack")
    xg_conceded = _compute_xg(team_norm, xg_data, is_home, "defense")
    if xg_scored   == 0.0: xg_scored   = goals_scored   * 1.05
    if xg_conceded == 0.0: xg_conceded = goals_conceded * 1.05

    def pts(row):
        s, c = row[col_scored], row[col_recv]
        return 3 if s > c else (1 if s == c else 0)

    df["pts"]    = df.apply(pts, axis=1)
    forma        = (df["pts"] * df["weight"]).sum() / W

    df["btts"]   = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    btts_rate    = (df["btts"]  * df["weight"]).sum() / W

    df["over25"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
    over25_rate  = (df["over25"] * df["weight"]).sum() / W

    corners_avg, fouls_avg = _compute_corners_fouls(team_norm, match_summary, is_home)

    last_date = df["match_date"].max()
    days_rest = int((reference_date - last_date).days) if not pd.isna(last_date) else 7

    return {
        f"{p}_xg_scored":       round(xg_scored,    3),
        f"{p}_xg_conceded":     round(xg_conceded,  3),
        f"{p}_goals_scored":    round(goals_scored,  3),
        f"{p}_goals_conceded":  round(goals_conceded,3),
        f"{p}_forma":           round(forma,         3),
        f"{p}_btts_rate":       round(btts_rate,     3),
        f"{p}_over25_rate":     round(over25_rate,   3),
        f"{p}_corners_avg":     round(corners_avg,   2),
        f"{p}_fouls_avg":       round(fouls_avg,     2),
        f"{p}_days_rest":       days_rest,
        f"{p}_n_matches":       len(df),
        f"{p}_n_matches_total": n_total,
    }


def compute_h2h(home_team: str, away_team: str,
                historical: pd.DataFrame, ref_date: datetime,
                last_n: int = 10) -> dict:
    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)

    mask = (
        ((historical["home_team_norm"] == h) & (historical["away_team_norm"] == a)) |
        ((historical["home_team_norm"] == a) & (historical["away_team_norm"] == h))
    )
    df = historical[mask & (historical["match_date"] < ref_date)]
    df = df.sort_values("match_date", ascending=False).head(last_n)

    if df.empty:
        return {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
                "h2h_avg_goals": 2.5,  "h2h_btts_rate": 0.5, "h2h_n": 0}

    total = len(df)
    hw = len(df[
        ((df["home_team_norm"] == h) & (df["home_goals"] > df["away_goals"])) |
        ((df["away_team_norm"] == h) & (df["away_goals"] > df["home_goals"]))
    ])
    aw = len(df[
        ((df["home_team_norm"] == a) & (df["home_goals"] > df["away_goals"])) |
        ((df["away_team_norm"] == a) & (df["away_goals"] > df["home_goals"]))
    ])
    draws = total - hw - aw

    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["btts"]        = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    return {
        "h2h_home_wins":  round(hw / total,               3),
        "h2h_draws":      round(draws / total,             3),
        "h2h_away_wins":  round(aw / total,               3),
        "h2h_avg_goals":  round(df["total_goals"].mean(),  3),
        "h2h_btts_rate":  round(df["btts"].mean(),         3),
        "h2h_n":          total,
    }


def get_elo_diff(home_team: str, away_team: str, elo_df: pd.DataFrame) -> float:
    if elo_df.empty or "team_norm" not in elo_df.columns:
        return 0.0
    elo_col = next((c for c in ["Elo", "elo"] if c in elo_df.columns), None)
    if not elo_col:
        return 0.0
    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)
    home_row = elo_df[elo_df["team_norm"] == h]
    away_row = elo_df[elo_df["team_norm"] == a]
    he = float(home_row[elo_col].values[0]) if not home_row.empty else 1500.0
    ae = float(away_row[elo_col].values[0]) if not away_row.empty else 1500.0
    return round(he - ae, 1)


def _extract_market_features(home: str, away: str,
                              historical: pd.DataFrame,
                              ref_date: datetime) -> dict:
    """Solo para inferencia — NO incluir en training set."""
    h = normalize_team_name(home)
    a = normalize_team_name(away)
    default = {"market_prob_home": 0.45, "market_prob_draw": 0.27,
               "market_prob_away": 0.28}

    if "home_team_norm" not in historical.columns:
        return default

    mask   = (historical["home_team_norm"] == h) & (historical["away_team_norm"] == a)
    recent = historical[mask & (historical["match_date"] < ref_date)]
    recent = recent.sort_values("match_date", ascending=False).head(5)

    for col_h, col_d, col_a in [
        ("PSH",   "PSD",   "PSA"),
        ("B365H", "B365D", "B365A"),
        ("BWH",   "BWD",   "BWA"),
    ]:
        if not all(c in recent.columns for c in [col_h, col_d, col_a]):
            continue
        valid = recent[[col_h, col_d, col_a]].dropna()
        if valid.empty:
            continue
        row   = valid.iloc[0]
        total = 1/row[col_h] + 1/row[col_d] + 1/row[col_a]
        return {
            "market_prob_home": round((1/row[col_h]) / total, 4),
            "market_prob_draw": round((1/row[col_d]) / total, 4),
            "market_prob_away": round((1/row[col_a]) / total, 4),
        }

    return default


# ─── BUILDER PRINCIPAL ───────────────────────────────────────────────────────

def build_features_for_fixtures(fixtures: pd.DataFrame,
                                 espn_client=None) -> pd.DataFrame:
    """
    Construye features para todos los fixtures del día.

    v7: además de cuotas ESPN, enriquece con:
      - Lesiones (si ESPN_INJURIES_ENABLED=true y espn_client disponible)
      - ESPN BPI (si ESPN_BPI_ENABLED=true y espn_client disponible)

    v8 (standings context): enriquece con features de motivación si
      ESPN_STANDINGS_FEATURES=true. Añade relegation_threat, title_race,
      motivation_score, etc. para local y visitante.

    espn_client: si se pasa, activa el enriquecimiento con fuentes ESPN en tiempo real.
    """
    log.info("Cargando datos históricos para feature building...")
    historical    = load_historical_results()
    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()

    if historical.empty:
        log.error("Sin datos históricos. Imposible calcular features.")
        return pd.DataFrame()

    # ── Enriquecimiento con fuentes ESPN en tiempo real ───────────────────
    if espn_client is not None and not fixtures.empty:

        # 1. Cuotas en tiempo real
        try:
            from src.espn_collector import enrich_fixtures_with_odds
            fixtures = enrich_fixtures_with_odds(espn_client, fixtures)
            log.info("Fixtures enriquecidos con cuotas ESPN")
        except Exception as e:
            log.warning(f"No se pudo enriquecer con cuotas ESPN: {e}")

        # 2. Lesiones (v7 — NUEVO)
        if ESPN_INJURIES_ENABLED:
            try:
                from src.espn_collector import enrich_fixtures_with_injuries
                fixtures = enrich_fixtures_with_injuries(espn_client, fixtures)
                log.info("Fixtures enriquecidos con lesiones ESPN")
            except Exception as e:
                log.warning(f"No se pudo enriquecer con lesiones ESPN: {e}")

        # 3. ESPN BPI (v7 — NUEVO, v8 — condicional por liga)
        # ESPN BPI no devuelve datos para ligas LATAM. Se salta el enriquecimiento
        # cuando el slug de la liga está en SLUGS_SIN_BPI para ahorrar llamadas
        # a la API que siempre devuelven 0.
        if ESPN_BPI_ENABLED:
            try:
                from config.settings import SLUGS_SIN_BPI
                # Detectar slugs de los fixtures actuales
                slugs_en_fixtures: set[str] = set()
                if "slug" in fixtures.columns:
                    slugs_en_fixtures = set(fixtures["slug"].dropna().unique())
                elif "league_id" in fixtures.columns:
                    # Mapeo inverso league_id → slug desde LIGAS_ESPN
                    from config.settings import LIGAS_ESPN
                    _id_to_slug = {lid: slug for slug, (lid, _) in LIGAS_ESPN.items()}
                    for lid in fixtures["league_id"].dropna().unique():
                        s = _id_to_slug.get(int(lid))
                        if s:
                            slugs_en_fixtures.add(s)

                # Solo llamar al API si hay al menos un slug con BPI disponible
                slugs_con_bpi = slugs_en_fixtures - SLUGS_SIN_BPI
                if slugs_con_bpi:
                    from src.espn_collector import enrich_fixtures_with_bpi
                    fixtures = enrich_fixtures_with_bpi(espn_client, fixtures)
                    log.info(f"Fixtures enriquecidos con ESPN BPI (slugs: {slugs_con_bpi})")
                else:
                    log.info("ESPN BPI omitido — todos los slugs son LATAM/sin cobertura BPI")
            except Exception as e:
                log.warning(f"No se pudo enriquecer con ESPN BPI: {e}")

        # 4. Standings context — motivación (v8 — NUEVO)
        # Una sola llamada por liga, cacheada en memoria durante el pipeline.
        # Añade relegation_threat, title_race, motivation_score, etc.
        _standings_context: dict = {}
        if ESPN_STANDINGS_FEATURES:
            try:
                from src.espn_collector import (
                    get_standings_context, enrich_fixtures_with_standings,
                )
                from config.settings import LIGAS_ESPN, LIGAS_ESPN_ACTIVAS

                # Solo pedir standings de las ligas que aparecen en los fixtures
                ligas_en_fixtures: dict[str, tuple] = {}
                if "league_id" in fixtures.columns:
                    _id_to_slug = {lid: (slug, name)
                                   for slug, (lid, name) in LIGAS_ESPN.items()}
                    for lid in fixtures["league_id"].dropna().unique():
                        info = _id_to_slug.get(int(lid))
                        if info:
                            slug_f, name_f = info
                            if slug_f in LIGAS_ESPN_ACTIVAS:
                                ligas_en_fixtures[slug_f] = (int(lid), name_f)

                if ligas_en_fixtures:
                    _standings_context = get_standings_context(
                        espn_client, slugs=ligas_en_fixtures
                    )
                    fixtures = enrich_fixtures_with_standings(
                        fixtures, _standings_context
                    )
                    log.info(
                        f"Fixtures enriquecidos con standings context "
                        f"({len(ligas_en_fixtures)} ligas)"
                    )
            except Exception as e:
                log.warning(f"No se pudo enriquecer con standings context: {e}")

    ref_date     = datetime.now()
    all_features = []

    for _, fixture in fixtures.iterrows():
        home = fixture["home_team"]
        away = fixture["away_team"]

        home_norm = normalize_team_name(home)
        away_norm = normalize_team_name(away)
        if home_norm != home.lower().strip():
            log.info(f"  Nombre resuelto: '{home}' → '{home_norm}'")
        if away_norm != away.lower().strip():
            log.info(f"  Nombre resuelto: '{away}' → '{away_norm}'")

        log.info(f"Features: {home} vs {away}")

        home_stats = compute_team_stats(home, True,  historical, xg_data,
                                        match_summary, ref_date)
        away_stats = compute_team_stats(away, False, historical, xg_data,
                                        match_summary, ref_date)
        h2h        = compute_h2h(home, away, historical, ref_date)
        elo_diff   = get_elo_diff(home, away, elo_df)
        weather    = get_weather_for_fixture(
                        home, fixture.get("date", ref_date.isoformat()))
        market     = _extract_market_features(home, away, historical, ref_date)

        feature_row = {
            "fixture_id":      fixture.get("fixture_id", 0),
            "league_id":       fixture.get("league_id", 0),
            "league_name":     fixture.get("league_name", ""),
            "home_team":       home,
            "away_team":       away,
            "match_date":      fixture.get("date", str(ref_date.date())),
            "source":          fixture.get("source", "fdorg"),
            "n_home_matches":  home_stats.get("home_n_matches_total", 0),
            "n_away_matches":  away_stats.get("away_n_matches_total", 0),
            "elo_diff":        elo_diff,
            **home_stats,
            **away_stats,
            **h2h,
            **weather,
            **market,
        }

        # ── Cuotas ESPN ───────────────────────────────────────────────────
        if fixture.get("espn_odds_available"):
            feature_row["espn_odds_home"]     = fixture.get("espn_odds_home")
            feature_row["espn_odds_draw"]      = fixture.get("espn_odds_draw")
            feature_row["espn_odds_away"]      = fixture.get("espn_odds_away")
            feature_row["espn_odds_provider"]  = fixture.get("espn_odds_provider")
            feature_row["espn_odds_available"] = True
        else:
            feature_row["espn_odds_available"] = False

        # ── Lesiones ESPN (v7 — NUEVO) ────────────────────────────────────
        feature_row["home_injured_count"]  = fixture.get("home_injured_count",  0)
        feature_row["home_injury_score"]   = fixture.get("home_injury_score",   0.0)
        feature_row["home_out_count"]      = fixture.get("home_out_count",      0)
        feature_row["away_injured_count"]  = fixture.get("away_injured_count",  0)
        feature_row["away_injury_score"]   = fixture.get("away_injury_score",   0.0)
        feature_row["away_out_count"]      = fixture.get("away_out_count",      0)

        # Diferencial de lesiones: positivo = visitante más afectado que local
        feature_row["injury_score_diff"] = round(
            feature_row["away_injury_score"] - feature_row["home_injury_score"], 2
        )

        # ── ESPN BPI (v7 — NUEVO, v8 — flag bpi_available) ──────────────────
        # Se guarda el flag binario `bpi_available` en lugar de las probs crudas.
        # Las probs siempre son 0 en LATAM (fillna), lo que introduce ruido.
        # El flag permite al modelo distinguir ausencia de dato vs BPI real=0.
        bpi_home = fixture.get("espn_bpi_home_prob")
        bpi_away = fixture.get("espn_bpi_away_prob")
        bpi_avail = bool(fixture.get("espn_bpi_available", False))
        feature_row["espn_bpi_home_prob"] = bpi_home   # guardado para diagnóstico
        feature_row["espn_bpi_away_prob"] = bpi_away   # guardado para diagnóstico
        feature_row["espn_bpi_available"] = bpi_avail
        feature_row["bpi_available"]      = int(bpi_avail)  # feature binaria para XGBoost

        # ── Standings context — motivación (v8 — NUEVO) ──────────────────
        # Features de situación en tabla: relegation_threat, title_race,
        # motivation_score, points_to_safety, rank, etc.
        # Se propagan desde fixtures (ya enriquecido por enrich_fixtures_with_standings).
        _standing_cols = [
            "standing_rank", "standing_points", "standing_pts_per_game",
            "standing_goal_diff", "points_to_safety", "points_to_clasif",
            "title_race", "clasif_race", "relegation_threat",
            "season_progress", "es_tramo_final", "motivation_score",
        ]
        for prefix in ("home", "away"):
            for col in _standing_cols:
                val = fixture.get(f"{prefix}_{col}", float("nan"))
                # Convertir NaN a 0 para XGBoost — NaN significa "dato no disponible"
                # (equipo no encontrado en standings, p.ej. copa con fase de grupos)
                import math
                feature_row[f"{prefix}_{col}"] = 0.0 if (val is None or (isinstance(val, float) and math.isnan(val))) else val

        # Diferenciales de motivación
        for diff_col in ("rank_diff", "points_diff_standing",
                         "motivation_diff", "pressure_asymmetry"):
            val = fixture.get(diff_col, 0.0)
            import math
            feature_row[diff_col] = 0.0 if (val is None or (isinstance(val, float) and math.isnan(val))) else val

        all_features.append(feature_row)

    df = pd.DataFrame(all_features)
    if not df.empty:
        df = _add_derived_features(df)
        path = os.path.join(DATA_PROCESSED, f"features_{date.today()}.csv")
        df.to_csv(path, index=False)
        log.info(f"Features guardados: {len(df)} partidos → {path}")
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["xg_diff"]      = df["home_xg_scored"]   - df["away_xg_scored"]
    df["xg_total_exp"] = df["home_xg_scored"]   + df["away_xg_scored"]
    df["goals_diff"]   = df["home_goals_scored"] - df["away_goals_scored"]
    df["forma_diff"]   = df["home_forma"]        - df["away_forma"]
    df["rest_diff"]    = df["away_days_rest"]    - df["home_days_rest"]
    df["fatiga_flag"]  = (
        (df["home_days_rest"] < 4) | (df["away_days_rest"] < 4)
    ).astype(int)
    return df


# ─── DATASET DE ENTRENAMIENTO ────────────────────────────────────────────────

def _precompute_rolling_cache(
    historical: pd.DataFrame,
    xg_data: pd.DataFrame,
    match_summary: pd.DataFrame,
    window: int = VENTANA_FORMA,
) -> dict[tuple, dict]:
    """
    Pre-computa stats de forma por (team_norm, is_home, match_idx).
    O(n) en lugar de O(n²).
    """
    log.info("Pre-computando rolling stats por equipo (O(n) cache)...")

    home_indices: dict[str, list[int]] = {}
    away_indices: dict[str, list[int]] = {}

    for idx, row in historical.iterrows():
        h = str(row.get("home_team_norm", normalize_team_name(str(row.get("home_team", "")))))
        a = str(row.get("away_team_norm", normalize_team_name(str(row.get("away_team", "")))))
        home_indices.setdefault(h, []).append(idx)
        away_indices.setdefault(a, []).append(idx)

    cache: dict[tuple, dict] = {}

    def _stats_from_rows(team: str, is_home: bool, rows: pd.DataFrame,
                         ref_date, team_norm: str) -> dict:
        p        = "home" if is_home else "away"
        col_scored = "home_goals" if is_home else "away_goals"
        col_recv   = "away_goals" if is_home else "home_goals"

        n_total = len(rows)
        df      = rows.sort_values("match_date", ascending=False).head(window).copy()

        if df.empty:
            return _empty_team_stats(is_home)

        df["days_ago"] = (ref_date - df["match_date"]).dt.days
        df["weight"]   = df["days_ago"].apply(exponential_weight)
        W = df["weight"].sum() or 1.0

        goals_scored   = (df[col_scored] * df["weight"]).sum() / W
        goals_conceded = (df[col_recv]   * df["weight"]).sum() / W

        xg_scored   = _compute_xg(team_norm, xg_data, is_home, "attack")
        xg_conceded = _compute_xg(team_norm, xg_data, is_home, "defense")
        if xg_scored   == 0.0: xg_scored   = goals_scored   * 1.05
        if xg_conceded == 0.0: xg_conceded = goals_conceded * 1.05

        def pts(row):
            s, c = row[col_scored], row[col_recv]
            return 3 if s > c else (1 if s == c else 0)

        df["pts"]    = df.apply(pts, axis=1)
        forma        = (df["pts"] * df["weight"]).sum() / W

        df["btts"]   = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
        btts_rate    = (df["btts"]  * df["weight"]).sum() / W

        df["over25"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
        over25_rate  = (df["over25"] * df["weight"]).sum() / W

        corners_avg, fouls_avg = _compute_corners_fouls(team_norm, match_summary, is_home)

        last_date = df["match_date"].max()
        days_rest = int((ref_date - last_date).days) if not pd.isna(last_date) else 7

        return {
            f"{p}_xg_scored":       round(xg_scored,    3),
            f"{p}_xg_conceded":     round(xg_conceded,  3),
            f"{p}_goals_scored":    round(goals_scored,  3),
            f"{p}_goals_conceded":  round(goals_conceded,3),
            f"{p}_forma":           round(forma,         3),
            f"{p}_btts_rate":       round(btts_rate,     3),
            f"{p}_over25_rate":     round(over25_rate,   3),
            f"{p}_corners_avg":     round(corners_avg,   2),
            f"{p}_fouls_avg":       round(fouls_avg,     2),
            f"{p}_days_rest":       days_rest,
            f"{p}_n_matches":       len(df),
            f"{p}_n_matches_total": n_total,
        }

    all_teams   = set(home_indices.keys()) | set(away_indices.keys())
    total_teams = len(all_teams)

    for t_idx, team_norm in enumerate(all_teams):
        if t_idx % 50 == 0:
            log.info(f"  Rolling cache: equipo {t_idx}/{total_teams}")

        h_idxs = sorted(home_indices.get(team_norm, []))
        a_idxs = sorted(away_indices.get(team_norm, []))

        seen_h: list[int] = []
        for i in h_idxs:
            past_rows = historical.loc[[j for j in seen_h]]
            ref_date  = historical.at[i, "match_date"]
            cache[(team_norm, True, i)] = _stats_from_rows(
                team_norm, True, past_rows, ref_date, team_norm
            )
            seen_h.append(i)

        seen_a: list[int] = []
        for i in a_idxs:
            past_rows = historical.loc[[j for j in seen_a]]
            ref_date  = historical.at[i, "match_date"]
            cache[(team_norm, False, i)] = _stats_from_rows(
                team_norm, False, past_rows, ref_date, team_norm
            )
            seen_a.append(i)

    log.info(f"Rolling cache listo: {len(cache)} entradas para {total_teams} equipos")
    return cache


def build_training_dataset(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset walk-forward sin data leakage.
    v7: añade columnas de lesiones y BPI como NaN (no disponibles en histórico)
    para mantener schema consistente con build_features_for_fixtures.
    """
    import time
    log.info("Construyendo dataset walk-forward (sin leakage)...")
    t0 = time.time()

    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()

    historical = historical.sort_values("match_date").reset_index(drop=True)

    if "home_team_norm" not in historical.columns:
        historical["home_team_norm"] = historical["home_team"].apply(normalize_team_name)
    if "away_team_norm" not in historical.columns:
        historical["away_team_norm"] = historical["away_team"].apply(normalize_team_name)

    rolling_cache = _precompute_rolling_cache(historical, xg_data, match_summary)

    start_idx   = max(200, len(historical) // 4)
    training_rows = []
    total_range = len(historical) - start_idx

    for idx in range(start_idx, len(historical)):
        match = historical.iloc[idx]

        home      = str(match.get("home_team", ""))
        away      = str(match.get("away_team", ""))
        home_norm = str(match.get("home_team_norm", normalize_team_name(home)))
        away_norm = str(match.get("away_team_norm", normalize_team_name(away)))
        if not home or not away:
            continue

        home_stats = rolling_cache.get(
            (home_norm, True,  idx),
            compute_team_stats(home, True,  historical.iloc[:idx], xg_data,
                               match_summary, match["match_date"]),
        )
        away_stats = rolling_cache.get(
            (away_norm, False, idx),
            compute_team_stats(away, False, historical.iloc[:idx], xg_data,
                               match_summary, match["match_date"]),
        )

        h2h      = compute_h2h(home, away, historical.iloc[:idx], match["match_date"])
        elo_diff = get_elo_diff(home, away, elo_df)

        hg = int(match["home_goals"])
        ag = int(match["away_goals"])

        training_rows.append({
            "home_team":           home,
            "away_team":           away,
            "match_date":          match["match_date"],
            "league_name":         match.get("league_name", ""),
            "elo_diff":            elo_diff,
            "n_home_matches":      home_stats.get("home_n_matches_total", 0),
            "n_away_matches":      away_stats.get("away_n_matches_total", 0),
            **home_stats,
            **away_stats,
            **h2h,
            # v7: columnas de lesiones como NaN (no disponibles en histórico)
            "home_injured_count":  np.nan,
            "home_injury_score":   np.nan,
            "home_out_count":      np.nan,
            "away_injured_count":  np.nan,
            "away_injury_score":   np.nan,
            "away_out_count":      np.nan,
            "injury_score_diff":   np.nan,
            # v7/v8: BPI — probs como NaN (no disponible en histórico);
            # flag bpi_available=0 (histórico nunca tiene BPI → fillna(0) correcto)
            "espn_bpi_home_prob":  np.nan,
            "espn_bpi_away_prob":  np.nan,
            "bpi_available":       0,
            # v11: standings context — inicializado en 0 para histórico
            # (no tenemos standings históricos, pero el modelo aprende
            # que 0 = "dato no disponible" y lo pondera menos)
            "home_relegation_threat":    0,
            "away_relegation_threat":    0,
            "home_title_race":           0,
            "away_title_race":           0,
            "home_clasif_race":          0,
            "away_clasif_race":          0,
            "home_motivation_score":     0.0,
            "away_motivation_score":     0.0,
            "home_standing_pts_per_game":0.0,
            "away_standing_pts_per_game":0.0,
            "home_points_to_safety":     0.0,
            "away_points_to_safety":     0.0,
            "home_es_tramo_final":       0,
            "away_es_tramo_final":       0,
            "motivation_diff":           0.0,
            "pressure_asymmetry":        0.0,
            "rank_diff":                 0.0,
            "points_diff_standing":      0.0,
            # targets
            "target_home_win":     int(hg > ag),
            "target_draw":         int(hg == ag),
            "target_away_win":     int(hg < ag),
            "target_btts":         int(hg > 0 and ag > 0),
            "target_over25":       int(hg + ag > 2.5),
            "home_goals_actual":   hg,
            "away_goals_actual":   ag,
        })

        if (idx - start_idx) % 500 == 0:
            pct = (idx - start_idx) / total_range * 100
            elapsed = time.time() - t0
            log.info(f"  Training dataset: {pct:.0f}% ({idx}/{len(historical)}) "
                     f"— {elapsed:.1f}s")

    df = pd.DataFrame(training_rows)
    if not df.empty:
        df = _add_derived_features(df)
        if "home_team_norm" not in df.columns:
            df["home_team_norm"] = df["home_team"].apply(normalize_team_name)
            df["away_team_norm"] = df["away_team"].apply(normalize_team_name)
        path = os.path.join(DATA_PROCESSED, "training_dataset.csv")
        df.to_csv(path, index=False)
        elapsed = time.time() - t0
        log.info(f"Dataset entrenamiento: {len(df)} partidos → {path} ({elapsed:.1f}s total)")
    return df