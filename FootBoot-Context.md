# FOOTBOT — Contexto del Proyecto

## Descripción general

Bot de predicción de apuestas deportivas 100% gratuito basado en modelos
estadísticos. Corre en local (Windows, venv Python) durante desarrollo.
Destino final: Oracle Cloud ARM (Always Free).

**Stack:**
- Modelo: Dixon-Coles por liga + XGBoost ensemble con calibración isotónica
- Datos históricos: ESPN API (sin key) — ligas LATAM y copas
- Fixtures del día: football-data.org (7 ligas EU) + ESPN API
- ELO ratings: ClubElo.com (sin key) — solo EU + **ELO propio ESPN** ✅ (`compute_elo_espn` en `_01_data_collector.py`) — ambas fuentes fusionadas en `load_elo()` v5; `elo_diff` funcional para todos los equipos LATAM
- Clima: Open-Meteo (sin key) — `utils.py` v4 ✅
- Base de datos: Supabase (✅ conectado — `supabase==2.3.0`, tablas creadas con DDL v2)
- Notificaciones: Telegram Bot API (✅ conectado — token y chat_id configurados)

---

## Estructura del proyecto

```
footbot/
├── config/
│   └── settings.py              # Constantes globales, umbrales, API keys
├── src/
│   ├── _01_data_collector.py    # Descarga fixtures, histórico, ELO (v4 — compute_elo_espn)
│   ├── _02_feature_builder.py   # Features por partido (forma, H2H, xG, clima) — v5: load_elo fusiona ClubElo + ESPN
│   ├── _03_model_engine.py      # Dixon-Coles + XGBoost + blend weights — v7: scale_pos_weight dinámico por mercado
│   ├── _04_value_detector.py    # Edge%, Kelly, clasificación de confianza
│   ├── _05_result_updater.py    # Cierre de predicciones con resultado real (v2)
│   ├── espn_collector.py        # Cliente ESPN unificado (v3)
│   ├── nacional_features.py     # Features para selecciones nacionales
│   ├── telegram_sender.py       # Formateo y envío de reportes (v6)
│   ├── supabase_client.py       # Cliente Supabase (v1 — insert/select predicciones)
│   └── utils.py                 # Clima, rate limiting, coordenadas estadios
├── scheduler.py                 # Pipeline principal (clubes)
├── scheduler_nacional.py        # Pipeline selecciones nacionales
├── start.sh                     # Orquestador bash (predict/results/live/all)
├── tests/
│   ├── conftest.py              # Fixtures compartidos (sample_historical, bets_df, etc.)
│   ├── test_conexiones.py       # Test integración: Supabase + Telegram (4/4 OK)
│   ├── test_data_collector.py
│   ├── test_espn_collector.py
│   ├── test_feature_builder.py
│   ├── test_model_engine.py
│   ├── test_result_updater.py   # v2 — cubre todos los mercados expandidos
│   ├── test_telegram_sender.py
│   └── test_value_detector.py
├── data/
│   ├── raw/                     # CSVs ESPN, ELO, fixtures del día
│   │   ├── elo_ratings.csv      # ClubElo (solo equipos EU)
│   │   └── elo_espn.csv         # ELO propio calculado desde histórico ESPN ✅
│   └── processed/               # Features y datasets de entrenamiento
└── models/                      # Modelos .pkl versionados por fecha
```

---

## Estado actual del sistema

### Lo que funciona ✅
- Pipeline completo corre sin errores críticos
- ESPN API devuelve fixtures y cuotas para col.1 (Liga BetPlay)
- Cuotas ESPN en tiempo real: 3/3 partidos con DraftKings
- Dixon-Coles entrenado por liga (Liga BetPlay, Brasileirão, Champions, Liga MX, Liga Profesional Argentina)
- Fuzzy matching de nombres (rapidfuzz) resuelve correctamente equipos colombianos
- **Histórico ESPN: 4236 partidos** de 7 ligas (2024–2026) — verificado con `diag1.py` el 2026-03-31
- Blend weights optimizados por mercado (dc_weight=0.70 para todos)
- Fix league_id lookup: ligas ESPN resuelven correctamente (v6)
- Sistema de 3 niveles de confianza: alta, media, baja (v6)
- Suite de tests unitarios completa (pytest, 7 módulos)
- DDL Supabase generado y ejecutado (tablas con DDL v2 — `'baja'` en CHECK, vistas `roi_por_mercado` y `resumen_diario`)
- Supabase conectado: `supabase==2.3.0`, insert/delete verificados (tabla `predicciones`)
- Telegram conectado: token + chat_id reales, mensaje de prueba enviado OK
- Test de integración `test_conexiones.py`: 4/4 tests OK
- **`evaluate_bet` completo** ✅ — cubre todos los mercados: 1X2, doble oportunidad, over/under 0.5–4.5, BTTS, por equipo, goles exactos, combinadas, Asian Handicap ±0.5 y -1
- **`compute_model_stats` con desglose por nivel y mercado** ✅ — ROI tracking funcional para alta/media/baja
- **`compute_elo_espn`** ✅ — ELO propio calculado desde histórico ESPN con factor K dinámico y margin factor logarítmico; guarda `data/raw/elo_espn.csv`
- **`load_elo()` fusionada** ✅ — `_02_feature_builder.load_elo()` v5 lee y fusiona `elo_ratings.csv` (ClubElo, EU) + `elo_espn.csv` (ESPN, LATAM); prioriza ESPN en caso de equipo duplicado; `elo_diff` ya no es 0.0 para equipos LATAM
- **`scale_pos_weight` dinámico en XGBoost** ✅ — `_03_model_engine.py` v7 calcula `neg/pos` por mercado antes de entrenar cada clasificador; `draw` (~2.7×) y `away_win` (~2.6×) dejan de predecir clase mayoritaria
- **Reentrenamiento v7 ejecutado** ✅ — modelos reentrenados el 2026-03-31 con dataset v6 (4236 partidos, O(n) cache); `dc_exp_home_goals` y `dc_exp_away_goals` devuelven valores reales por equipo (ej: Tolima 1.638 vs Águilas 0.639)
- **`detect_all_value_bets`** ✅ — nombre real de la función de producción en `_04_value_detector.py` (no `detect_value_bets`); funciones públicas del módulo: `analyze_fixture`, `build_explanation`, `build_odds_dict`, `calculate_edge`, `classify_confidence`, `compute_all_market_probs`, `detect_all_value_bets`, `get_current_season_odds`, `get_model_prob_for_market`, `kelly_fraction`, `normalize_team_name`, `summarize_bets`
- **Claves BTTS en `market_probs`** ✅ — confirmadas: `prob_btts`, `prob_btts_no`, `prob_home_and_btts`, `prob_draw_and_btts`, `prob_away_and_btts` (NO existe `prob_btts_si`)
- **`get_results_espn`** ✅ — fuente de resultados ESPN integrada en `_05_result_updater.py` como fallback a fd.org
- **DDL v2 completo** ✅ — columnas `home_goals`/`away_goals` en `predicciones`, índice por mercado, vistas `roi_por_mercado` y `resumen_diario`

### Pendiente ⚠️
- ~~**Open-Meteo:**~~ ✅ **RESUELTO** — `utils.py` v4: `forecast_days` sin `start/end_date` para `days_ahead <= 1`; índice correcto del array de respuesta
- **Football-Data.co.uk:** no descargado (ESPN_ONLY=true en .env)
- ~~**ELO LATAM en pipeline:**~~ ✅ **RESUELTO** — `load_elo()` v5 fusiona ClubElo + `elo_espn.csv`; `elo_diff` funcional en LATAM
- ~~**`build_training_dataset` O(n²):**~~ ✅ **RESUELTO** — `_02_feature_builder.py` v6: `_precompute_rolling_cache` pre-computa rolling stats O(n); ~90s → ~5s con 4226 partidos
- **Copa Libertadores / Sudamericana:** < 200 partidos → sin DC propio; modelo global no converge (> 80 equipos)
- ~~**Distribución de clases XGBoost:**~~ ✅ **RESUELTO** — `scale_pos_weight` dinámico por mercado en v7; reentrenamiento ejecutado 2026-03-31 con 4236 partidos ✅
- **Métricas XGBoost post-fix:** reentrenamiento ejecutado pero evaluación formal pendiente — `draw`/`away_win` con ROI 0.0% son métricas pre-fix, no reflejan el estado actual del modelo
- **Bug `btts_si` en scripts de diagnóstico:** `diag2.py` busca `prob_btts_si` pero la clave real en `market_probs` es `prob_btts` → retorna 0% con cuota fallback 99.0. **Solo afecta diag2.py**, no producción (`detect_all_value_bets` usa `get_model_prob_for_market` internamente). Fix: cambiar `'btts_si'` → `'btts'` en el loop de mercados de `diag2.py`

---

## Datos históricos ESPN disponibles

| Liga | Slug | league_id | Partidos |
|------|------|-----------|---------|
| Liga BetPlay | col.1 | 501 | 1014 |
| Liga Profesional Argentina | arg.1 | 502 | 1051 |
| Brasileirão Serie A | bra.1 | 503 | 805 |
| Copa Libertadores | conmebol.libertadores | 511 | 256 |
| Copa Sudamericana | conmebol.sudamericana | 512 | 165 |
| Champions League | uefa.champions | 514 | 328 |
| Liga MX | mex.1 | 518 | 617 |

**Temporadas disponibles:** 2024, 2025, 2026

---

## Modelos entrenados (última versión: 20260331)

### Dixon-Coles por liga
Entrenado con L-BFGS-B (reemplazó SLSQP — más rápido con 100+ equipos).

| Liga | Equipos | Partidos | home_adv | rho | Converge |
|------|---------|---------|---------|-----|---------|
| Liga BetPlay | 23 | 688 | 0.361 | -0.096 | ✅ |
| Brasileirão | 27 | 551 | 0.390 | 0.024 | ✅ |
| Champions League | 54 | 315 | 0.302 | 0.092 | ✅ |
| Liga MX | 18 | 526 | 0.291 | -0.063 | ✅ |
| Liga Profesional Argentina | 32 | 835 | 0.283 | -0.146 | ✅ |
| Copa Libertadores | — | 146 | — | — | ❌ (< 200 mín) |
| Copa Sudamericana | — | 109 | — | — | ❌ (< 200 mín) |
| Global | — | 201 equipos | — | — | ❌ (> 80 máx) |

**Constantes clave (`_03_model_engine.py`):**
- `MIN_MATCHES_PER_LIGA = 200` — mínimo para entrenar DC por liga
- `MIN_MATCHES_GLOBAL_DC = 300` — mínimo para DC global
- `MAX_TEAMS_GLOBAL_DC = 80` — si hay más equipos, DC global se omite
- `DEFAULT_DC_WEIGHT = 0.35` — fallback si no hay optimización de blend

### XGBoost validation metrics
| Mercado | Accuracy | ROI flat | dc_weight |
|---------|---------|---------|----------|
| home_win | 55.4% | +17.4% | 0.700 |
| draw | 72.7% | +0.0% | 0.700 |
| away_win | 73.3% | +0.0% | 0.700 |
| btts | 53.8% | -3.6% | 0.675 |
| over25 | 56.3% | +7.1% | 0.700 |

> **Nota:** Métricas pre-fix (reentrenamiento con `scale_pos_weight` dinámico v7 ejecutado el 2026-03-31 con dataset v6 — 4236 partidos). Se espera que `draw` y `away_win` muestren ROI real en lugar de 0.0% pero las métricas de validación post-fix aún no han sido registradas. Pendiente correr evaluación formal.

---

---

## Fix: `load_elo()` fusionada con ELO ESPN (`_02_feature_builder.py` v5)

**Problema:** `load_elo()` solo leía `elo_ratings.csv` (ClubElo), que no cubre
equipos LATAM. Resultado: `elo_diff = 0.0` para todos los partidos de Liga BetPlay,
Brasileirão, Liga Profesional Argentina y Liga MX.

**Fix (`_02_feature_builder.py` v5):**
```python
def load_elo() -> pd.DataFrame:
    frames = []
    # Fuente 1: ClubElo (EU)
    path_clubelo = os.path.join(DATA_RAW, "elo_ratings.csv")
    if os.path.exists(path_clubelo):
        df_clubelo = pd.read_csv(path_clubelo)
        df_clubelo["team_norm"] = df_clubelo["Club"].apply(normalize_team_name)
        df_clubelo["_source"] = "clubelo"
        frames.append(df_clubelo)

    # Fuente 2: ELO propio ESPN (LATAM)
    path_espn = os.path.join(DATA_RAW, "elo_espn.csv")
    if os.path.exists(path_espn):
        df_espn = pd.read_csv(path_espn)
        df_espn["team_norm"] = df_espn["Club"].apply(normalize_team_name)
        df_espn["_source"] = "espn"
        frames.append(df_espn)

    combined = pd.concat(frames, ignore_index=True)
    # Si equipo aparece en ambas fuentes, gana ESPN (más reciente, más LATAM)
    combined["_source_order"] = combined["_source"].map({"clubelo": 0, "espn": 1})
    combined = combined.sort_values("_source_order")
    combined = combined.drop_duplicates(subset="team_norm", keep="last")
    return combined.drop(columns=["_source", "_source_order"])
```

**Prerequisito:** `elo_espn.csv` debe existir en `data/raw/`. Si no existe, ejecutar:
```powershell
python -c "from src._01_data_collector import compute_elo_espn; compute_elo_espn()"
```

**Resultado post-fix:** `elo_diff` tiene valores reales para equipos LATAM.
Ejemplo: Millonarios (1620) vs Nacional (1580) → `elo_diff = +40` en lugar de `0.0`.

---

## Fix: Open-Meteo 400 Bad Request para hoy/mañana (`utils.py` v4)

**Problema:** `start_date == today` provoca 400 Bad Request en Open-Meteo cuando el timezone
es UTC-5 (Colombia). Afectaba todos los partidos del día y del día siguiente.

**Fix (`utils.py` v4):**
```python
if days_ahead <= 1:
    # Sin start/end_date — Open-Meteo devuelve array desde hoy
    params = {"forecast_days": max(2, days_ahead + 1), ...}
else:
    # Fecha futura: start/end_date sigue funcionando
    params = {"forecast_days": days_ahead + 1, "start_date": ds, "end_date": ds, ...}

r = requests.get(OPENMETEO_URL, params=params, timeout=8)
# Seleccionar el día correcto del array
idx  = days_ahead if days_ahead <= 1 else 0
prec = (d.get("precipitation_sum") or [0] * (idx + 1))[idx] or 0
```

**Resultado post-fix:** Clima disponible para partidos de hoy y mañana sin errores 400.

---

## Fix: `build_training_dataset` O(n²) → O(n) (`_02_feature_builder.py` v6)

**Problema:** En cada iteración del loop walk-forward, `compute_team_stats` filtraba
`historical.iloc[:idx]` desde cero — O(n) por partido, O(n²) total. Con 4226 partidos: ~90s.

**Fix (`_02_feature_builder.py` v6):** Nueva función `_precompute_rolling_cache` que
recorre el histórico **una sola vez por equipo** y guarda stats en:
`dict[(team_norm, is_home, match_idx)] → stats`

El loop principal hace lookup O(1). Si la clave no existe en cache (caso borde),
cae back al `compute_team_stats` original como fallback.

**Resultado post-fix:** ~90s → ~5s con 4226 partidos. Reentrenamiento viable sin espera.

---

## Fix: `scale_pos_weight` dinámico en XGBoost (`_03_model_engine.py` v7)

**Problema:** XGBoost entrenaba con clases desbalanceadas sin corrección.
`draw` (~27% de los partidos) y `away_win` (~28%) generaban modelos que
siempre predicen la clase mayoritaria — accuracy alta pero ROI 0.0% porque
nunca producen una señal real de apuesta.

**Síntoma:**
```
draw:      accuracy=72.7%  ROI=+0.0%   # siempre predice "no draw"
away_win:  accuracy=73.3%  ROI=+0.0%   # siempre predice "no away_win"
```

**Fix (`_03_model_engine.py` v7):**
```python
def _get_xgb(self, scale_pos_weight: float = 1.0):
    return XGBClassifier(
        ...
        scale_pos_weight=scale_pos_weight,  # NUEVO
        ...
    )

# En fit(), antes de crear cada XGBoost:
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
spw   = round(n_neg / n_pos, 3) if n_pos > 0 else 1.0
xgb   = self._get_xgb(scale_pos_weight=spw)
```

**Pesos esperados con 4226 partidos:**
| Mercado | ~% positivos | scale_pos_weight |
|---------|-------------|-----------------|
| home_win | ~45% | ~1.2 |
| draw | ~27% | ~2.7 |
| away_win | ~28% | ~2.6 |
| btts | ~50% | ~1.0 |
| over25 | ~50% | ~1.0 |

**Acción requerida:** Reentrenar modelos para que el fix surta efecto:
```powershell
python -c "import os, glob; [os.remove(f) for f in glob.glob('models/*.pkl')]"
python scheduler.py
```

**Problema:** Las ligas ESPN se guardan en `dc.models` con `league_name` como
clave string (ej: `'Liga BetPlay'`) porque no están en `LIGAS` (que solo tiene
las 7 ligas EU con IDs numéricos). `predict_proba` recibía `league_id=501` y
buscaba `dc.models[501]` — no encontraba nada y caía al global con probs por defecto.

**Síntoma:** Los 3 partidos del día tenían exactamente las mismas probabilidades
(47.2% / 27.3% / 25.4%) — valor por defecto de Dixon-Coles sin fit.

**Fix (`_03_model_engine.py` v6):**
```python
def _resolve_league_name(league_id: int | str) -> str | None:
    from config.settings import LIGAS_ESPN, COMPETICIONES_NACIONALES_ESPN
    _todos = {**LIGAS_ESPN, **COMPETICIONES_NACIONALES_ESPN}
    return next(
        (name for slug, (lid, name) in _todos.items() if lid == league_id),
        None,
    )

# En predict_proba — paso 2:
league_name = _resolve_league_name(league_id)
if league_name and league_name in self.models:
    model = self.models[league_name]
    if model.fitted:
        return model.predict_proba(home_team, away_team)
```

El mismo fix se aplica en `FootbotEnsemble.fit` al pre-calcular probs DC para
la optimización de blend weights.

**Resultado post-fix:**
```
Cúcuta vs Boyacá:   home=64.6% (antes 47.2%)
Tolima vs Jaguares: home=81.3% (antes 47.2%)
Cali vs Pereira:    home=49.1% (antes 47.2%)
```

---

## Sistema de confianza (3 niveles — v6)

### Umbrales en `config/settings.py`
```python
# Alta
UMBRAL_EDGE_ALTA   = 8.0
UMBRAL_PROB_ALTA   = 0.62
MIN_PARTIDOS_ALTA  = 30

# Media
UMBRAL_EDGE_MEDIA  = 4.0
UMBRAL_PROB_MEDIA  = 0.55
MIN_PARTIDOS_MEDIA = 15

# Baja
UMBRAL_EDGE_BAJA   = 2.0
UMBRAL_PROB_BAJA   = 0.52
MIN_PARTIDOS_BAJA  = 6

# Nacionales (menos partidos disponibles)
MIN_PARTIDOS_ALTA_NACIONAL  = 15
MIN_PARTIDOS_MEDIA_NACIONAL = 8

# ESPN sin cuotas reales (más conservador)
UMBRAL_EDGE_ALTA_ESPN  = 10.0
UMBRAL_EDGE_MEDIA_ESPN =  6.0
```

### Restricciones del nivel BAJA
1. Solo con cuotas reales (`espn_live`, `exact_match`, `contextual_avg`, `fd_historical`)
2. Solo mercados estándar (`_MERCADOS_NIVEL_BAJA`): 1X2, BTTS, Over/Under 1.5–3.5, Doble oportunidad, AH ±0.5
3. Kelly al 50% de la fracción normal (`kelly_fraction(..., confidence="baja")`)
4. Mensaje Telegram incluye `"señal débil — stake reducido"`

### Mercados bloqueados con model_implied
```python
_MERCADOS_BLOQUEADOS_MODEL_IMPLIED = {
    "home_win", "draw", "away_win",
    "over25", "under25",
    "double_1x", "double_x2", "double_12",
}
```

---

## Cuotas ESPN — schema confirmado (Core API /odds)

```
ESPN Core API: /v2/sports/soccer/leagues/{slug}/events/{id}/competitions/{id}/odds

Campos relevantes por item:
  provider.id / name / priority  → Bet365 priority=0, DraftKings priority=1
  homeTeamOdds.moneyLine         → 1X2 local (americano)
  awayTeamOdds.moneyLine         → 1X2 visitante (americano)
  drawOdds.moneyLine             → empate (solo fútbol)
  overUnder                      → línea de goles totales (float: 2.5)
  overOdds / underOdds           → cuotas Over/Under (americano)
  spread                         → handicap de línea (ej: -0.5 para local)
  homeTeamOdds.spreadOdds        → cuota spread local
  awayTeamOdds.spreadOdds        → cuota spread visitante
  open.over.value / open.spread.home.line → apertura (movimiento de línea)
```

**Notas confirmadas:**
- Liga BetPlay (col.1): solo 1 provider (DraftKings, priority=1)
- Premier League (eng.1): Bet365 (priority=0) tiene `moneyLine: null` → el código lo salta y usa DraftKings correctamente
- El sort ascendente por priority es correcto — no necesita cambios
- `_to_decimal` normaliza coma decimal antes de `float()` (FIX A3 en espn_collector.py)

---

## Mercados soportados (`_04_value_detector.py` + `_05_result_updater.py`)

| Categoría | Mercados | evaluate_bet |
|-----------|---------|:---:|
| 1X2 | home_win, draw, away_win | ✅ |
| Doble oportunidad | double_1x, double_x2, double_12 | ✅ |
| Goles totales | over05–over45, under05–under45 | ✅ |
| BTTS | btts_si, btts_no | ✅ |
| Por equipo | home_over05/15, home_under05/15, away_over05/15, away_under05/15 | ✅ |
| Goles exactos | exact_0, exact_1, exact_2, exact_3, exact_4plus | ✅ |
| Combinadas | home_and_btts, draw_and_btts, away_and_btts | ✅ |
| Asian Handicap | ah_home/away ±0.5, ah_home/away -1 | ✅ |

Todos los mercados tienen evaluador completo en `evaluate_bet`. Mercados desconocidos
retornan `False` sin lanzar excepción.

---

## ELO propio ESPN — `compute_elo_espn` (`_01_data_collector.py` v4)

Solución al problema de ClubElo que no cubre equipos LATAM (devuelve 0.0).

**Algoritmo:**
- Motor ELO estándar con factor K dinámico: `k_new=40` para equipos con < 30 partidos, `k_base=20` para el resto
- Margin factor logarítmico: `log2(|gd| + 1.5)` — penaliza goleadas sin explotar
- Ventaja de local configurable: `home_advantage=65.0` puntos (estándar fútbol)
- Rating inicial: `initial_elo=1500.0`
- Iteración cronológica sobre todo el histórico ESPN

**Parámetros:**
```python
compute_elo_espn(
    historical=None,        # carga automáticamente si es None
    k_base=20.0,
    k_new=40.0,
    new_team_threshold=30,
    initial_elo=1500.0,
    home_advantage=65.0,
    save=True,              # guarda en data/raw/elo_espn.csv
)
```

**Output:** DataFrame con columnas `Club`, `Elo`, `n_partidos`, `league_name`
compatible con `load_elo()` de `_02_feature_builder.py`.

**Archivo generado:** `data/raw/elo_espn.csv`

> **Pendiente:** `_02_feature_builder.load_elo()` debe leer `elo_espn.csv`
> además de `elo_ratings.csv` (ClubElo) para que el feature `elo_diff` sea
> funcional en ligas LATAM.

---

## `_05_result_updater.py` v2 — cambios principales

### evaluate_bet — cobertura completa
Todos los mercados del sistema tienen evaluador. Implementación con dict de lambdas:

```python
evaluators = {
    # 1X2
    "home_win":  lambda: home_goals > away_goals,
    "draw":      lambda: home_goals == away_goals,
    "away_win":  lambda: home_goals < away_goals,
    # Doble oportunidad
    "double_1x": lambda: home_goals >= away_goals,
    "double_x2": lambda: away_goals >= home_goals,
    "double_12": lambda: home_goals != away_goals,
    # BTTS
    "btts_si":   lambda: btts,
    "btts_no":   lambda: not btts,
    # Over/Under 0.5–4.5 (total = home_goals + away_goals)
    "over05": ..., "under05": ...,
    "over15": ..., "under15": ...,
    "over25": ..., "under25": ...,
    "over35": ..., "under35": ...,
    "over45": ..., "under45": ...,
    # Por equipo
    "home_over05": ..., "home_under05": ...,
    "home_over15": ..., "home_under15": ...,
    "away_over05": ..., "away_under05": ...,
    "away_over15": ..., "away_under15": ...,
    # Goles exactos
    "exact_0": ..., "exact_1": ..., "exact_2": ...,
    "exact_3": ..., "exact_4plus": lambda: total >= 4,
    # Combinadas
    "home_and_btts": lambda: (home_goals > away_goals) and btts,
    "draw_and_btts": lambda: (home_goals == away_goals) and btts,
    "away_and_btts": lambda: (home_goals < away_goals) and btts,
    # Asian Handicap
    "ah_home_minus05": ..., "ah_away_minus05": ...,
    "ah_home_plus05":  ..., "ah_away_plus05":  ...,
    "ah_home_minus1":  lambda: (home_goals - away_goals) >= 2,
    "ah_away_minus1":  lambda: (home_goals - away_goals) < 2,
}
```

### compute_model_stats — desglose completo
Calcula ROI y tasa de acierto total + por nivel (alta/media/baja) + por mercado.
Guarda snapshot en tabla `estadisticas_modelo`.

```python
stats = {
    "total": N, "ganadas": N, "tasa_pct": X, "roi_pct": X,
    # Por nivel
    "n_alta": N, "ganadas_alta": N, "tasa_alta_pct": X, "roi_alta_pct": X,
    "n_media": N, ...,
    "n_baja": N, ...,
    # Por mercado (dinámico — una entrada por cada mercado con datos)
    "n_home_win": N, "tasa_home_win_pct": X, "roi_home_win_pct": X,
    "n_over25": N, ...
}
```

### Fuentes de resultados (orden de prioridad)
1. `football-data.org` — ligas EU (fuente principal)
2. `ESPN API` — LATAM + Champions (sin key, fallback automático)
3. `API-Football` — fallback final (requiere key)

### DDL v2 — cambios respecto a v1
- CHECK `confianza IN ('alta','media','baja')` (v1 solo tenía `alta` y `media`)
- Columnas `home_goals` / `away_goals` en tabla `predicciones`
- Índice adicional `idx_pred_mercado`
- Vista `roi_por_mercado` — desglose por mercado y nivel
- Vista `resumen_diario` — estadísticas por fecha

---

## Columnas ESPN en features_df (post-enrich)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| espn_odds_available | bool | True si hay cuotas válidas |
| espn_odds_home | float | Cuota decimal local 1X2 |
| espn_odds_draw | float | Cuota decimal empate |
| espn_odds_away | float | Cuota decimal visitante |
| espn_odds_provider | str | Nombre del provider |
| espn_total_line | float | Línea O/U (ej: 2.5) |
| espn_over_odds | float | Cuota decimal Over |
| espn_under_odds | float | Cuota decimal Under |
| espn_spread_line | float | Handicap de línea (ej: -0.5) |
| espn_spread_home_odds | float | Cuota spread local |
| espn_spread_away_odds | float | Cuota spread visitante |
| espn_open_total_line | float | Línea de apertura |
| espn_open_spread_home | float | Spread de apertura |

---

## Formato del mensaje Telegram (v6)

```
⚽ FOOTBOT · Sábado 28 Mar
_N apuestas_

🟢 ALTA CONFIANZA
────────────────────────────
*Home vs Away*
🏆 Liga BetPlay
📈 Esperados: 2.5 – 0.5 (total 3.0)
_🏆 Resultado_
  📌 *Victoria Home*
     Prob 81.3% · Cuota 1.339 · Edge +8.93%
     Kelly 1.2% | forma local 2.2pts/PJ

🟡 MEDIA CONFIANZA
────────────────────────────
...

🔵 BAJA CONFIANZA
_Edge reducido — señales exploratorias_
────────────────────────────
  📌 *Over 2.5 goles*
     Prob 57.3% · Cuota 1.657 · Edge +2.1%
     Kelly 0.3% | señal débil — stake reducido

────────────────────────────
📊 Rendimiento (N apuestas): Acierto X% · ROI +Y%
  🟢 Alta: X% · ROI +Y%
  🟡 Media: X% · ROI +Y%
  🔵 Baja: X% · ROI +Y%

⚠️ Modelo estadístico experimental. Apuesta con responsabilidad.
```

**Grupos de mercados en el mensaje:**
`resultado` | `goles` | `btts` | `por_equipo` | `exactos` | `combinadas` | `handicap`

---

## Standings ESPN — URL correcta

```powershell
# INCORRECTO — devuelve {} vacío:
Invoke-RestMethod "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/standings"

# CORRECTO — devuelve tabla completa:
Invoke-RestMethod "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings"
```
El código usa `ESPN_SITE_V2B` que apunta a `/apis/v2/` — correcto.

---

## Slugs ESPN confirmados

### Ligas LATAM activas (LIGAS_ESPN_ACTIVAS)
| Slug | Liga | league_id |
|------|------|-----------|
| col.1 | Liga BetPlay Colombia | 501 |
| arg.1 | Liga Profesional Argentina | 502 |
| bra.1 | Brasileirão Serie A | 503 |
| conmebol.libertadores | Copa Libertadores | 511 |
| conmebol.sudamericana | Copa Sudamericana | 512 |
| uefa.champions | Champions League | 514 |
| mex.1 | Liga MX | 518 |

### Ligas EU en ESPN (slugs correctos, no usadas como activas aún)
| Slug | Liga |
|------|------|
| eng.1 | Premier League |
| esp.1 | La Liga |
| ger.1 | Bundesliga |
| ita.1 | Serie A |
| fra.1 | Ligue 1 |
| ned.1 | Eredivisie |
| por.1 | Primeira Liga |

### Competiciones nacionales (COMPETICIONES_NACIONALES_ESPN)
| Slug | Competición | league_id |
|------|-------------|-----------|
| fifa.worldq.conmebol | Eliminatorias CONMEBOL | 361 |
| conmebol.america | Copa América | 271 |
| fifa.world | Copa del Mundo | 1 |
| uefa.nations | UEFA Nations League | 5 |

### Ligas a agregar cuando haya más datos
`ksa.1` (Saudi Pro League), `tur.1` (Super Lig), `sco.1` (Scottish Prem)

---

## Problemas conocidos y soluciones

### Open-Meteo 400 Bad Request ✅ RESUELTO
**Causa:** Para partidos de hoy `days_ahead = 0`, Open-Meteo rechaza `start_date`
igual a `end_date` en timezones UTC-5 (Colombia). Partidos de mañana también afectados.
**Fix aplicado en `utils.py` v4:**
- `days_ahead <= 1`: usa `forecast_days=N` **sin** `start_date/end_date`; selecciona el elemento `[days_ahead]` del array de respuesta.
- `days_ahead >= 2`: sigue usando `start_date/end_date` como antes.
- `days_ahead > 16`: devuelve valores neutros sin llamar a la API (sin cambios).

### Supabase — error de versión (resuelto)
```
'typing.Union' object has no attribute '__module__'
```
✅ **Resuelto** — `supabase==2.3.0` es la versión compatible confirmada.

### Supabase DDL — nivel baja no incluido (resuelto)
**Causa:** El CHECK de la tabla `predicciones` solo permitía `('alta','media')`.
Con la v6 se emiten apuestas de nivel `'baja'` que Supabase rechazaba (error 23514).
**Fix aplicado en DDL v2:**
```sql
-- Anterior (incorrecto):
CHECK (confianza IN ('alta','media'))
-- DDL v2 (correcto — ejecutado en Supabase SQL Editor):
CHECK (confianza IN ('alta','media','baja'))
```
✅ **Resuelto** — insert nivel `'baja'` verificado en `test_conexiones.py` (id=2, registro eliminado).

### Cúcuta Deportivo — pocos partidos históricos
Solo 6 partidos como local. Con `MIN_PARTIDOS_BAJA=6` ya genera alertas de nivel baja.

### btts_si prob=0% (resuelto en v6)
Causado por el bug de league_id lookup — corregido junto con el fix principal.

### `_parse_fixture` — score como string vs dict (FIX A1)
`/scoreboard` devuelve `score` como string plano; `/schedule` lo devuelve como
dict `{"value": 3.0, "displayValue": "3"}`. En entornos ES/COL `value` tiene
coma decimal ("3,0") que hace fallar `float()`. `_parse_score()` usa
`displayValue` como fuente primaria — ya corregido.

### `_safe_int` — coma decimal en standings (FIX B1)
ESPN standings devuelve stats como `"27,0"` en entornos ES/COL.
`int("27,0")` explota. `_safe_int()` convierte con `str(val).replace(",", ".")`.

---

## Tests unitarios

```
tests/
├── conftest.py               # sample_historical (240p), sample_training_df,
│                             # sample_fixture_row, sample_predictions, sample_bets_df
├── test_data_collector.py    # _make_session, get_fixtures_today, download_elo_ratings
├── test_espn_collector.py    # _parse_score (FIX A1), _to_decimal (FIX A3),
│                             # _safe_int (FIX B1), _norm_status, ESPNClient
├── test_feature_builder.py   # _clean_name, TeamNameResolver, exponential_weight,
│                             # compute_team_stats, compute_h2h, get_elo_diff
├── test_model_engine.py      # _resolve_league_name (FIX v6), DixonColesModel,
│                             # DixonColesEnsemble, blend_predictions,
│                             # FootbotEnsemble (@slow)
├── test_result_updater.py    # v2: evaluate_bet (parametrizado, TODOS los mercados ~40),
│                             # compute_model_stats (3 niveles + por mercado),
│                             # update_results_in_supabase (mock + fuzzy match),
│                             # save_predictions_to_supabase (mock),
│                             # DDL v2 (baja, resumen_diario, home/away_goals, idx_mercado)
├── test_telegram_sender.py   # format_message (3 niveles), format_date_es
└── test_value_detector.py    # _poisson_matrix, compute_all_market_probs,
                              # calculate_edge, kelly_fraction, classify_confidence,
                              # build_odds_dict, analyze_fixture
```

**Cobertura de `test_result_updater.py` v2:**
- 90+ casos parametrizados cubriendo los ~40 mercados de `evaluate_bet`
- Tests de consistencia: `0-0`, partido con muchos goles `4-2`, `1-1` combinadas
- Simetría Asian Handicap: `ah_minus05` son complementarios en partidos sin empate; `ah_plus05` ambos True en empate
- `compute_model_stats`: vacío, None, totales, desglose por 3 niveles, ROI ±, por mercado
- `update_results_in_supabase`: ganadora, perdedora, sin resultado, fuzzy match por substring
- DDL: todos los campos, vistas y constraints requeridos

**Ejecución:**
```powershell
pytest tests/                                              # todos los tests rápidos
pytest tests/ -m slow                                      # incluye entrenamiento de modelos
pytest tests/ --cov=src                                    # con cobertura
pytest tests/test_result_updater.py -v -k "evaluate_bet"  # solo mercados
```

**Estado tests:** todos pasan ✅ (el bug `test_supabase_ddl_includes_baja` está resuelto en DDL v2)

---

## Comandos útiles de diagnóstico

### Verificar histórico por liga
```powershell
python -c "
from src._01_data_collector import load_espn_historical
hist = load_espn_historical()
print(hist.groupby('league_name').size().to_string())
"
```

### Calcular ELO propio desde histórico ESPN
```powershell
python -c "
from src._01_data_collector import compute_elo_espn
df = compute_elo_espn()
print(df.head(20).to_string())
print(f'Total equipos con ELO: {len(df)}')
"
```

### Verificar DC lookup por liga
```python
# diag3.py
import pandas as pd
from src._02_feature_builder import load_historical_results, normalize_team_name
from src._03_model_engine import load_models, predict_match

features = pd.read_csv('data/processed/features_YYYY-MM-DD.csv')
dc, ensemble = load_models()

for _, row in features.iterrows():
    preds = predict_match(row['home_team'], row['away_team'], row.to_dict(), dc, ensemble)
    print(row['home_team'] + ' vs ' + row['away_team'])
    print('  league_id:        ' + str(row.get('league_id')))
    print('  dc_exp_home:      ' + str(preds.get('dc_exp_home_goals')))
    print('  prob_home_win:    ' + str(preds.get('prob_home_win')))
    print('  ligas entrenadas: ' + str(list(dc.models.keys())))
    print()
```

### Verificar edges y probabilidades
```python
# diag2.py
import pandas as pd
from src._03_model_engine import load_models, predict_match
from src._04_value_detector import compute_all_market_probs, build_odds_dict, calculate_edge
from src._02_feature_builder import load_historical_results

features  = pd.read_csv('data/processed/features_YYYY-MM-DD.csv')
historical = load_historical_results()
dc, ensemble = load_models()

for _, row in features.iterrows():
    preds = predict_match(row['home_team'], row['away_team'], row.to_dict(), dc, ensemble)
    mu  = preds.get('dc_exp_home_goals', 1.4)
    lam = preds.get('dc_exp_away_goals', 1.1)
    market_probs = compute_all_market_probs(mu, lam)
    odds, is_real, method = build_odds_dict(row['home_team'], row['away_team'],
                                             historical, row, market_probs)
    print(row['home_team'] + ' vs ' + row['away_team'] + ' [' + method + ']')
    for market in ['home_win','draw','away_win','btts_si','over25','under25']:
        prob = market_probs.get('prob_' + market, 0)
        odd  = odds.get(market, 0)
        edge = calculate_edge(prob, odd)
        print(f'  {market}: prob={round(prob*100,1)}% | cuota={odd} | edge={edge}%')
    print()
```

### Verificar ROI por mercado (Supabase)
```python
# diag_roi.py
from src._05_result_updater import init_supabase, compute_model_stats
sb = init_supabase()
stats = compute_model_stats(sb)
for k, v in sorted(stats.items()):
    if 'roi' in k or 'tasa' in k:
        print(f'  {k}: {v}')
```

### Eliminar modelos y forzar reentrenamiento
```powershell
python -c "
import os, glob
for f in glob.glob('models/*.pkl'):
    os.remove(f)
    print('Eliminado: ' + f)
"
python scheduler.py
```

---

## Variables de entorno (.env)

```
TELEGRAM_TOKEN=<configurado>             # ✅ real
TELEGRAM_CHAT_ID=<configurado>           # ✅ real
SUPABASE_URL=<configurado>               # ✅ real
SUPABASE_KEY=<configurado>               # ✅ real (sb_secret_tS...)
FOOTBALL_DATA_ORG_KEY=                   # opcional — ligas EU (10 req/min free tier)
API_FOOTBALL_KEY=                        # opcional — fallback resultados
ESPN_ONLY=true                           # activo — saltea fd.co.uk y StatsBomb
```

---

## URLs base de APIs

```python
ESPN_SITE_V2  = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_SITE_V2B = "https://site.api.espn.com/apis/v2/sports/soccer"       # standings
ESPN_CORE_V2  = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
CLUBELO_URL   = "http://api.clubelo.com"
FOOTBALL_DATA_ORG_URL = "https://api.football-data.org/v4"
```

---

## Versiones de archivos

| Archivo | Versión | Cambio principal |
|---------|---------|-----------------|
| src/_03_model_engine.py | v7 | `scale_pos_weight` dinámico por mercado en `FootbotEnsemble.fit`; `_get_xgb` acepta parámetro `scale_pos_weight` |
| src/_02_feature_builder.py | v6 | `_precompute_rolling_cache`: pre-computa rolling stats por equipo O(n) antes del loop; `build_training_dataset` pasa de ~90s a ~5s con 4226 partidos; header actualizado |
| src/_03_model_engine.py | v6 | Fix league_id lookup ESPN en predict_proba y FootbotEnsemble.fit |
| config/settings.py | v6 | Umbrales nivel BAJA, ESPN_ONLY flag |
| src/_04_value_detector.py | v6 | Nivel baja en classify_confidence, kelly reducido al 50% |
| src/telegram_sender.py | v6 | Sección 🔵 BAJA CONFIANZA en format_message |
| src/espn_collector.py | v3 | FIX A1 _parse_score, FIX A2 providers, FIX A3 _to_decimal, FIX B1 _safe_int, FIX D1/D2 seasons |
| src/supabase_client.py | v1 | Cliente Supabase — insert/select/delete predicciones |
| supabase_ddl_v2.sql | v2 | CHECK corregido: `'baja'` incluido; columnas home/away_goals; idx_mercado; vistas roi_por_mercado y resumen_diario |
| src/_01_data_collector.py | v4 | Retry backoff WinError 10054; fusión fd.org + ESPN; **compute_elo_espn** (ELO propio LATAM con K dinámico y margin factor log) |
| src/_05_result_updater.py | v2 | evaluate_bet completo (~40 mercados); compute_model_stats con 3 niveles y por mercado; get_results_espn como fallback; DDL v2 |
| src/utils.py | v4 | Fix Open-Meteo 400 Bad Request para hoy/mañana: `forecast_days` sin `start/end_date` cuando `days_ahead <= 1`; indexación correcta del array por `days_ahead` |
| src/nacional_features.py | v4 | Eliminado import roto de nacional_collector |
| tests/test_result_updater.py | v2 | 90+ casos evaluate_bet; tests compute_model_stats 3 niveles; mocks update/save Supabase; DDL v2 completo |

---

## Notas para próximas sesiones

1. ~~**Supabase:** `pip install supabase==1.2.0` y corregir DDL (`'baja'` en CHECK)~~ ✅ **RESUELTO** — `supabase==2.3.0`, DDL v2 ejecutado, 4/4 tests OK
2. ~~**Telegram:** Configurar token real y verificar envío~~ ✅ **RESUELTO** — token y chat_id configurados, mensaje de prueba enviado
3. ~~**`evaluate_bet`:** Añadir evaluadores para over15, under15, over45, by-team markets, exactos, combinadas y AH~~ ✅ **RESUELTO** — `_05_result_updater.py` v2 cubre todos los mercados (~40 evaluadores)
4. ~~**`compute_model_stats` sin desglose por nivel ni mercado**~~ ✅ **RESUELTO** — desglose completo alta/media/baja + por mercado en v2
5. ~~**ELO LATAM = 0.0 (ClubElo no cubre LATAM)**~~ ✅ **RESUELTO** — `load_elo()` v5 fusiona ClubElo + `elo_espn.csv`; prioridad ESPN en equipos duplicados; `elo_diff` funcional en todas las ligas
6. ~~**Open-Meteo:** Usar `forecast_days=1` sin `start_date/end_date` para partidos de hoy~~ ✅ **RESUELTO** — `utils.py` v4 implementado
7. ~~**`_02_feature_builder.load_elo()`:** Leer `elo_espn.csv` además de `elo_ratings.csv`~~ ✅ **RESUELTO** — v5
8. ~~**`build_training_dataset` O(n²):** ~90s con 4226 partidos. Pre-computar rolling stats por equipo antes de agregar ligas EU~~ ✅ **RESUELTO** — `_02_feature_builder.py` v6: `_precompute_rolling_cache` implementado
9. **`_parse_fixture`:** Reemplazar `int(home_score)` por `_parse_score(home_score)` para consistencia con el endpoint `/schedule`
10. **Copa Libertadores / Sudamericana:** < 200 partidos → sin DC propio. El modelo global tampoco converge (> 80 equipos). Pendiente solución arquitectural (DC cross-liga o umbral dinámico)
11. **Tests @slow:** `FootbotEnsemble.fit` no corre en CI normal — marcar y ejecutar explícitamente con `pytest -m slow`
12. ~~**Distribución de clases XGBoost:**~~ ✅ **RESUELTO** — `scale_pos_weight` dinámico en v7; reentrenamiento ejecutado 2026-03-31 con 4236 partidos
13. **Métricas XGBoost post-fix:** correr evaluación formal post-reentrenamiento para registrar ROI real de `draw` y `away_win`; las métricas en la tabla son pre-fix y no reflejan el modelo actual
14. **Bug `btts_si` en `diag2.py`:** cambiar `'btts_si'` → `'btts'` en el loop de mercados; la clave correcta en `market_probs` es `prob_btts`, no `prob_btts_si`. Solo afecta el script de diagnóstico, producción usa `get_model_prob_for_market` internamente