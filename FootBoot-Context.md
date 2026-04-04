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
- Clima: Open-Meteo (sin key) — `utils.py` v5 ✅ (retry backoff exponencial)
- Base de datos: Supabase (✅ conectado — `supabase==2.3.0`, tablas creadas con DDL v3, **`service_role key`** en `.env`)
- Notificaciones: Telegram Bot API (✅ conectado — token y chat_id configurados)

---

## Estructura del proyecto

```
footbot/
├── config/
│   └── settings.py              # Constantes globales, umbrales, API keys — v9: SLUGS_SIN_BPI añadido
├── src/
│   ├── _01_data_collector.py    # Descarga fixtures, histórico, ELO (v6 — grupos LATAM/EU/copas, nuevos slugs)
│   ├── _02_feature_builder.py   # Features por partido (forma, H2H, xG, clima) — v8: BPI condicional por slug, flag bpi_available
│   ├── _03_model_engine.py      # Dixon-Coles + XGBoost + blend weights — v10: FEATURE_COLS usa bpi_available en vez de probs BPI crudas
│   ├── _04_value_detector.py    # Edge%, Kelly, clasificación de confianza (v7 — fix btts_si prob key)
│   ├── _05_result_updater.py    # Cierre de predicciones con resultado real (v2)
│   ├── espn_collector.py        # Cliente ESPN unificado (v5 — injuries, BPI, top scorers, enrich helpers)
│   ├── nacional_features.py     # Features para selecciones nacionales
│   ├── telegram_sender.py       # Formateo y envío de reportes (v6)
│   ├── supabase_client.py       # Cliente Supabase (v3 — fix fecha_partido, cuota_referencia, service_role key)
│   └── utils.py                 # Clima, rate limiting, coordenadas estadios — v5: retry backoff Open-Meteo
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
- ESPN API devuelve fixtures y cuotas para todas las ligas LATAM activas
- Cuotas ESPN en tiempo real: 12/12 partidos con DraftKings (verificado 2026-04-02)
- Dixon-Coles entrenado por liga — **15 ligas** incluyendo Copa Libertadores y Copa Sudamericana (post-backfill)
- Fuzzy matching de nombres (rapidfuzz) resuelve correctamente equipos colombianos
- **Histórico ESPN: 14.314 partidos de 14 ligas** (2022-01-07 → 2026-04-01) — backfill 2022–2023 ejecutado ✅
- ELO fusionado: **927 equipos únicos** (630 ClubElo + 408 ESPN)
- Blend weights optimizados por mercado (dc_weight=0.70 para todos)
- Fix league_id lookup: ligas ESPN resuelven correctamente (v6)
- Sistema de 3 niveles de confianza: alta, media, baja (v6)
- Suite de tests unitarios completa (pytest, 7 módulos)
- DDL Supabase generado y ejecutado (tablas con DDL v3 — tabla `resultados` añadida, columnas `league_id`/`odds_source`/`odds_provider`/`dc_exp_home`/`dc_exp_away` en `predicciones`, vista `predicciones_abiertas`)
- Supabase conectado: `supabase==2.3.0`, insert/delete verificados — **`service_role key`** en `.env` (bypasea RLS, correcto para backend server-side sin usuarios)
- Telegram conectado: token + chat_id reales, mensaje de prueba enviado OK
- Test de integración `test_conexiones.py`: 4/4 tests OK
- **`evaluate_bet` completo** ✅ — cubre todos los mercados: 1X2, doble oportunidad, over/under 0.5–4.5, BTTS, por equipo, goles exactos, combinadas, Asian Handicap ±0.5 y -1
- **`compute_model_stats` con desglose por nivel y mercado** ✅ — ROI tracking funcional para alta/media/baja
- **`compute_elo_espn`** ✅ — ELO propio calculado desde histórico ESPN con factor K dinámico y margin factor logarítmico; guarda `data/raw/elo_espn.csv`
- **`load_elo()` fusionada** ✅ — `_02_feature_builder.load_elo()` v5 lee y fusiona `elo_ratings.csv` (ClubElo, EU) + `elo_espn.csv` (ESPN, LATAM); prioriza ESPN en caso de equipo duplicado; `elo_diff` ya no es 0.0 para equipos LATAM
- **`scale_pos_weight` dinámico en XGBoost** ✅ — `_03_model_engine.py` v7 calcula `neg/pos` por mercado antes de entrenar cada clasificador; `draw` (~2.7×) y `away_win` (~2.6×) dejan de predecir clase mayoritaria
- **Reentrenamiento post-backfill ejecutado** ✅ — modelos reentrenados el 2026-04-02 con dataset completo (14.314 partidos, 14 ligas, 2022–2026)
- **`detect_all_value_bets`** ✅ — nombre real de la función de producción en `_04_value_detector.py` (no `detect_value_bets`); funciones públicas del módulo: `analyze_fixture`, `build_explanation`, `build_odds_dict`, `calculate_edge`, `classify_confidence`, `compute_all_market_probs`, `detect_all_value_bets`, `get_current_season_odds`, `get_model_prob_for_market`, `kelly_fraction`, `normalize_team_name`, `summarize_bets`
- **Claves BTTS en `market_probs`** ✅ — confirmadas: `prob_btts`, `prob_btts_no`, `prob_home_and_btts`, `prob_draw_and_btts`, `prob_away_and_btts` (NO existe `prob_btts_si`). **Fix aplicado en `_04_value_detector.py` v7**: `build_odds_dict` captura `_btts_prob = market_probs.get("prob_btts", 0)` y asigna `model_odds["btts_si"]` explícitamente.
- **`get_results_espn`** ✅ — fuente de resultados ESPN integrada en `_05_result_updater.py` como fallback a fd.org
- **DDL v3 completo** ✅ — tabla `resultados` añadida; columnas `league_id`, `odds_source`, `odds_provider`, `dc_exp_home`, `dc_exp_away` en `predicciones`; vista `predicciones_abiertas`; índice compuesto `idx_pred_fecha_ganada WHERE ganada IS NULL`; RLS policies documentadas (comentadas — se usa `service_role key`)
- **`MARKET_THRESHOLDS` en `_03_model_engine.py` v8** ✅ — threshold `draw=0.33` activo; `FootbotEnsemble.predict` emite `xgb_signal_{market}` (bool) consultable desde `_04_value_detector`; `_compute_validation_roi` usa threshold correcto en logs de entrenamiento
- **`_parse_fixture` defensivo en `espn_collector.py` v4** ✅ — `home_goals`/`away_goals`/`home_ft`/`away_ft` usan `_parse_score()` en lugar de `int()` directo; evita crashes con score como dict o coma decimal
- **`ESPN_HISTORICAL_SEASONS` en `config/settings.py` v7** ✅ — `list(range(2022, date.today().year + 1))`; `download_espn_historical` lo usa como default; se extiende solo cada año
- **`settings.py` v9: `SLUGS_SIN_BPI`** ✅ — set con todos los slugs LATAM y CONCACAF donde ESPN BPI no devuelve datos; usado por `_02_feature_builder.py` para omitir llamadas innecesarias al API
- **`settings.py` v10: Time decay configurable** ✅ — `DC_TIME_DECAY_XI` (default 0.003), `XGB_TIME_DECAY_LAMBDA` (default 0.002) y `TIME_DECAY_REFERENCE_DATE` configurables desde `.env`. Semivida DC ~231 días con xi=0.003; semivida XGB ~347 días con λ=0.002.
- **`_03_model_engine.py` v11: Time decay + standings features** ✅ — DC aplica `w = exp(−ξ × días_atrás)` a la log-likelihood. XGBoost recibe `sample_weight = exp(−λ × días_atrás)`. FEATURE_COLS ampliado con 18 features de standings context. **Requiere reentrenamiento (#29).**
- **`espn_collector.py` v6: Standings context** ✅ — `get_standings_context` calcula features de motivación por equipo desde `/standings`. `enrich_fixtures_with_standings` enriquece fixtures con relegation_threat, title_race, motivation_score, etc. Fuzzy match con rapidfuzz. Una sola llamada por liga por día.
- **`_02_feature_builder.py` v9: Pipeline de motivación** ✅ — `build_features_for_fixtures` incluye paso 4 de standings enrichment. `build_training_dataset` inicializa columnas en 0 (retrocompatible).
- **`espn_collector.py` v5** ✅ — 4 funciones nuevas: `get_team_injuries()`, `enrich_fixtures_with_injuries()`, `get_match_summary_bpi()`, `enrich_fixtures_with_bpi()`, `get_league_top_scorers()`
- **`_01_data_collector.py` v6** ✅ — `download_espn_historical()` con lógica de grupos: LATAM siempre, EU solo si `ESPN_ONLY=true`, copas UEFA siempre; importa `SLUGS_*` desde `settings`
- **`_02_feature_builder.py` v8** ✅ — enriquecimiento BPI condicional por slug (omite LATAM via `SLUGS_SIN_BPI`); `bpi_available` como feature binaria `0/1` en `feature_row`; `build_training_dataset()` inicializa `bpi_available=0` para el histórico
- **`_03_model_engine.py` v10** ✅ — `FEATURE_COLS` reemplaza `espn_bpi_home_prob`/`espn_bpi_away_prob` por `bpi_available` (flag binario); probs BPI crudas siguen en DataFrame para diagnóstico pero no entran al modelo
- **ESPN BPI en LATAM: RESUELTO** ✅ — `espn_bpi_home_prob`/`away_prob` eliminadas de `FEATURE_COLS`; sustituidas por `bpi_available=0` para LATAM. Se ahorran llamadas API innecesarias via `SLUGS_SIN_BPI`
- **`utils.py` v5: retry Open-Meteo** ✅ — backoff exponencial (1s→2s→4s ± 0.3s jitter) ante `SSLError`/`ConnectionError`/`Timeout`; hasta 3 reintentos; errores no-retriables (400) fallan rápido; loguea `ERROR` al agotar intentos

### Pendiente ⚠️
- ~~**Open-Meteo:**~~ ✅ **RESUELTO** — `utils.py` v4: `forecast_days` sin `start/end_date` para `days_ahead <= 1`; índice correcto del array de respuesta
- ~~**Open-Meteo SSL error en Colombia:**~~ ✅ **RESUELTO** — `utils.py` v5: retry con backoff exponencial (3 intentos, 1s→2s→4s ± jitter). El `SSLEOFError` observado el 2026-04-02 era recuperable con reintento.
- **Football-Data.co.uk:** no descargado (ESPN_ONLY=true en .env)
- ~~**ELO LATAM en pipeline:**~~ ✅ **RESUELTO** — `load_elo()` v5 fusiona ClubElo + `elo_espn.csv`; `elo_diff` funcional en LATAM
- ~~**`build_training_dataset` O(n²):**~~ ✅ **RESUELTO** — `_02_feature_builder.py` v6: `_precompute_rolling_cache` pre-computa rolling stats O(n)
- ~~**Copa Libertadores / Sudamericana sin DC propio:**~~ ✅ **RESUELTO** — post-backfill: Copa Libertadores 472 partidos y Copa Sudamericana 298 partidos — ambas superan el mínimo de 200 y tienen DC propio entrenado.
- **Reentrenamiento post-BPI fix pendiente:** `_03_model_engine.py` v10 cambió `FEATURE_COLS` (elimina probs BPI, añade `bpi_available`). **Requiere reentrenar modelos** con `python scheduler.py` antes de correr en producción.
- **Backfill top scorers:** `get_league_top_scorers()` disponible — evaluar si el dato de goleadores por liga aporta como feature adicional (ej: `top_scorer_in_lineup`).
- ~~**Bug `btts_si` en producción:**~~ ✅ **RESUELTO** — `_04_value_detector.py` v7: `build_odds_dict` ya no busca `prob_btts_si` (inexistente); captura `prob_btts` explícitamente. **Pendiente menor:** `diag2.py` aún usa `'btts_si'` en el loop de mercados → fix: cambiar a `'btts'`.

---

## Datos históricos ESPN disponibles

| Liga | Slug | league_id | Grupo | Partidos |
|------|------|-----------|-------|---------|
| Liga BetPlay | col.1 | 501 | LATAM | 1.901 |
| Liga Profesional Argentina | arg.1 | 502 | LATAM | 1.798 |
| Brasileirão Serie A | bra.1 | 503 | LATAM | 1.497 |
| Liga MX | mex.1 | 518 | LATAM | 1.470 |
| Liga 1 Perú | per.1 | 523 | LATAM | 1.304 |
| Liga AUF Uruguay | uru.1 | 524 | LATAM | 1.208 |
| LigaPro Ecuador | ecu.1 | 522 | LATAM | 1.063 |
| Primera División Paraguay | — | — | LATAM | 1.037 |
| Primera División Chile | chi.1 | 521 | LATAM | 984 |
| Champions League | uefa.champions | 514 | COPAS_UEFA | 542 |
| Europa League | uefa.europa | 515 | COPAS_UEFA | 427 |
| Copa Libertadores | conmebol.libertadores | 511 | COPAS | 472 |
| Conference League | uefa.europa.conference | 516 | COPAS_UEFA | 313 |
| Copa Sudamericana | conmebol.sudamericana | 512 | COPAS | 298 |
| Premier League | eng.1 | 600 | EU | — |
| La Liga | esp.1 | 601 | EU | — |
| Bundesliga | ger.1 | 602 | EU | — |
| Serie A | ita.1 | 603 | EU | — |
| Ligue 1 | fra.1 | 604 | EU | — |
| Eredivisie | ned.1 | 605 | EU | — |
| Primeira Liga | por.1 | 606 | EU | — |
| Liga Venezuela | ven.1 | 525 | LATAM | — |

**Grupos de slugs (`settings.py` v8):**
- `SLUGS_LATAM_CLUBES` — col.1, arg.1, bra.1, mex.1, chi.1, ecu.1, per.1, uru.1, ven.1 *(descarga siempre)*
- `SLUGS_EU_ESPN` — eng.1, esp.1, ger.1, ita.1, fra.1, ned.1, por.1 *(solo si `ESPN_ONLY=true`)*
- `SLUGS_COPAS_UEFA` — conmebol.libertadores, conmebol.sudamericana, uefa.champions, uefa.europa, uefa.europa.conference *(descarga siempre)*

**Temporadas disponibles:** 2022, 2023, 2024, 2025, 2026 ✅ — backfill completo ejecutado el 2026-04-02. Rango: 2022-01-07 → 2026-04-01.

---

## Modelos entrenados (última versión: 20260402)

### Dixon-Coles por liga
Entrenado con L-BFGS-B (reemplazó SLSQP — más rápido con 100+ equipos).

| Liga | Partidos | Converge |
|------|---------|---------|
| Liga BetPlay | 1.901 | ✅ |
| Liga Profesional Argentina | 1.798 | ✅ |
| Brasileirão Serie A | 1.497 | ✅ |
| Liga MX | 1.470 | ✅ |
| Liga 1 Perú | 1.304 | ✅ |
| Liga AUF Uruguay | 1.208 | ✅ |
| LigaPro Ecuador | 1.063 | ✅ |
| Primera División Paraguay | 1.037 | ✅ |
| Primera División Chile | 984 | ✅ |
| Champions League | 542 | ✅ |
| Europa League | 427 | ✅ |
| Copa Libertadores | 472 | ✅ *(antes: ❌ < 200 mín — resuelto con backfill)* |
| Conference League | 313 | ✅ |
| Copa Sudamericana | 298 | ✅ *(antes: ❌ < 200 mín — resuelto con backfill)* |
| Global | — | ❌ (> 80 equipos máx) |

**Constantes clave (`_03_model_engine.py`):**
- `MIN_MATCHES_PER_LIGA = 200` — mínimo para entrenar DC por liga
- `MIN_MATCHES_GLOBAL_DC = 300` — mínimo para DC global
- `MAX_TEAMS_GLOBAL_DC = 80` — si hay más equipos, DC global se omite
- `DEFAULT_DC_WEIGHT = 0.35` — fallback si no hay optimización de blend

### XGBoost validation metrics — PRE-fix v6 (referencia histórica)
| Mercado | Accuracy | ROI flat | dc_weight |
|---------|---------|---------|----------|
| home_win | 55.4% | +17.4% | 0.700 |
| draw | 72.7% | +0.0% | 0.700 |
| away_win | 73.3% | +0.0% | 0.700 |
| btts | 53.8% | -3.6% | 0.675 |
| over25 | 56.3% | +7.1% | 0.700 |

### XGBoost validation metrics — POST-fix v7, pre-backfill (evaluación formal 2026-03-31)
Evaluado con `eval_v7.py` — split temporal 80/20, 848 partidos de validación (2025-11-03 → 2026-03-31), cuota model-implied.

| Mercado | Señales | Accuracy | ROI flat | vs pre-fix |
|---------|---------|---------|---------|-----------|
| home_win | 182 | 66.5% | +5.9% | ROI ↓ pero accuracy +11pp (menos pero mejores señales) |
| draw | 0 | — | — | ⚠️ Sin señales (threshold 0.50) |
| away_win | 44 | 70.5% | +15.9% | ✅ Pasó de 0 señales a ROI +15.9% |
| btts | 149 | 65.1% | +12.0% | ✅ Mejoró de -3.6% a +12.0% |
| over25 | 170 | 67.1% | +8.9% | ✅ Estable, leve mejora |
| double_1x | 400 | 79.2% | +8.1% | — |
| over15 | 481 | 75.5% | +6.5% | — |
| under35 | 422 | 78.9% | +3.6% | — |
| over05 | 496 | 90.3% | -0.1% | (cuota demasiado baja) |
| btts_no | 353 | 52.4% | -11.5% | (señal débil) |

---

## Evaluación formal post-backfill — `eval_v7.py` (2026-04-02)

**Dataset:** 14.314 partidos de 14 ligas (2022-01-07 → 2026-04-01).
**Metodología:** Split temporal 80/20 — 11.451 partidos entrenamiento (hasta 2025-07-20), 2.863 partidos validación (2025-07-20 → 2026-04-01). Cuota model-implied (1/prob). ROI flat (1 unidad por señal). Filtros: prob ≥ 50% | edge ≥ 0%. Partidos procesados: 2.856 | Errores: 7.

**Dixon-Coles entrenados (15 ligas):** global, Brasileirão Serie A, Champions League, Conference League, Copa Libertadores ✅, Copa Sudamericana ✅, Europa League, Liga 1 Perú, Liga AUF Uruguay, Liga BetPlay, Liga MX, Liga Profesional Argentina, LigaPro Ecuador, Primera División Chile, Primera División Paraguay.

### Resultados completos (2.863 partidos, prob ≥ 50%, edge ≥ 0%)

| Mercado | Señales | Accuracy | ROI flat | Prob media |
|---------|---------|---------|---------|-----------|
| home_win | 602 | 64.0% | +2.5% | 62.4% ◄ |
| draw | 1 | 100.0% | +0.9% | 99.1% ◄ (n=1, estadísticamente ruidoso) |
| away_win | 140 | 63.6% | +7.3% | 59.0% ◄ |
| btts | 508 | 60.0% | +6.3% | 56.4% ◄ |
| btts_no | 1.202 | 53.6% | -9.2% | 58.9% |
| over05 | 1.511 | 92.1% | +1.9% | 90.5% |
| over15 | 1.554 | 73.2% | +4.1% | 70.6% |
| over25 | 531 | 63.5% | +7.5% | 59.2% ◄ |
| over35 | 56 | 60.7% | +3.8% | 59.0% |
| over45 | 10 | 70.0% | +25.9% | 55.4% |
| under15 | 84 | 52.4% | -9.3% | 57.4% |
| under25 | 1.194 | 60.3% | -3.2% | 62.1% |
| under35 | 1.574 | 77.5% | +1.3% | 76.3% |
| double_1x | 1.505 | 74.6% | +1.7% | 73.2% |
| double_x2 | 1.041 | 62.7% | -3.4% | 64.4% |
| double_12 | 1.662 | 74.2% | -0.0% | 74.2% |

### Comparación vs pre-backfill (v7 con 848 partidos)

| Mercado | ROI pre-backfill | ROI post-backfill | Delta | Acc pre | Acc post |
|---------|-----------------|-------------------|-------|---------|---------|
| home_win | +5.9% | +2.5% | -3.4pp | 66.5% | 64.0% |
| draw | — | +0.9% (n=1) | — | — | 100% (ruido) |
| away_win | +15.9% | +7.3% | -8.6pp | 70.5% | 63.6% |
| btts | +12.0% | +6.3% | -5.7pp | 65.1% | 60.0% |
| over25 | +8.9% | +7.5% | -1.4pp | 67.1% | 63.5% |

> **Nota:** La dilución de ROI es esperada — más partidos de ligas menores LATAM (Paraguay, Uruguay, Ecuador, Perú, Chile) tienen menor predictibilidad y mayor varianza. El modelo es más robusto pero menos sobreajustado al conjunto original de 5 ligas.

### Análisis mercado `draw` post-backfill
- Con threshold=0.33 activo, el modelo generó **1 sola señal** en 2.863 partidos de validación (prob media 99.1% — caso atípico). Estadísticamente ruidoso.
- El threshold=0.33 fue calibrado con el dataset anterior (848 partidos). Con el dataset expandido la distribución de prob_draw cambió.
- **`eval_v7.py` v2 — `--sweep-draw`** ✅ — flag CLI añadido para barrido de thresholds 0.25→0.45 (paso 0.02) sobre los 2.863 partidos de validación. Pre-computa prob_draw en una sola pasada y muestra señales, accuracy, ROI flat, prob media y EV por señal por threshold. **Ejecutar para recalibrar antes del próximo reentrenamiento.**
```powershell
python eval_v7.py --sweep-draw
python eval_v7.py --sweep-draw --output sweep.csv
```

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

**Bug activo (2026-04-02):** `SSLEOFError` en Open-Meteo para 3 partidos colombianos — error de SSL/red, no de lógica. Los partidos continúan con `weather_precipitation=0` como fallback. Monitorear si persiste.

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

**Pesos con 14.314 partidos (post-backfill):**
| Mercado | ~% positivos | scale_pos_weight |
|---------|-------------|-----------------|
| home_win | ~45% | ~1.2 |
| draw | ~27% | ~2.7 |
| away_win | ~28% | ~2.6 |
| btts | ~50% | ~1.0 |
| over25 | ~50% | ~1.0 |

---

## Fix: Threshold por mercado en XGBoost (`_03_model_engine.py` v8)

**Problema:** XGBoost usa threshold=0.50 por defecto para decidir si "señala" un mercado. El clasificador `draw` genera probs reales (media 35.8%, max 49.2%) pero nunca supera 0.50 → 0 señales en producción a pesar de tener ROI potencial documentado.

**Análisis (eval_v7.py, 848 partidos pre-backfill):**
| Threshold | Señales | Accuracy | ROI flat |
|-----------|---------|---------|---------|
| 0.50 (anterior) | 0 | — | — |
| 0.33 (nuevo) | 50 | 44.0% | +24.5% |
| 0.35 | 25 | 40.0% | +7.0% |

El 44% de accuracy genera ROI positivo porque la cuota justa del empate (~1/0.33 ≈ 3.0) cubre el error.

**Fix (`_03_model_engine.py` v8):**
```python
MARKET_THRESHOLDS: dict[str, float] = {
    "home_win": 0.50,
    "draw":     0.33,   # ← cambio clave
    "away_win": 0.50,
    "btts":     0.50,
    "over25":   0.50,
}

threshold = MARKET_THRESHOLDS.get(market, 0.50) if market else 0.50
preds = (cal_probs > threshold).astype(int)

threshold = MARKET_THRESHOLDS.get(market, 0.50)
result[f"xgb_signal_{market}"] = bool(cal > threshold)
```

**Estado post-backfill (2026-04-02):** con el dataset expandido, el threshold=0.33 generó solo 1 señal en 2.863 partidos de validación. Recalibración del threshold pendiente sobre el nuevo dataset. Ver sección **Análisis mercado `draw` post-backfill**.

---

## Fix: `_parse_fixture` defensivo (`espn_collector.py` v4)

**Problema:** `_parse_fixture` usaba `int(home_score)` / `int(away_score)` directamente. ESPN devuelve `score` de dos formas según el endpoint:
- `/scoreboard` → string plano `"3"`
- `/schedule` → dict `{"value": 3.0, "displayValue": "3"}`

`int({"value": 3.0, ...})` lanza `TypeError`. En entornos ES/COL `value` puede ser `"3,0"` (coma decimal), haciendo fallar `float()` también.

**Fix (`espn_collector.py` v4):**
```python
parsed_home = _parse_score(home_score)
parsed_away = _parse_score(away_score)
"home_goals": parsed_home,
"away_goals": parsed_away,
"home_ft":    parsed_home if status in FINISHED_STATUSES and parsed_home is not None else None,
"away_ft":    parsed_away if status in FINISHED_STATUSES and parsed_away is not None else None,
```

`_parse_score()` usa `displayValue` como fuente primaria y normaliza coma decimal como fallback.

---

## Fix: Histórico ESPN ampliado a 2022 (`settings.py` v7 + `_01_data_collector.py` v5)

**Problema:** `download_espn_historical` usaba las últimas 3 temporadas hardcodeadas. El clasificador `draw` y Dixon-Coles en ligas pequeñas sufrían de dataset insuficiente.

**Fix:**
```python
# config/settings.py v7
ESPN_HISTORICAL_SEASONS: list[int] = list(range(2022, date.today().year + 1))
# Resultado actual: [2022, 2023, 2024, 2025, 2026]
# Se extiende automáticamente cada año sin modificar código.
```

**Backfill ejecutado (2026-04-02):**
```powershell
python -c "from src._01_data_collector import download_espn_historical; download_espn_historical()"
python -c "import os,glob; [os.remove(f) for f in glob.glob('models/*.pkl')]"
python scheduler.py
```

**Resultado:** 14.314 partidos de 14 ligas (2022–2026). Copa Libertadores y Copa Sudamericana ahora tienen DC propio. 9 ligas nuevas con modelos propios.

---

## Fix: league_id lookup ESPN (`_03_model_engine.py` v6)

**Problema:** Las ligas ESPN se guardan en `dc.models` con `league_name` como clave string (ej: `'Liga BetPlay'`) pero `predict_proba` recibía `league_id=501` → caía al global con probs por defecto.

**Síntoma:** Los 3 partidos del día tenían exactamente las mismas probabilidades (47.2% / 27.3% / 25.4%).

**Fix (`_03_model_engine.py` v6):**
```python
def _resolve_league_name(league_id: int | str) -> str | None:
    from config.settings import LIGAS_ESPN, COMPETICIONES_NACIONALES_ESPN
    _todos = {**LIGAS_ESPN, **COMPETICIONES_NACIONALES_ESPN}
    return next(
        (name for slug, (lid, name) in _todos.items() if lid == league_id),
        None,
    )
```

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
- **ESPN BPI:** no disponible para ligas LATAM (0/12 partidos confirmado 2026-04-02). Solo EU y competiciones UEFA.

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

Todos los mercados tienen evaluador completo en `evaluate_bet`. Mercados desconocidos retornan `False` sin lanzar excepción.

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

**Output:** DataFrame con columnas `Club`, `Elo`, `n_partidos`, `league_name` compatible con `load_elo()`.
**Post-backfill:** ELO ESPN cubre **408 equipos** LATAM. ELO total fusionado: **927 equipos únicos**.

---

## `_05_result_updater.py` v2 — cambios principales

### evaluate_bet — cobertura completa
Todos los mercados del sistema tienen evaluador. Implementación con dict de lambdas:

```python
evaluators = {
    "home_win":  lambda: home_goals > away_goals,
    "draw":      lambda: home_goals == away_goals,
    "away_win":  lambda: home_goals < away_goals,
    "double_1x": lambda: home_goals >= away_goals,
    "double_x2": lambda: away_goals >= home_goals,
    "double_12": lambda: home_goals != away_goals,
    "btts_si":   lambda: btts,
    "btts_no":   lambda: not btts,
    # Over/Under 0.5–4.5, por equipo, exactos, combinadas, Asian Handicap...
    "home_and_btts": lambda: (home_goals > away_goals) and btts,
    "draw_and_btts": lambda: (home_goals == away_goals) and btts,
    "away_and_btts": lambda: (home_goals < away_goals) and btts,
    "ah_home_minus1":  lambda: (home_goals - away_goals) >= 2,
    "ah_away_minus1":  lambda: (home_goals - away_goals) < 2,
}
```

### compute_model_stats — desglose completo
Calcula ROI y tasa de acierto total + por nivel (alta/media/baja) + por mercado.
Guarda snapshot en tabla `estadisticas_modelo`.

### Fuentes de resultados (orden de prioridad)
1. `football-data.org` — ligas EU (fuente principal)
2. `ESPN API` — LATAM + Champions (sin key, fallback automático)
3. `API-Football` — fallback final (requiere key)

### DDL v3 — cambios respecto a v2
- Tabla `resultados` añadida
- Columnas `league_id`, `odds_source`, `odds_provider`, `dc_exp_home`, `dc_exp_away` en `predicciones`
- Vista `predicciones_abiertas`
- Índice compuesto `idx_pred_fecha_ganada WHERE ganada IS NULL`
- RLS policies documentadas (comentadas — se usa `service_role key`)

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
| home_injury_score | float | Score de lesiones local (sum. impacto lesionados) |
| away_injury_score | float | Score de lesiones visitante |
| injury_score_diff | float | `home_injury_score − away_injury_score` (feature derivada) |
| espn_bpi_home_prob | float | Probabilidad pre-partido ESPN BPI — local (siempre 0 en LATAM) |
| espn_bpi_away_prob | float | Probabilidad pre-partido ESPN BPI — visitante (siempre 0 en LATAM) |

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
**Fix aplicado en `utils.py` v4:**
- `days_ahead <= 1`: usa `forecast_days=N` **sin** `start_date/end_date`; selecciona el elemento `[days_ahead]` del array de respuesta.
- `days_ahead >= 2`: sigue usando `start_date/end_date` como antes.
- `days_ahead > 16`: devuelve valores neutros sin llamar a la API.

**Bug activo:** `SSLEOFError` intermitente en Colombia (2026-04-02). Probablemente red/proxy. No es bug de código.

### Supabase — error de versión ✅ RESUELTO
`supabase==2.3.0` es la versión compatible confirmada.

### Supabase DDL — nivel baja no incluido ✅ RESUELTO
CHECK corregido en DDL v2: `CHECK (confianza IN ('alta','media','baja'))`.

### Supabase 403 Forbidden en inserts ✅ RESUELTO
Causa: `anon key` sin permisos de escritura con RLS desactivado. Fix: `service_role key` en `.env`.

### Copa Libertadores / Sudamericana sin DC propio ✅ RESUELTO
Post-backfill 2022–2023: Copa Libertadores 472 partidos y Copa Sudamericana 298 partidos — ambas superan mínimo 200 y tienen DC propio entrenado.

### ESPN BPI en LATAM — confirmado sin cobertura
`espn_bpi_home_prob` siempre es 0 en ligas LATAM por `fillna(0)`. Ver pendiente #19.

### Cúcuta Deportivo — pocos partidos históricos
Solo 6 partidos como local. Con `MIN_PARTIDOS_BAJA=6` ya genera alertas de nivel baja.

### `_parse_fixture` — score como string vs dict ✅ RESUELTO (v4)

### `_safe_int` — coma decimal en standings (FIX B1)
ESPN standings devuelve stats como `"27,0"` en entornos ES/COL. `_safe_int()` convierte con `str(val).replace(",", ".")`.

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

**Ejecución:**
```powershell
pytest tests/                                              # todos los tests rápidos
pytest tests/ -m slow                                      # incluye entrenamiento de modelos
pytest tests/ --cov=src                                    # con cobertura
pytest tests/test_result_updater.py -v -k "evaluate_bet"  # solo mercados
```

**Estado tests:** todos pasan ✅

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
    # NOTA: usar 'btts' (no 'btts_si') en el loop de mercados
    for market in ['home_win','draw','away_win','btts','over25','under25']:
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
SUPABASE_KEY=<configurado>               # ✅ real — service_role key (sb_secret_tS...)
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

## `settings.py` v8 — Catálogo ampliado y feature flags

### Slugs por región
```python
# config/settings.py v8

SLUGS_LATAM_CLUBES: list[str] = [
    "col.1", "arg.1", "bra.1", "mex.1",
    "chi.1", "ecu.1", "per.1", "uru.1", "ven.1",
]

SLUGS_EU_ESPN: list[str] = [
    "eng.1", "esp.1", "ger.1", "ita.1",
    "fra.1", "ned.1", "por.1",
]

SLUGS_COPAS_UEFA: list[str] = [
    "conmebol.libertadores", "conmebol.sudamericana",
    "uefa.champions", "uefa.europa", "uefa.europa.conference",
]
```

### Feature flags
```python
ESPN_INJURIES_ENABLED: bool = True   # get_team_injuries + enrich_fixtures_with_injuries
ESPN_BPI_ENABLED:      bool = True   # get_match_summary_bpi + enrich_fixtures_with_bpi
ESPN_STANDINGS_FEATURES: bool = True # standings como feature adicional (posición, racha)
```

**Impacto en pipeline:** `_02_feature_builder.build_features_for_fixtures()` comprueba cada flag antes de llamar al enriquecedor correspondiente. Con todos en `False`, el comportamiento es idéntico a v6.

---

## `espn_collector.py` v5 — Funciones de lesiones, BPI y goleadores

### `get_team_injuries(slug, team_id)`
```python
[
    {"player": "Radamel Falcao", "status": "Out", "type": "Muscle",
     "return_date": "2026-04-10", "impact_score": 0.85},
]
```
`impact_score`: `Out=1.0`, `Doubtful=0.6`, `Questionable=0.35`, `Probable=0.1`.

### `enrich_fixtures_with_injuries(fixtures_df, slug)`
Añade `home_injury_score`, `away_injury_score`, `injury_score_diff`. Respeta `ESPN_INJURIES_ENABLED`.

### `get_match_summary_bpi(slug, event_id)`
```python
{"home_win_prob": 0.54, "away_win_prob": 0.28, "tie_prob": 0.18,
 "home_bpi": 73.2, "away_bpi": 68.1}
```
Retorna `None` para ligas LATAM (sin cobertura BPI — confirmado 2026-04-02).

### `enrich_fixtures_with_bpi(fixtures_df, slug)`
Añade `espn_bpi_home_prob` y `espn_bpi_away_prob`. Quedan como `NaN` si BPI no disponible.

### `get_league_top_scorers(slug, season=None)`
Retorna el top-10 de goleadores de la liga.

---

## `_01_data_collector.py` v6 — Lógica de grupos en `download_espn_historical()`

```python
def download_espn_historical(fetch_plays=False, max_per_team=None, seasons=None):
    if seasons is None:
        seasons = ESPN_HISTORICAL_SEASONS

    slugs_to_fetch = list(SLUGS_LATAM_CLUBES)          # siempre
    slugs_to_fetch += list(SLUGS_COPAS_UEFA)            # siempre
    if ESPN_ONLY:
        slugs_to_fetch += list(SLUGS_EU_ESPN)           # solo si no hay fd.org activo

    for slug in slugs_to_fetch:
        for season in seasons:
            _download_league_season(slug, season, ...)
```

---

## `_02_feature_builder.py` v7 — Enriquecimiento encadenado

```python
def build_features_for_fixtures(fixtures_df, historical, ...):
    df = enrich_fixtures_with_odds(df, slug)

    if settings.ESPN_INJURIES_ENABLED:
        df = enrich_fixtures_with_injuries(df, slug)
    else:
        df[["home_injury_score","away_injury_score","injury_score_diff"]] = np.nan

    if settings.ESPN_BPI_ENABLED:
        df = enrich_fixtures_with_bpi(df, slug)
    else:
        df[["espn_bpi_home_prob","espn_bpi_away_prob"]] = np.nan

    return df
```

### `build_training_dataset()` — schema consistente
```python
NEW_COLS = [
    "home_injury_score", "away_injury_score", "injury_score_diff",
    "espn_bpi_home_prob", "espn_bpi_away_prob",
]
for col in NEW_COLS:
    if col not in df.columns:
        df[col] = np.nan
```

---

## `_03_model_engine.py` v9 — FEATURE_COLS ampliado

```python
FEATURE_COLS: list[str] = [
    "elo_diff", "home_form", "away_form", "h2h_home_wins",
    "home_goals_scored_avg", "away_goals_scored_avg",
    "home_goals_conceded_avg", "away_goals_conceded_avg",
    "dc_exp_home_goals", "dc_exp_away_goals",
    "espn_odds_home", "espn_odds_draw", "espn_odds_away",
    "espn_total_line", "espn_spread_line",
    "weather_precipitation", "weather_temp",
    # nuevas features v9
    "home_injury_score",
    "away_injury_score",
    "injury_score_diff",
    "espn_bpi_home_prob",   # siempre 0 en LATAM
    "espn_bpi_away_prob",
]

X = df[FEATURE_COLS].fillna(0)
```

---

## Versiones de archivos

### Sesión 2026-04-03

| Archivo | Versión | Cambio principal |
|---------|---------|-----------------|
| eval_v7.py | v2 | `--sweep-draw`: barrido de thresholds 0.25→0.45 (paso 0.02) para recalibrar mercado `draw` post-backfill; pre-computa probs en una sola pasada; muestra señales/accuracy/ROI/EV por threshold; guarda CSV si `--output` |
| config/settings.py | v9 | `SLUGS_SIN_BPI`: set con slugs LATAM/CONCACAF sin cobertura ESPN BPI — usado por `_02_feature_builder` para omitir llamadas API innecesarias |
| src/_02_feature_builder.py | v8 | Enriquecimiento BPI condicional por slug (omite `SLUGS_SIN_BPI`); `bpi_available` como feature binaria `int(0/1)` en `feature_row`; `build_training_dataset()` inicializa `bpi_available=0` |
| src/_03_model_engine.py | v10 | `FEATURE_COLS` reemplaza `espn_bpi_home_prob`/`espn_bpi_away_prob` por `bpi_available`; probs BPI crudas se conservan en DataFrame solo para diagnóstico |
| src/utils.py | v5 | Retry con backoff exponencial en `get_weather_for_fixture`: 3 intentos, 1s→2s→4s ± 0.3s jitter; reintenta ante `SSLError`/`ConnectionError`/`Timeout`; errores no-retriables (400) fallan rápido |

### Sesión 2026-04-03

| Archivo | Versión | Cambio principal |
|---------|---------|-----------------|
| config/settings.py | v10 | Añade `DC_TIME_DECAY_XI` (float, default 0.003), `XGB_TIME_DECAY_LAMBDA` (float, default 0.002) y `TIME_DECAY_REFERENCE_DATE` (str\|None) — configurables desde `.env`. Documentación inline de semividas y rangos recomendados para sweep. |
| src/_03_model_engine.py | v11 | **Time decay DC**: `DixonColesModel.fit` acepta `xi` — cada partido recibe `w = exp(−ξ × días_atrás)` aplicado a la log-likelihood. `DixonColesEnsemble.fit` propaga `xi=DC_TIME_DECAY_XI` a todos los modelos por liga. **Sample weight XGBoost**: `FootbotEnsemble.fit` calcula `sample_weight = exp(−λ × días_atrás)` para el set de entrenamiento usando `XGB_TIME_DECAY_LAMBDA`. Ambos parámetros en 0.0 reproducen v10 exactamente. Requiere `match_date` en el dataset (ya presente en `build_training_dataset`). **FEATURE_COLS v11**: añade 18 features de standings context (relegation_threat, title_race, clasif_race, motivation_score, pts_per_game, points_to_safety, es_tramo_final, motivation_diff, pressure_asymmetry, rank_diff, points_diff_standing — home/away + diferenciales). fillna(0) retrocompatible con histórico. |
| src/espn_collector.py | v6 | **`get_standings_context`** (F1): descarga standings por liga y calcula features de motivación por equipo. Calcula `relegation_threat`, `title_race`, `clasif_race`, `motivation_score` (score compuesto 0–1.5), `points_to_safety`, `points_to_clasif`, `season_progress`, `es_tramo_final`, `forma_wdl`. Lógica de zonas diferenciada LATAM vs EU. Fuzzy match con rapidfuzz. **`enrich_fixtures_with_standings`** (F2): enriquece fixtures_df con las 12 features por equipo + 4 diferenciales. Una sola llamada por liga al inicio del pipeline. |
| src/_02_feature_builder.py | v9 | `build_features_for_fixtures`: añade paso 4 — standings context (si `ESPN_STANDINGS_FEATURES=true`). Solo pide standings de las ligas activas que aparecen en los fixtures del día. Propaga las 28 columnas de motivación al `feature_row`. `build_training_dataset`: inicializa columnas de standings en 0 para retrocompatibilidad (histórico no tiene standings en tiempo real). |

### Sesión 2026-04-02

| Archivo | Versión | Cambio principal |
|---------|---------|-----------------|
| data/raw/espn_historical_*.csv | — | Backfill 2022–2023 ejecutado: 14.314 partidos de 14 ligas (2022-01-07 → 2026-04-01) |
| models/*.pkl | 20260402 | Reentrenamiento post-backfill: 15 ligas DC (incluye Copa Libertadores y Copa Sudamericana por primera vez), XGBoost con 11.451 partidos de entrenamiento |

### Sesión 2026-04-02 (anterior — Supabase fixes)

| Archivo | Versión | Cambio principal |
|---------|---------|-----------------|
| src/supabase_client.py | v3 | FIX 1: `"fecha"` → `"fecha_partido"`; FIX 2: `"cuota"` → `"cuota_referencia"`; FIX 3: `obtener_predicciones_abiertas` filtra por `fecha_partido`; FIX 4: añade campo `partido` al insert |
| src/_04_value_detector.py | v7 | FIX `btts_si`: `build_odds_dict` captura `_btts_prob = market_probs.get("prob_btts", 0)` y asigna `model_odds["btts_si"]` explícitamente; elimina búsqueda de `prob_btts_si` inexistente |
| supabase_ddl_v3.sql | v3 | Tabla `resultados` añadida; columnas `league_id`, `odds_source`, `odds_provider`, `dc_exp_home`, `dc_exp_away` en `predicciones`; vista `predicciones_abiertas`; índice compuesto `idx_pred_fecha_ganada`; RLS policies documentadas |
| .env | — | `SUPABASE_KEY` cambiada de `anon key` a `service_role key` — resuelve 403 Forbidden en inserts del Paso 5 |

### Sesión 2026-03-31 y anteriores

| Archivo | Versión | Cambio principal |
|---------|---------|-----------------|
| config/settings.py | v8 | Catálogo ~20 ligas; `SLUGS_LATAM_CLUBES`, `SLUGS_EU_ESPN`, `SLUGS_COPAS_UEFA`; feature flags `ESPN_INJURIES_ENABLED`, `ESPN_BPI_ENABLED`, `ESPN_STANDINGS_FEATURES` |
| src/espn_collector.py | v5 | `get_team_injuries()`, `enrich_fixtures_with_injuries()`, `get_match_summary_bpi()`, `enrich_fixtures_with_bpi()`, `get_league_top_scorers()` |
| src/_01_data_collector.py | v6 | `download_espn_historical` con lógica de grupos |
| src/_02_feature_builder.py | v7 | `build_features_for_fixtures` encadena cuotas→lesiones→BPI; `injury_score_diff` |
| src/_03_model_engine.py | v9 | `FEATURE_COLS` +5 nuevas features; `fillna(0)` retrocompatible |
| src/_03_model_engine.py | v8 | `MARKET_THRESHOLDS` por mercado (`draw=0.33`); `xgb_signal_{market}` |
| src/_01_data_collector.py | v5 | `download_espn_historical` usa `ESPN_HISTORICAL_SEASONS` (2022–actual) |
| config/settings.py | v7 | `ESPN_HISTORICAL_SEASONS = list(range(2022, date.today().year + 1))` |
| src/espn_collector.py | v4 | FIX `_parse_fixture` usa `_parse_score()` para home/away_goals |
| src/_03_model_engine.py | v7 | `scale_pos_weight` dinámico por mercado |
| src/_02_feature_builder.py | v6 | `_precompute_rolling_cache` O(n) |
| src/_03_model_engine.py | v6 | Fix league_id lookup ESPN |
| config/settings.py | v6 | Umbrales nivel BAJA, ESPN_ONLY flag |
| src/_04_value_detector.py | v6 | Nivel baja en classify_confidence, kelly reducido al 50% |
| src/telegram_sender.py | v6 | Sección 🔵 BAJA CONFIANZA en format_message |
| src/espn_collector.py | v3 | FIX A1 _parse_score, FIX A2 providers, FIX A3 _to_decimal, FIX B1 _safe_int, FIX D1/D2 seasons |
| src/supabase_client.py | v1 | Cliente Supabase — insert/select/delete predicciones |
| supabase_ddl_v2.sql | v2 | CHECK corregido: `'baja'` incluido; columnas home/away_goals; idx_mercado; vistas roi_por_mercado y resumen_diario |
| src/_01_data_collector.py | v4 | Retry backoff WinError 10054; fusión fd.org + ESPN; **compute_elo_espn** |
| src/_05_result_updater.py | v2 | evaluate_bet completo (~40 mercados); compute_model_stats con 3 niveles y por mercado |
| src/utils.py | v4 | Fix Open-Meteo 400 Bad Request para hoy/mañana |
| src/nacional_features.py | v4 | Eliminado import roto de nacional_collector |
| tests/test_result_updater.py | v2 | 90+ casos evaluate_bet; tests compute_model_stats 3 niveles |

---

## Notas para próximas sesiones

1. ~~**Supabase:** `pip install supabase==1.2.0` y corregir DDL (`'baja'` en CHECK)~~ ✅ **RESUELTO**
2. ~~**Telegram:** Configurar token real y verificar envío~~ ✅ **RESUELTO**
3. ~~**`evaluate_bet`:** Añadir evaluadores para over15, under15, over45, by-team markets, exactos, combinadas y AH~~ ✅ **RESUELTO**
4. ~~**`compute_model_stats` sin desglose por nivel ni mercado**~~ ✅ **RESUELTO**
5. ~~**ELO LATAM = 0.0 (ClubElo no cubre LATAM)**~~ ✅ **RESUELTO**
6. ~~**Open-Meteo:** Usar `forecast_days=1` sin `start_date/end_date` para partidos de hoy~~ ✅ **RESUELTO**
7. ~~**`_02_feature_builder.load_elo()`:** Leer `elo_espn.csv` además de `elo_ratings.csv`~~ ✅ **RESUELTO**
8. ~~**`build_training_dataset` O(n²):**~~ ✅ **RESUELTO**
9. ~~**`_parse_fixture`:** Reemplazar `int(home_score)` por `_parse_score(home_score)`~~ ✅ **RESUELTO**
10. ~~**Copa Libertadores / Sudamericana:** < 200 partidos → sin DC propio~~ ✅ **RESUELTO** — post-backfill 2022–2023: Copa Libertadores 472 partidos, Copa Sudamericana 298 partidos. Ambas tienen DC propio entrenado.
11. **Tests @slow:** `FootbotEnsemble.fit` no corre en CI normal — marcar y ejecutar explícitamente con `pytest -m slow`.
12. ~~**Distribución de clases XGBoost:**~~ ✅ **RESUELTO** — `scale_pos_weight` dinámico en v7.
13. ~~**Métricas XGBoost post-fix:** correr evaluación formal post-reentrenamiento~~ ✅ **RESUELTO** — `eval_v7.py` ejecutado el 2026-04-02 con 14.314 partidos (2.863 de validación). Ver sección **Evaluación formal post-backfill**.
14. ~~**Bug `btts_si` en producción:**~~ ✅ **RESUELTO** — `_04_value_detector.py` v7. **Pendiente menor:** `diag2.py` aún usa `'btts_si'` → cambiar a `'btts'` en el loop de mercados (no afecta producción).
15. ~~**Mercado `draw` — threshold pendiente:**~~ ✅ **RESUELTO (código)** — `_03_model_engine.py` v8: `MARKET_THRESHOLDS["draw"] = 0.33`. **Recalibración pendiente con sweep-draw:** `eval_v7.py --sweep-draw` disponible para correr el barrido 0.25→0.45 sobre 2.863 partidos y encontrar threshold óptimo post-backfill.
16. ~~**Ampliar histórico con temporadas 2022–2023:**~~ ✅ **RESUELTO** — backfill ejecutado el 2026-04-02. 14.314 partidos de 14 ligas.
17. ~~**Reentrenamiento post-backfill + nuevas features:**~~ ✅ **RESUELTO** — reentrenamiento ejecutado 2026-04-02. `eval_v7.py` corrido. Feature importance de lesiones/BPI pendiente de analizar formalmente.
18. **Activar `ESPN_STANDINGS_FEATURES`:** una vez implementado el enriquecedor de standings, añadir features de posición en tabla y racha de últimos 5 partidos como inputs adicionales para XGBoost.
19. ~~**Cobertura BPI en LATAM:**~~ ✅ **RESUELTO** — `espn_bpi_home_prob`/`away_prob` eliminadas de `FEATURE_COLS`; sustituidas por `bpi_available` (flag binario `0/1`). `SLUGS_SIN_BPI` en `settings.py` evita llamadas API innecesarias para LATAM. **Requiere reentrenar modelos** post-cambio de schema.
20. **Test unitario lesiones + BPI:** añadir `test_espn_collector.py` casos para las 5 nuevas funciones v5 (mock del endpoint ESPN `/injuries` y `/bpi`).
21. ~~**Supabase 403 Forbidden en inserts (Paso 5):**~~ ✅ **RESUELTO** — `service_role key` en `.env`.
22. ~~**Verificar inserts post-fix:**~~ ✅ **RESUELTO** — pipeline 2026-04-02 insertó 5 predicciones correctamente (HTTP 201 ×5).
23. ~~**`diag2.py` bug menor pendiente:**~~ cambiar `'btts_si'` → `'btts'` en el loop de mercados del script de diagnóstico (no afecta producción).
24. ~~**Recalibrar threshold `draw` post-backfill:**~~ ✅ **HERRAMIENTA LISTA** — `eval_v7.py --sweep-draw` implementado. Pendiente ejecutarlo y actualizar `MARKET_THRESHOLDS["draw"]` en `_03_model_engine.py` con el resultado.
25. ~~**Open-Meteo SSL error Colombia:**~~ ✅ **RESUELTO** — `utils.py` v5: retry con backoff exponencial (3 intentos, 1s→2s→4s ± 0.3s jitter).
26. ~~**Optimizar `espn_bpi_home_prob` en LATAM:**~~ ✅ **RESUELTO** — `bpi_available` reemplaza las probs crudas en `FEATURE_COLS`; `SLUGS_SIN_BPI` en `settings.py` evita llamadas API para slugs sin cobertura.
27. ~~**Reentrenar modelos post-cambio FEATURE_COLS (v10):**~~ ✅ **RESUELTO** — ver nota 27 siguiente: el reentrenamiento post-v11 cubre este punto también.
28. **Correr `eval_v7.py --sweep-draw`:** tras el reentrenamiento con time decay (#29), correr el barrido para recalibrar `MARKET_THRESHOLDS["draw"]` con el nuevo modelo. Actualizar el valor en `_03_model_engine.py` con el threshold óptimo encontrado.
29. **Reentrenar modelos post-v11 (time decay + standings features):** `_03_model_engine.py` v11 cambió la log-likelihood de DC, añadió `sample_weight` a XGBoost y amplió `FEATURE_COLS` con 18 features de standings. Los modelos `.pkl` actuales (20260402) son incompatibles. Ejecutar `python scheduler.py` para reentrenar. **Hacerlo antes de correr en producción.**
30. **Sweep de parámetros de decay:** tras el primer reentrenamiento con defaults (`xi=0.003`, `λ=0.002`), evaluar con `eval_v7.py` y comparar ROI vs v10 (sin decay). Si mejora, hacer sweep: `DC_TIME_DECAY_XI` en [0.001, 0.002, 0.003, 0.004, 0.005] y `XGB_TIME_DECAY_LAMBDA` en [0.001, 0.002, 0.003] — configurables desde `.env` sin tocar código.
31. **Activar `ESPN_STANDINGS_FEATURES` y validar feature importance:** tras el reentrenamiento (#29), correr `eval_v7.py` y revisar `feature_importances` de XGBoost para los nuevos campos (`home_motivation_score`, `home_relegation_threat`, etc.). Si aparecen en el top-10 de algún mercado, la señal es real. Si importancia ~0 en todos, considerar reducir o eliminar las features menos informativas.
32. **Calibrar `total_jornadas_est` por liga:** en `espn_collector._compute_standings_features`, el dict `total_jornadas_est` tiene valores aproximados. Verificar con datos reales al final de las temporadas 2025 y ajustar (especialmente par.1=22, bol.1=22, ven.1=18 que son estimaciones).
33. **Test unitario standings context:** añadir `test_espn_collector.py` casos para `get_standings_context` y `_compute_standings_features` con datos mock de la respuesta de col.1 confirmada el 2026-04-03.