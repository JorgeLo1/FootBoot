# FOOTBOT — Bot de Apuestas de Fútbol

Bot de predicción de apuestas 100% gratuito basado en modelos estadísticos avanzados.

## Stack tecnológico
- **Predicción**: Dixon-Coles + XGBoost ensemble con calibración isotónica
- **Datos históricos**: Football-Data.co.uk + StatsBomb Open Data + FBref
- **Fixtures del día**: API-Football (free tier — 100 req/día)
- **ELO ratings**: ClubElo.com
- **Clima**: Open-Meteo (sin API key)
- **Base de datos**: Supabase (free tier)
- **Notificaciones**: Telegram Bot API
- **Servidor**: Oracle Cloud ARM (Always Free — 4 OCPUs, 24 GB RAM)

---

## Configuración inicial

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/footbot.git
cd footbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
```bash
export API_FOOTBALL_KEY="tu_key_de_api_football"
export TELEGRAM_TOKEN="tu_token_de_telegram_bot"
export TELEGRAM_CHAT_ID="tu_chat_id"
export SUPABASE_URL="https://xxxxx.supabase.co"
export SUPABASE_KEY="tu_anon_key"
```

O editar directamente `config/settings.py`.

### 3. Crear tablas en Supabase
```bash
python3 src/05_result_updater.py
```
Copia el SQL que imprime y ejecútalo en el SQL Editor de Supabase.

### 4. Primera ejecución (descarga datos + entrena modelo)
```bash
python3 scheduler.py
```
La primera vez descarga todos los datos históricos y entrena el modelo.
Puede tardar 10–20 minutos.

---

## Despliegue en Oracle Cloud (Always Free)

### Crear instancia ARM A1
1. Ir a Oracle Cloud Console → Compute → Instances → Create Instance
2. Shape: VM.Standard.A1.Flex → 2 OCPUs, 8 GB RAM
3. OS: Ubuntu 22.04 (ARM)
4. Agregar tu SSH key pública

### Instalar en el servidor
```bash
ssh ubuntu@TU_IP_ORACLE

# Instalar dependencias del sistema
sudo apt update && sudo apt install python3-pip python3-venv git cron -y

# Clonar y configurar
git clone https://github.com/tu-usuario/footbot.git
cd footbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configurar variables de entorno permanentes
echo 'export API_FOOTBALL_KEY="tu_key"' >> ~/.bashrc
echo 'export TELEGRAM_TOKEN="tu_token"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="tu_chat_id"' >> ~/.bashrc
echo 'export SUPABASE_URL="tu_url"' >> ~/.bashrc
echo 'export SUPABASE_KEY="tu_key"' >> ~/.bashrc
source ~/.bashrc

# Test inicial
python3 scheduler.py
```

### Configurar cron
```bash
crontab -e
```

Agregar estas dos líneas:
```
# Predicciones del día — todos los días a las 8:00 AM
0 8 * * * cd /home/ubuntu/footbot && /home/ubuntu/footbot/venv/bin/python scheduler.py >> /home/ubuntu/footbot/logs/cron.log 2>&1

# Actualizar resultados del día anterior — todos los días a las 11:00 PM
0 23 * * * cd /home/ubuntu/footbot && /home/ubuntu/footbot/venv/bin/python src/05_result_updater.py >> /home/ubuntu/footbot/logs/cron_results.log 2>&1
```

---

## Estructura del proyecto
```
footbot/
├── config/
│   └── settings.py          # Configuración central (API keys, umbrales)
├── data/
│   ├── raw/                 # CSVs de Football-Data + ELO
│   ├── statsbomb/           # JSONs de StatsBomb
│   └── processed/           # Features y datasets procesados
├── src/
│   ├── 01_data_collector.py    # Descarga de todas las fuentes
│   ├── 02_feature_builder.py   # Construcción de variables
│   ├── 03_model_engine.py      # Dixon-Coles + XGBoost
│   ├── 04_value_detector.py    # Edge% + Kelly criterion
│   ├── 05_result_updater.py    # Cierre de predicciones
│   └── telegram_sender.py      # Notificaciones
├── models/                  # Modelos entrenados (versionados)
├── logs/                    # Logs diarios
├── scheduler.py             # Orquestador principal
└── requirements.txt
```

---

## Flujo diario del sistema

```
08:00 AM — scheduler.py arranca
  │
  ├── 01 Descarga fixtures del día (API-Football)
  ├── 02 Descarga ELO actualizado (ClubElo)
  ├── 03 Calcula features por partido (xG, forma, H2H, clima...)
  ├── 04 Genera predicciones (Dixon-Coles + XGBoost)
  ├── 05 Detecta value bets (edge > 4% mínimo)
  ├── 06 Guarda predicciones en Supabase
  └── 07 Envía recomendaciones a Telegram

11:00 PM — result_updater.py arranca
  ├── Obtiene resultados finales del día (API-Football)
  ├── Cierra predicciones con ganada/perdida
  └── Actualiza estadísticas del modelo
```

---

## Umbrales de confianza

| Nivel | Edge mínimo | Prob mínima | Datos mínimos | Kelly |
|-------|-------------|-------------|---------------|-------|
| Alta  | > 8%        | > 62%       | 30+ partidos  | 2–3.5% bankroll |
| Media | > 4%        | > 55%       | 15+ partidos  | 0.5–1.5% bankroll |

---

## Re-entrenamiento automático

El modelo se re-entrena automáticamente cada lunes con todos los
nuevos resultados disponibles. Los modelos se guardan versionados
en `/models/` con formato `YYYYMMDD`.

---

## Ligas cubiertas

| Liga | Cobertura histórica |
|------|---------------------|
| Premier League | 2019–2024 |
| La Liga | 2019–2024 |
| Bundesliga | 2019–2024 |
| Serie A | 2019–2024 |
| Ligue 1 | 2019–2024 |
| Eredivisie | 2019–2024 |
| Primeira Liga | 2019–2024 |

---

## Advertencia

Este sistema es un modelo estadístico experimental con fines educativos.
Las predicciones no garantizan resultados. Apuesta con responsabilidad.
