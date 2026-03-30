"""
FOOTBOT — tests/test_conexiones.py
Verifica que Supabase y Telegram están correctamente configurados.
Ejecutar: python tests/test_conexiones.py
"""

import os
import sys
import json
import urllib.request
from datetime import date

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Variables de entorno
# ─────────────────────────────────────────────────────────────────────────────

def test_env_vars():
    print("\n📋 TEST 1: Variables de entorno")
    vars_requeridas = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "TELEGRAM_TOKEN",
        "TELEGRAM_CHAT_ID",
    ]
    ok = True
    for var in vars_requeridas:
        val = os.getenv(var, "")
        if not val or "TU_" in val:
            print(f"  ❌ {var} — no configurada")
            ok = False
        else:
            print(f"  ✅ {var} — OK ({val[:12]}...)")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Supabase conexión
# ─────────────────────────────────────────────────────────────────────────────

def test_supabase():
    print("\n🗄️  TEST 2: Supabase")
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        client = create_client(url, key)
        client.table("predicciones").select("id").limit(1).execute()
        print("  ✅ Conexión exitosa — tabla 'predicciones' accesible")
        return True
    except ImportError:
        print("  ❌ supabase no instalado — ejecuta: pip install supabase==2.3.0")
        return False
    except Exception as e:
        if "relation" in str(e).lower() or "does not exist" in str(e).lower():
            print("  ⚠️  Conexión OK pero tablas no creadas — ejecuta supabase_ddl_v2.sql")
            return True  # conexión funciona, solo faltan las tablas
        print(f"  ❌ Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Supabase insert de prueba
# ─────────────────────────────────────────────────────────────────────────────

def test_supabase_insert():
    print("\n💾 TEST 3: Supabase insert (nivel baja)")
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        client = create_client(url, key)

        # Insert con nivel 'baja' — este era el bug del CHECK
        data = {
            "fecha":       str(date.today()),
            "liga":        "TEST",
            "league_id":   0,
            "home_team":   "EquipoTest",
            "away_team":   "EquipoRival",
            "mercado":     "home_win",
            "prob_modelo": 0.5500,
            "cuota":       1.850,
            "edge_pct":    2.5,
            "kelly_pct":   0.0125,
            "confianza":   "baja",   # ← bug corregido en DDL v2
            "odds_source": "test",
        }
        res = client.table("predicciones").insert(data).execute()
        pred_id = res.data[0]["id"]
        print(f"  ✅ Insert nivel 'baja' OK — id={pred_id}")

        # Limpiar el registro de prueba
        client.table("predicciones").delete().eq("id", pred_id).execute()
        print(f"  🧹 Registro de prueba eliminado")
        return True
    except Exception as e:
        print(f"  ❌ Error en insert: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Telegram envío de mensaje
# ─────────────────────────────────────────────────────────────────────────────

def test_telegram():
    print("\n📱 TEST 4: Telegram")
    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("  ❌ TELEGRAM_TOKEN o TELEGRAM_CHAT_ID no configurados")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id":    chat_id,
        "text":       "✅ *FOOTBOT conectado* — Supabase y Telegram funcionando correctamente.",
        "parse_mode": "Markdown",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                print("  ✅ Mensaje enviado a Telegram correctamente")
                return True
            else:
                print(f"  ❌ Telegram respondió: {result}")
                return False
    except Exception as e:
        print(f"  ❌ Error enviando a Telegram: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  FOOTBOT — Test de conexiones")
    print("=" * 50)

    resultados = {
        "env":      test_env_vars(),
        "supa_con": test_supabase(),
        "supa_ins": test_supabase_insert(),
        "telegram": test_telegram(),
    }

    print("\n" + "=" * 50)
    total = sum(resultados.values())
    print(f"  Resultado: {total}/{len(resultados)} tests OK")

    if total == len(resultados):
        print("  🎉 Todo listo — FOOTBOT puede guardar y notificar")
    else:
        fallidos = [k for k, v in resultados.items() if not v]
        print(f"  ⚠️  Pendiente: {', '.join(fallidos)}")

    print("=" * 50)
    sys.exit(0 if total == len(resultados) else 1)