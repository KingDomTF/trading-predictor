"""
🚀 TITAN Oracle - Script di Verifica Setup
Usa questo script per verificare che tutto sia configurato correttamente
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Verifica presenza e validità file .env"""
    print("\n📝 Controllo file .env...")
    
    if not os.path.exists(".env"):
        print("❌ File .env NON trovato!")
        print("   → Crea il file .env copiando .env.template")
        print("   → Compila con i tuoi dati Supabase e MT4_PATH")
        return False
    
    # Leggi variabili
    from dotenv import load_dotenv
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    mt4_path = os.getenv("MT4_PATH")
    
    issues = []
    
    if not url or "tuoprogetto" in url:
        issues.append("   ❌ SUPABASE_URL non configurato correttamente")
    else:
        print(f"   ✅ SUPABASE_URL: {url[:30]}...")
    
    if not key or "tua_supabase" in key:
        issues.append("   ❌ SUPABASE_KEY non configurato correttamente")
    else:
        print(f"   ✅ SUPABASE_KEY: {key[:20]}...")
    
    if not mt4_path or "TuoNome" in mt4_path:
        issues.append("   ❌ MT4_PATH non configurato correttamente")
    elif not os.path.isdir(mt4_path):
        issues.append(f"   ❌ MT4_PATH non esiste: {mt4_path}")
    else:
        print(f"   ✅ MT4_PATH: {mt4_path}")
    
    if issues:
        print("\n".join(issues))
        return False
    
    return True

def check_dependencies():
    """Verifica dipendenze Python"""
    print("\n📦 Controllo dipendenze Python...")
    
    required = ["streamlit", "supabase", "dotenv", "pandas", "numpy"]
    missing = []
    
    for pkg in required:
        try:
            if pkg == "dotenv":
                __import__("dotenv")
            else:
                __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"   ❌ {pkg} - MANCANTE")
    
    if missing:
        print(f"\n💡 Installa dipendenze mancanti con:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def check_mt4_files():
    """Verifica presenza file JSON da MT4"""
    print("\n📂 Controllo file MT4...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    mt4_path = os.getenv("MT4_PATH", "")
    
    if not mt4_path or not os.path.isdir(mt4_path):
        print("   ⚠️  MT4_PATH non valido - salta controllo")
        return True
    
    symbols = ["XAUUSD", "BTCUSD", "US500", "ETHUSD", "XAGUSD"]
    found = 0
    
    for symbol in symbols:
        filepath = os.path.join(mt4_path, f"{symbol}_data.json")
        if os.path.exists(filepath):
            print(f"   ✅ {symbol}_data.json")
            found += 1
        else:
            print(f"   ❌ {symbol}_data.json - NON TROVATO")
    
    if found == 0:
        print("\n   ⚠️  Nessun file JSON trovato!")
        print("   → Assicurati che l'EA TITAN_DataExporter.mq4 sia attivo in MT4")
        print("   → Verifica che sia compilato e in esecuzione su un grafico")
        return False
    elif found < len(symbols):
        print(f"\n   ⚠️  Trovati {found}/{len(symbols)} file")
        print("   → Alcuni simboli potrebbero non essere disponibili nel tuo broker")
        print("   → Puoi rimuoverli da Config.ASSETS in bridge_improved.py")
    
    return True

def check_supabase_connection():
    """Verifica connessione a Supabase"""
    print("\n🔌 Test connessione Supabase...")
    
    try:
        from dotenv import load_dotenv
        from supabase import create_client
        
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("   ⚠️  Credenziali mancanti - salta test")
            return True
        
        client = create_client(url, key)
        
        # Test semplice
        result = client.table("trading_signals").select("*").limit(1).execute()
        print("   ✅ Connessione riuscita!")
        return True
        
    except Exception as e:
        print(f"   ❌ Errore connessione: {str(e)[:100]}")
        print("   → Verifica che le credenziali Supabase siano corrette")
        print("   → Assicurati che le tabelle esistano nel database")
        return False

def main():
    print("="*60)
    print("🎯 TITAN Oracle - Verifica Setup")
    print("="*60)
    
    checks = [
        ("Dipendenze", check_dependencies),
        ("File .env", check_env_file),
        ("File MT4", check_mt4_files),
        ("Connessione Supabase", check_supabase_connection),
    ]
    
    results = []
    
    for name, func in checks:
        try:
            result = func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Errore durante controllo '{name}': {e}")
            results.append(False)
    
    print("\n" + "="*60)
    
    if all(results):
        print("✅ TUTTO OK! Sistema pronto per l'avvio")
        print("\n🚀 Prossimi passi:")
        print("   1. python bridge_improved.py  (avvia rilevamento)")
        print("   2. streamlit run app.py       (apri dashboard)")
    else:
        print("⚠️  Alcuni controlli non sono passati")
        print("   → Consulta la GUIDA_COMPLETA.md per istruzioni dettagliate")
        print("   → Risolvi i problemi evidenziati sopra e riesegui questo script")
    
    print("="*60)

if __name__ == "__main__":
    main()
