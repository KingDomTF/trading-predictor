"""
Quick MT4 Connection Test - Palantir Engineering
Testa immediatamente se MT4 sta comunicando correttamente
"""

import json
import os
from pathlib import Path
from datetime import datetime

# IL TUO PERCORSO ESATTO MT4
MT4_PATH = r"C:\Users\dcbat\AppData\Roaming\MetaQuotes\Terminal\B8925BF731C22E88F33C7A8D7CD3190E\MQL4\Files\MT4_Bridge"

print("=" * 70)
print("🔍 MT4 BRIDGE CONNECTION TEST")
print("=" * 70)

# Test 1: Path exists
print(f"\n1️⃣ Testing path: {MT4_PATH}")
if os.path.exists(MT4_PATH):
    print("   ✅ Path EXISTS")
else:
    print("   ❌ Path NOT FOUND")
    print("   💡 Solution: The EA hasn't created the folder yet")
    print("   🔧 Action: Make sure EA is running on MT4 chart")
    exit(1)

# Test 2: List files
print(f"\n2️⃣ Files in folder:")
try:
    files = os.listdir(MT4_PATH)
    if files:
        for f in files:
            file_path = os.path.join(MT4_PATH, f)
            size = os.path.getsize(file_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            age = (datetime.now() - mtime).total_seconds()
            print(f"   📄 {f} ({size} bytes, {age:.0f}s ago)")
    else:
        print("   ⚠️ Folder is EMPTY")
        print("   💡 EA is not writing files")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Heartbeat
print(f"\n3️⃣ Testing HEARTBEAT:")
heartbeat_file = os.path.join(MT4_PATH, "heartbeat.json")

if os.path.exists(heartbeat_file):
    print(f"   ✅ heartbeat.json EXISTS")
    
    try:
        with open(heartbeat_file, 'r') as f:
            data = json.load(f)
        
        print(f"   📊 Content:")
        for key, value in data.items():
            print(f"      {key}: {value}")
        
        # Check timestamp
        timestamp_str = data.get('timestamp', '')
        if timestamp_str:
            try:
                # Try parsing
                last_beat = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                age = (datetime.now() - last_beat).total_seconds()
                
                print(f"\n   ⏱️  Heartbeat age: {age:.1f} seconds")
                
                if age < 10:
                    print(f"   ✅✅✅ MT4 IS ACTIVE AND CONNECTED!")
                    print(f"   🎉 Everything working correctly!")
                elif age < 60:
                    print(f"   ⚠️ MT4 connection is STALE")
                    print(f"   💡 EA might have stopped updating")
                else:
                    print(f"   ❌ MT4 connection is OLD")
                    print(f"   💡 Restart EA or check if it's running")
            except Exception as e:
                print(f"   ⚠️ Could not parse timestamp: {e}")
                print(f"   ℹ️  But file exists, so EA is writing")
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
else:
    print(f"   ❌ heartbeat.json NOT FOUND")
    print(f"   💡 EA is not creating this file")
    print(f"   🔧 Check MT4 Journal (Ctrl+T) for errors")

# Test 4: Live Price
print(f"\n4️⃣ Testing LIVE PRICE:")
price_file = os.path.join(MT4_PATH, "live_price.json")

if os.path.exists(price_file):
    print(f"   ✅ live_price.json EXISTS")
    
    try:
        with open(price_file, 'r') as f:
            data = json.load(f)
        
        print(f"   📊 Current Market Data:")
        print(f"      Symbol: {data.get('symbol', 'N/A')}")
        print(f"      Bid: {data.get('bid', 'N/A')}")
        print(f"      Ask: {data.get('ask', 'N/A')}")
        print(f"      Spread: {data.get('spread', 'N/A')} points")
        print(f"      Time: {data.get('timestamp', 'N/A')}")
        
        print(f"   ✅ Live prices are FLOWING!")
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
else:
    print(f"   ⚠️ live_price.json NOT FOUND")
    print(f"   💡 Set SendLivePrice=true in EA parameters")

# Test 5: Status
print(f"\n5️⃣ Testing STATUS:")
status_file = os.path.join(MT4_PATH, "status.json")

if os.path.exists(status_file):
    print(f"   ✅ status.json EXISTS")
    
    try:
        with open(status_file, 'r') as f:
            data = json.load(f)
        
        print(f"   📊 Account Info:")
        print(f"      Balance: ${data.get('balance', 0):.2f}")
        print(f"      Equity: ${data.get('equity', 0):.2f}")
        print(f"      Open Trades: {data.get('open_trades', 0)}")
        print(f"      Auto Trade: {data.get('auto_trade', False)}")
        
        print(f"   ✅ Status data is AVAILABLE!")
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
else:
    print(f"   ⚠️ status.json NOT FOUND")

# Summary
print("\n" + "=" * 70)
print("📋 SUMMARY & RECOMMENDATIONS")
print("=" * 70)

all_good = all([
    os.path.exists(MT4_PATH),
    os.path.exists(heartbeat_file),
    os.path.exists(price_file),
    os.path.exists(status_file)
])

if all_good:
    print("\n✅✅✅ ALL SYSTEMS GO!")
    print("\n🎯 Your Python app should connect successfully with:")
    print(f'\n    bridge = MT4Bridge(bridge_folder=r"{MT4_PATH}")')
    print("\n🚀 Start your Streamlit app now!")
else:
    print("\n⚠️ ISSUES DETECTED")
    print("\n🔧 TROUBLESHOOTING STEPS:")
    print("   1. Open MT4")
    print("   2. Check EA is on the chart (should see panel)")
    print("   3. Press Ctrl+T to open Journal")
    print("   4. Look for 'AI TRADING BRIDGE EA INITIALIZED'")
    print("   5. Verify the path printed in Journal matches:")
    print(f"      {MT4_PATH}")
    print("   6. If path is different, update Python code")
    print("   7. Make sure AutoTrading is enabled (Ctrl+E or click button)")

print("\n" + "=" * 70)
print("Press Enter to exit...")
input()
