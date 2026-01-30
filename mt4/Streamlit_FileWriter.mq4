//+------------------------------------------------------------------+
//| Streamlit_FileWriter.mq4                                          |
//| TITAN Trading Systems - Optimized for V90                        |
//+------------------------------------------------------------------+
#property strict
#property copyright "TITAN Trading Systems"
#property version   "2.0"

input int UpdateIntervalMs = 500;  // Update every 500ms

datetime lastUpdate = 0;

int OnInit()
{
   Print("═════════════════════════════════════════════");
   Print("TITAN FileWriter v2.0 Started");
   Print("Symbol: ", Symbol());
   Print("Path: ", TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL4\\Files\\");
   Print("═════════════════════════════════════════════");
   
   // Test write
   string testFile = "test.txt";
   int handle = FileOpen(testFile, FILE_WRITE|FILE_TXT);
   
   if(handle == INVALID_HANDLE)
   {
      Print("❌ ERROR: Cannot write files!");
      Print("Enable: Tools → Options → Expert Advisors → Allow DLL imports");
      return(INIT_FAILED);
   }
   
   FileWriteString(handle, "TITAN FileWriter OK");
   FileClose(handle);
   FileDelete(testFile);
   
   Print("✅ Write test SUCCESSFUL");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("TITAN FileWriter stopped");
}

void OnTick()
{
   if(GetTickCount() - lastUpdate < UpdateIntervalMs)
      return;
   
   lastUpdate = GetTickCount();
   
   // Get market data
   double bid = MarketInfo(Symbol(), MODE_BID);
   double ask = MarketInfo(Symbol(), MODE_ASK);
   double spread = ask - bid;
   double mid = (bid + ask) / 2.0;
   double equity = AccountEquity();
   double balance = AccountBalance();
   
   // Validate data
   if(bid <= 0 || ask <= 0 || ask < bid)
   {
      Print("⚠️ Invalid bid/ask: ", bid, " / ", ask);
      return;
   }
   
   // Build JSON
   string json = "{";
   json += "\"symbol\":\"" + Symbol() + "\",";
   json += "\"bid\":" + DoubleToString(bid, Digits) + ",";
   json += "\"ask\":" + DoubleToString(ask, Digits) + ",";
   json += "\"spread\":" + DoubleToString(spread, Digits) + ",";
   json += "\"price\":" + DoubleToString(mid, Digits) + ",";
   json += "\"equity\":" + DoubleToString(equity, 2) + ",";
   json += "\"balance\":" + DoubleToString(balance, 2) + ",";
   json += "\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\"";
   json += "}";
   
   // Write to file
   string filename = Symbol() + "_data.json";
   int handle = FileOpen(filename, FILE_WRITE|FILE_TXT);
   
   if(handle != INVALID_HANDLE)
   {
      FileWriteString(handle, json);
      FileClose(handle);
      
      // Visual feedback every 10s
      static datetime lastPrint = 0;
      if(TimeCurrent() - lastPrint >= 10)
      {
         Print("✅ ", Symbol(), " | Bid: ", bid, " | Ask: ", ask, " | Spread: ", spread);
         lastPrint = TimeCurrent();
      }
   }
   else
   {
      Print("❌ Cannot write: ", filename, " | Error: ", GetLastError());
   }
}
//+------------------------------------------------------------------+
