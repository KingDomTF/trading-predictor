//+------------------------------------------------------------------+
//|                                     Streamlit_FileWriter.mq4     |
//|                        TITAN Trading Systems                     |
//|           Esporta Dati JSON per il Ponte Python (V90)            |
//+------------------------------------------------------------------+
#property copyright "TITAN V90"
#property link      "https://github.com/"
#property version   "2.00"
#property strict

// --- INPUT ---
input int RefreshRateSeconds = 1; // Ogni quanti secondi aggiornare il file

// --- VARIABILI GLOBALI ---
string fileName;

//+------------------------------------------------------------------+
//| Inizializzazione                                                 |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Il nome del file sarà AUTOMATICO in base al grafico (es: EURUSD_data.json)
   fileName = Symbol() + "_data.json";
   
   // Imposta il timer per l'aggiornamento
   EventSetTimer(RefreshRateSeconds);
   
   Print("🚀 TITAN WRITER AVVIATO: Scrivo su ", fileName);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Chiusura                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   Print("🛑 TITAN WRITER FERMATO");
  }

//+------------------------------------------------------------------+
//| Timer Loop (Esegue ogni X secondi)                               |
//+------------------------------------------------------------------+
void OnTimer()
  {
   WriteData();
  }

//+------------------------------------------------------------------+
//| Tick Loop (Esegue ad ogni movimento di prezzo)                   |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Decommenta la riga sotto se vuoi aggiornamenti istantanei (usa più CPU)
   // WriteData(); 
  }

//+------------------------------------------------------------------+
//| Funzione di Scrittura File JSON                                  |
//+------------------------------------------------------------------+
void WriteData()
  {
   // 1. RECUPERA I DATI DI MERCATO
   double bid = MarketInfo(Symbol(), MODE_BID);
   double ask = MarketInfo(Symbol(), MODE_ASK);
   double spread = ask - bid;
   double price = (bid + ask) / 2.0;
   
   // Recupera Equity e Saldo
   double equity = AccountEquity();
   double balance = AccountBalance();
   
   // 2. CREA LA STRINGA JSON MANUALMENTE
   // Attenzione: MQL4 non ha librerie JSON native, lo costruiamo come stringa
   string json = "{";
   json += "\"symbol\": \"" + Symbol() + "\",";
   json += "\"time\": \"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\",";
   json += "\"bid\": " + DoubleToString(bid, Digits) + ",";
   json += "\"ask\": " + DoubleToString(ask, Digits) + ",";
   json += "\"price\": " + DoubleToString(price, Digits) + ",";
   json += "\"spread\": " + DoubleToString(spread, Digits) + ",";
   json += "\"equity\": " + DoubleToString(equity, 2) + ",";
   json += "\"balance\": " + DoubleToString(balance, 2);
   json += "}";
   
   // 3. SCRIVI SU FILE
   // FILE_WRITE sovrascrive il file ogni volta (così Python legge sempre l'ultimo)
   // I file vengono salvati in: MQL4/Files/
   int file_handle = FileOpen(fileName, FILE_WRITE|FILE_TXT|FILE_ANSI);
   
   if(file_handle != INVALID_HANDLE)
     {
      FileWrite(file_handle, json);
      FileClose(file_handle);
     }
   else
     {
      // Stampa errore solo se grave, per non spammare il log
      // Print("Errore scrittura file: ", GetLastError());
     }
  }
//+------------------------------------------------------------------+
