def simulate_scalping_trades(df_ind, n_trades=1000):
    """Simula trade scalping realistici con spread e slippage (robusto a dataset corti)"""
    X_list = []
    y_list = []

    n = len(df_ind)
    horizon = 20  # barre future usate per valutare TP/SL

    # Bound preferiti (storico 'ricco'), ma adattivi se i dati sono pochi
    low = 100
    high = n - (horizon + 1)

    # Se il dataset è corto, rilassa il bound inferiore mantenendo margine per l'horizon
    if high <= low:
        low = 20
        high = n - (horizon + 1)

    # Se ancora non c'è abbastanza spazio, ritorna insiemi vuoti (il caller già gestisce len(X) < 100)
    if high <= low:
        return np.array([]), np.array([])

    # Numero di campioni simulati coerente con la finestra disponibile
    draws = min(n_trades, max(100, high - low))

    for _ in range(draws):
        idx = np.random.randint(low, high)
        row = df_ind.iloc[idx]

        # Spread realistico (0.5-2 pips per forex, più alto per commodities)
        spread_pct = np.random.uniform(0.0001, 0.0005)
        spread = row['Close'] * spread_pct

        # Direzione ed entry
        direction = 'long' if np.random.random() < 0.5 else 'short'
        entry = row['Close']

        # Target/stop in funzione dell'ATR
        atr = row['ATR']
        tp_mult = np.random.uniform(0.3, 0.8)  # Target 30-80% ATR
        sl_mult = np.random.uniform(0.2, 0.5)  # Stop 20-50% ATR

        if direction == 'long':
            entry_real = entry + spread  # Paga ask
            sl = entry_real - (atr * sl_mult)
            tp = entry_real + (atr * tp_mult)
        else:
            entry_real = entry - spread  # Prende bid
            sl = entry_real + (atr * sl_mult)
            tp = entry_real - (atr * tp_mult)

        features = generate_scalping_features(df_ind.iloc[:idx+1], entry_real, spread)

        # Valuta outcome sulle prossime 'horizon' barre
        future_slice = df_ind.iloc[idx+1:idx+1+horizon]
        if future_slice.empty:
            continue

        future_high = future_slice['High'].values
        future_low = future_slice['Low'].values

        if direction == 'long':
            hit_tp = np.any(future_high >= tp)
            hit_sl = np.any(future_low <= sl)
        else:
            hit_tp = np.any(future_low <= tp)
            hit_sl = np.any(future_high >= sl)

        # Successo solo se TP toccato prima (e senza SL)
        if hit_tp and not hit_sl:
            success = 1
        elif hit_sl:
            success = 0
        else:
            # nessun esito -> non includere
            continue

        X_list.append(features)
        y_list.append(success)

    return np.array(X_list), np.array(y_list)
