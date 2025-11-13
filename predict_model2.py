import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


def make_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = np.log(out["Close"]).diff()
    out["ret_ma_5"]  = out["ret"].rolling(5).mean()
    out["ret_ma_21"] = out["ret"].rolling(21).mean()
    out["vol_21"]    = out["ret"].rolling(21).std()
    out["sma50"]     = out["Close"].rolling(50).mean()
    out["sma200"]    = out["Close"].rolling(200).mean()

    def rsi(series, window=14):
        d = series.diff()
        up = d.clip(lower=0).rolling(window).mean()
        dn = (-d.clip(upper=0)).rolling(window).mean()
        rs = up / (dn + 1e-9)
        return 100 - (100 / (1 + rs))
    out["rsi_14"] = rsi(out["Close"], 14)

    base_next = out["ret"].shift(-1)
    out["target_ret_next"] = base_next.rolling(horizon).sum() if horizon > 1 else base_next
    out["target_up_next"]  = (out["target_ret_next"] > 0).astype(int)
    return out.dropna().copy()


def equity_from_log_returns(log_rets: np.ndarray) -> np.ndarray:
    return np.exp(np.cumsum(log_rets))


def backtest_from_proba(proba, fwd_log_ret, cost, thr):
    sig = (proba > thr).astype(int)
    trades = np.abs(np.diff(np.r_[0, sig]))
    return sig * fwd_log_ret - trades * cost


def run(ticker: str, years: int, horizon_days: int):
    print("[INFO] štartujem…")
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(years * 365.25))
    df_raw = yf.download(ticker, start=start.isoformat(), end=None, auto_adjust=True, progress=False)
    if df_raw.empty:
        raise RuntimeError(f"Žiadne dáta pre {ticker} (interval {start} – {end}).")

    df = make_features(df_raw, horizon=horizon_days)
    FEATURES = ["ret", "ret_ma_5", "ret_ma_21", "vol_21", "rsi_14", "sma50", "sma200"]
    COST = 0.0005  # 0.05%

    X = df[FEATURES].values
    y_cls = df["target_up_next"].values
    y_reg = df["target_ret_next"].values

    X_train = X[:-1]
    y_train_cls = y_cls[:-1]
    y_train_reg = y_reg[:-1]
    X_today = X[-1:]

    # modely
    cls = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42))
    ])
    reg = Pipeline([("scaler", StandardScaler()), ("regr", Ridge(alpha=1.0))])

    # tuning prahu na konci tréningu
    train_df = df.iloc[:-1]
    VAL_DAYS = max(60, min(120, max(0, len(train_df) - 252)))
    pre_val_df = train_df.iloc[:-VAL_DAYS] if VAL_DAYS > 0 else train_df

    cls_tune = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42))
    ]).fit(pre_val_df[FEATURES].values, pre_val_df["target_up_next"].values)

    val_df = train_df.iloc[-VAL_DAYS:] if VAL_DAYS > 0 else train_df.iloc[-60:]
    proba_val = cls_tune.predict_proba(val_df[FEATURES].values)[:, 1]
    fwd_val   = val_df["target_ret_next"].values

    best_thr, best_total = 0.55, -9e9
    for thr in np.arange(0.50, 0.71, 0.01):
        res = backtest_from_proba(proba_val, fwd_val, COST, thr)
        total = np.exp(res.cumsum())[-1] - 1
        if total > best_total:
            best_total, best_thr = total, float(thr)
    print(f"[INFO] optimal threshold (validation): {best_thr:.2f} | val total: {best_total:.2%}")

    # finálny tréning a predikcia
    cls.fit(X_train, y_train_cls)
    reg.fit(X_train, y_train_reg)

    prob_up = float(cls.predict_proba(X_today)[:, 1][0])
    exp_log_ret = float(reg.predict(X_today)[0])
    exp_pct = (np.exp(exp_log_ret) - 1.0) * 100.0

    ENTER = max(best_thr + 0.04, 0.60)  # skús 0.60–0.62
    EXIT  = max(0.50, ENTER - 0.05)


    uptrend_today = df["sma50"].iloc[-1] > df["sma200"].iloc[-1]
    action = "LONG (nákup)" if (prob_up > ENTER and uptrend_today) else "CASH (neobchodovať)"

    # ročný backtest s hysterézou
    TEST_DAYS = min(252, max(10, len(df) - 21))
    test = df.iloc[-TEST_DAYS-1:-1]

    cls_loop = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42))
    ])

    signals, fwd_rets, strat_rets, proba_list = [], [], [], []
    prev_sig = 0
    min_hold = 5       # drž aspoň 5 dni
    cooldown_max = 3   # po výstupe pauza 3 dni
    hold_days = 0
    cooldown = 0

    for i in range(TEST_DAYS):
        end_idx = len(df) - TEST_DAYS - 1 + i
        train_i = df.iloc[:end_idx]

        # fit (ak chceš zrýchliť: refit len keď i==0 alebo i%5==0)
        cls_loop.fit(train_i[FEATURES].values, train_i["target_up_next"].values)

        # pravdepodobnosť ↑ pre dnešný krok
        x_i = df[FEATURES].iloc[end_idx:end_idx+1].values
        p_up = float(cls_loop.predict_proba(x_i)[:, 1][0])
        proba_list.append(p_up)

        # --- filtre ---
        uptrend = (df["sma50"].iloc[end_idx] > df["sma200"].iloc[end_idx] * 1.002)  # +0.2 %

        # vol threshold z minulých 252 dní (bez dneška), fallback ak málo dát
        past_vol = df["vol_21"].iloc[max(0, end_idx-252):end_idx]
        vol_q = past_vol.quantile(0.75) if len(past_vol) > 20 else df["vol_21"].quantile(0.80)
        hi_vol = df["vol_21"].iloc[end_idx] > vol_q

        # --- rozhodovanie (hysteréza + trendy + vol + min_hold/cooldown) ---
        if cooldown > 0:
            sig = 0
            cooldown -= 1
            hold_days = 0
        else:
            if prev_sig == 0:
                if (p_up > ENTER) and uptrend and not hi_vol:
                    sig = 1
                    hold_days = 1
                else:
                    sig = 0
            else:  # už sme v longu
                hold_days += 1
                should_exit = (p_up < EXIT) or (not uptrend) or hi_vol
                if should_exit and hold_days >= min_hold:
                    sig = 0
                    cooldown = cooldown_max
                    hold_days = 0
                else:
                    sig = 1

        # výnos „zajtra“ (alebo tvoj target_ret_next pre horizon=1)
        fwd = float(df["target_ret_next"].iloc[end_idx])

        # náklad len pri zmene pozície
        trade_cost = (COST if sig != prev_sig else 0.0)
        strat = sig * fwd - trade_cost

        signals.append(sig)
        fwd_rets.append(fwd)
        strat_rets.append(strat)
        prev_sig = sig

    fwd_rets = np.array(fwd_rets)
    strat_rets = np.array(strat_rets)
    eq_model = equity_from_log_returns(strat_rets)
    eq_bh = equity_from_log_returns(fwd_rets)

    total_model = eq_model[-1] - 1.0
    total_bh = eq_bh[-1] - 1.0
    hit_rate = (
        ((np.array(signals) == 1) & (np.array(fwd_rets) > 0)).sum()
        / max(1, (np.array(signals) == 1).sum())
    )

    print("\n=== PREDIKCIA ===")
    print(f"TICKER: {ticker} | História: {years}y | Horizont: {horizon_days}d")
    print(f"P(↑): {prob_up:.2%} | Očak. log-ret: {exp_log_ret:.5f} (~{exp_pct:.2f} %)")
    print(f"Sugescia: {action} | ENTER={ENTER:.2f}, EXIT={EXIT:.2f}")

    print("\n=== RÝCHLY BACKTEST ~1 rok ===")
    print(f"Model (netto, náklady {COST:.3%}): {total_model:.2%}")
    print(f"Buy & Hold: {total_bh:.2%}")
    print(f"Hit-rate pri vstupoch: {hit_rate:.2%} | Dní v trhu: {(np.array(signals)==1).sum()} z {len(signals)}")

    # ukladanie
    OUT = (Path(__file__).parent / "outputs" / ticker)
    OUT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "as_of": [df.index[-1]],
        "ticker": [ticker],
        "prob_up": [prob_up],
        "exp_log_ret": [exp_log_ret],
        "exp_pct": [exp_pct],
        "suggestion": [action],
        "threshold_enter": [ENTER],
        "threshold_exit": [EXIT],
        "horizon_days": [horizon_days]
    }).to_csv(OUT / f"prediction_{ticker}.csv", index=False)

    pd.DataFrame({
        "date": test.index,
        "fwd_log_ret": fwd_rets,
        "signal": signals,
        "strat_log_ret_net": strat_rets,
        "equity_model": eq_model,
        "equity_bh": eq_bh,
        "prob_up": proba_list
    }).to_csv(OUT / f"backtest_{ticker}.csv", index=False)

    # GRAFY (všetko vo vnútri run)
    plt.figure()
    plt.plot(test.index, eq_model, label="Model (netto)")
    plt.plot(test.index, eq_bh,    label="Buy & Hold")
    plt.title(f"Equity – {ticker}  ({years}y, horizon={horizon_days}d)")
    plt.xlabel("Dátum"); plt.ylabel("Relatívna hodnota")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / f"equity_{ticker}.png", dpi=160)

    try:
        close_test = df_raw["Close"].loc[test.index]
    except KeyError:
        close_test = df_raw["Close"].iloc[-len(test):]
    sig_arr = np.array(signals, dtype=int)
    plt.figure()
    plt.plot(close_test.index, close_test.values, label="Close")
    long_idx = close_test.index[sig_arr == 1]
    long_px  = close_test.values[sig_arr == 1]
    if len(long_idx) > 0:
        plt.scatter(long_idx, long_px, marker="^", s=18, label="LONG", zorder=3)
    plt.title(f"Cena a LONG signály – {ticker}")
    plt.xlabel("Dátum"); plt.ylabel("Cena"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / f"price_signals_{ticker}.png", dpi=160)

    if len(proba_list) > 0:
        plt.figure()
        plt.hist(proba_list, bins=20)
        plt.title(f"Rozdelenie P(↑) – {ticker}")
        plt.xlabel("P(↑) z klasifikátora"); plt.ylabel("Počet dní")
        plt.tight_layout(); plt.savefig(OUT / f"proba_hist_{ticker}.png", dpi=160)

    print(f"\n[OK] Súbory uložené do: {OUT.resolve()}")
    # plt.show()  # odkomentuj, ak chceš okná s grafmi


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI predikcia akcií (RandomForest + tuning prahu + hysteréza).")
    ap.add_argument("--ticker", type=str, default="NVDA")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()
    run(args.ticker.upper(), args.years, args.horizon)
