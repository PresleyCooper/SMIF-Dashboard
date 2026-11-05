"""
smif/ledger.py — transactions in, share/cash events out.

One row of the Schwab export becomes one event carrying up to five effects:

    dq        change in share count for `sym`
    dcash     change in the sweep cash balance
    flow      the part of dcash that crossed the fund's boundary (Deposit /
              Withdrawal only) — this is what a time-weighted return has to
              neutralise
    income    dividends, interest and return of principal, net of foreign tax
              and ADR fees, attributed to the sleeve that earned it
    netbuy    buys minus sells, attributed to the sleeve that traded

`flow`, `income` and `netbuy` are deliberately *views* of dcash rather than
independent quantities: every dollar that moves is already in dcash, and these
columns only say which return formula is entitled to neutralise it. Keeping
them separate is what lets the total portfolio ignore an internal dividend
(cash just moved from one pocket to another) while the equity sleeve, which
cannot see the cash pocket, has to add it back.

The two rules that are easy to get wrong and silently wreck the history:

  · A DRIP shows up as a dividend row AND a matching small Buy. Both are
    processed as written — the dividend adds cash, the Buy spends it and adds
    shares — which nets to the right place. The two "Reinvested" activities
    that carry a Quantity are the exception: those add shares directly and no
    cash ever appears, so treating them as cash income would invent money.

  · Spin-offs, mergers and renames arrive as paired Credit/Debit Security rows
    whose Amount is COST BASIS, not market value. They are therefore share-only
    events. Dividing that Amount by Quantity would produce a confident, wildly
    wrong price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EVENT_COLUMNS = ["date", "sym", "dq", "dcash", "flow", "income", "netbuy", "activity"]


def resolve_symbol(symbol, trade_date, activity: str = "") -> str | None:
    """
    The symbol a row's *economics* belong to.

    Only one name needs rewriting: the ticker GOLD is Randgold up to the
    2019-01-02 Barrick exchange (including that day's Debit Security, which is
    the Randgold side of the swap) and Barrick after it. Yahoo's present-day
    GOLD is a third, unrelated company, so the old position gets its own
    synthetic symbol and is priced off Barrick at the exchange ratio.
    """
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
        return None
    s = str(symbol).strip().upper()
    if s != "GOLD":
        return s
    d = pd.Timestamp(trade_date)
    if d < config.RANDGOLD_SWITCH:
        return config.RANDGOLD_SYMBOL
    if d == config.RANDGOLD_SWITCH and str(activity).startswith("Debit Security"):
        return config.RANDGOLD_SYMBOL
    return s


def description_symbol_map(tx: pd.DataFrame) -> dict[str, str]:
    """
    Description -> symbol, learned from the rows that do carry a symbol.

    Foreign-tax and ADR-fee rows often arrive with an empty Symbol but the same
    Description as the dividend they belong to ("Asml Hldgs Nv"). Without this
    map those expenses would float free of any sleeve and the equity sleeve's
    income would be overstated by the tax withheld.
    """
    known = tx.dropna(subset=["Symbol"])
    if known.empty:
        return {}
    return (known.groupby("Description")["Symbol"]
            .agg(lambda s: s.mode().iloc[0]).to_dict())


def _effects(activity: str, sym: str | None, qty: float, amt: float):
    """(dq, dcash, flow) for one row. Raises on an activity it does not know."""
    a = activity
    if a == "Deposit":
        return 0.0, amt, amt
    if a == "Withdrawal":
        return 0.0, -amt, -amt
    if a == "Buy":
        # A Buy of CASH is the sweep moving money between the settlement fund
        # and itself: it is bookkeeping, not a trade.
        return (0.0, 0.0, 0.0) if sym == "CASH" else (qty, -amt, 0.0)
    if a == "Sell":
        return (0.0, 0.0, 0.0) if sym == "CASH" else (-qty, amt, 0.0)
    if a == "Income (Reinvested Qualified Dividend)":
        return qty, 0.0, 0.0
    if a == "Income (Reinvested Dividend)":
        # Reinvested into the security itself (shares, no cash) unless the
        # "security" is the cash sweep, in which case it is interest.
        return (0.0, amt, 0.0) if sym == "CASH" else (qty, 0.0, 0.0)
    if a.startswith("Income"):
        return 0.0, amt, 0.0
    if a.startswith("Expense"):
        # A negative Amount here is a reversal of an earlier fee; keeping the
        # sign math rather than taking an absolute value makes it refund cash.
        return 0.0, -amt, 0.0
    if a == "Return of Principal":
        return 0.0, amt, 0.0
    if a == "Split":
        # Schwab records the shares ADDED, not the ratio: a 2-for-1 on 100
        # shares is a Split row with Quantity 100.
        return qty, 0.0, 0.0
    if a.startswith("Credit Security"):
        return qty, 0.0, 0.0
    if a.startswith("Debit Security"):
        return -qty, 0.0, 0.0
    raise ValueError(f"unknown Activity: {activity!r}")


def build_events(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Replay the ledger oldest-first into per-transaction events.

    `tx` is the canonical ledger in Schwab column names. The result is sorted by
    date and carries one row per transaction, including rows with no effect at
    all (cash sweeps), so an event frame can always be traced back to a trade.
    """
    if tx is None or tx.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    t = tx.copy()
    t["Trade Date"] = pd.to_datetime(t["Trade Date"])
    t = t.sort_values(["Trade Date"] + (["id"] if "id" in t.columns else []))
    t["Activity"] = t["Activity"].astype(str).str.strip()

    dmap = description_symbol_map(t)
    filled = t["Symbol"].where(t["Symbol"].notna(), t["Description"].map(dmap))

    rows = []
    for (_, r), sym_raw in zip(t.iterrows(), filled):
        act = r["Activity"]
        qty = 0.0 if pd.isna(r.get("Quantity")) else float(r["Quantity"])
        amt = 0.0 if pd.isna(r.get("Amount")) else float(r["Amount"])
        sym = resolve_symbol(sym_raw, r["Trade Date"], act)
        dq, dcash, flow = _effects(act, sym, qty, amt)

        # income: what the sleeve's assets earned, net of what was withheld.
        # The two share-paying "Reinvested" activities are excluded — they
        # produced shares, and dq above already carries them.
        income = 0.0
        if act.startswith("Income"):
            share_paying = (act == "Income (Reinvested Qualified Dividend)" or
                            (act == "Income (Reinvested Dividend)" and sym != "CASH"))
            if not share_paying:
                income = amt
        elif act == "Return of Principal":
            income = amt
        elif act.startswith("Expense"):
            income = -amt

        # netbuy: cash committed to (or released from) this sleeve by trading.
        # Sweep rows carry no symbol economics and are left out.
        netbuy = 0.0
        if sym != "CASH":
            if act == "Buy":
                netbuy = amt
            elif act == "Sell":
                netbuy = -amt

        rows.append((pd.Timestamp(r["Trade Date"]), sym, dq, dcash, flow,
                     income, netbuy, act))

    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return ev.reset_index(drop=True)


def final_positions(events: pd.DataFrame) -> pd.Series:
    """Share count per symbol after the whole history, tiny residuals dropped."""
    if events.empty:
        return pd.Series(dtype=float)
    q = events.dropna(subset=["sym"]).groupby("sym")["dq"].sum()
    return q[q.abs() > 1e-6].sort_index()


def final_cash(events: pd.DataFrame) -> float:
    return float(events["dcash"].sum()) if not events.empty else 0.0


def traded_symbols(events: pd.DataFrame) -> list[str]:
    """Every symbol that ever had a share effect — what prices are needed for."""
    if events.empty:
        return []
    moved = events.dropna(subset=["sym"])
    moved = moved[moved["dq"].abs() > 0]
    return sorted(set(moved["sym"]) - {"CASH"})


def fill_anchors(tx: pd.DataFrame, symbol: str) -> pd.Series:
    """
    The fund's own execution prices for one symbol, Amount / Quantity.

    Used both to validate rebuilt closes and to price the names Yahoo has no
    history for. Fractional DRIP top-ups are excluded: a 0.3-share reinvestment
    carries rounding noise large enough to move a log interpolation.
    """
    t = tx.copy()
    t["Trade Date"] = pd.to_datetime(t["Trade Date"])
    t["sym"] = [resolve_symbol(s, d, a) for s, d, a
                in zip(t["Symbol"], t["Trade Date"], t["Activity"])]
    m = ((t["sym"] == symbol) & t["Activity"].isin(["Buy", "Sell"])
         & (t["Quantity"].fillna(0) >= config.ANCHOR_MIN_QUANTITY)
         & (t["Amount"].fillna(0) > 0))
    f = t[m]
    if f.empty:
        return pd.Series(dtype=float)
    px = f["Amount"] / f["Quantity"]
    return px.groupby(f["Trade Date"]).mean().sort_index()


def external_flows(events: pd.DataFrame) -> pd.Series:
    """Deposits and withdrawals by date — the only flows the fund did not earn."""
    if events.empty:
        return pd.Series(dtype=float)
    f = events[events["flow"] != 0]
    return f.groupby("date")["flow"].sum()


def to_trading_day(dates, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Snap each date forward to the next trading day.

    A handful of trade dates in the export fall on weekends and holidays
    (settlement artefacts, mostly in the early years). Dropping them would lose
    shares; snapping backward would value a position before it existed.
    """
    idx = np.minimum(trading_days.searchsorted(pd.DatetimeIndex(dates)),
                     len(trading_days) - 1)
    return trading_days[idx]
