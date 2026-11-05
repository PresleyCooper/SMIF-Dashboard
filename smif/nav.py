"""
smif/nav.py — the daily book: shares, prices, sleeve values, cash and flows.

This is where the ledger stops being a list of trades and becomes a time series.
The output is one object holding every daily quantity the return math needs, so
nothing downstream has to re-read the ledger or re-derive a share count.

The one modelling choice worth defending: EVENTS ON NON-TRADING DAYS ARE SNAPPED
FORWARD. A few dozen rows in the export are dated to weekends and holidays —
settlement artefacts, mostly pre-2015. Dropping them would lose shares and
break the reconciliation; snapping backward would value a position on a day it
did not exist. Snapping forward delays the recognition by one session, which is
the smallest possible distortion and never creates a phantom holding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, ledger, prices as price_mod


def sleeve_of(symbol: str, overrides: dict[str, str] | None = None) -> str:
    """
    Which sleeve a symbol belongs to.

    Config wins over user classification for the names the fund has decided
    about (cash-like instruments and the bond funds in config.FIXED_INCOME);
    anything the user has classified comes next; everything else is equity.
    That ordering means a bond fund added to the config file cannot be
    accidentally overridden by a stale classification in the database.
    """
    s = str(symbol).strip().upper()
    if s in config.CASH_LIKE:
        return "cash"
    if s in config.FIXED_INCOME:
        return "fixed_income"
    ov = (overrides or {}).get(s)
    if ov in config.SLEEVES:
        return ov
    return "equity"


@dataclass
class Book:
    """Everything the return engine needs, all on one trading calendar."""
    days: pd.DatetimeIndex
    shares: pd.DataFrame                  # days x symbol, real share counts
    prices: pd.DataFrame                  # days x symbol, real closes
    mv: pd.DataFrame                      # days x symbol, market value
    cash: pd.Series                       # sweep cash balance
    total: pd.Series                      # securities + cash
    flows: pd.Series                      # external deposits/withdrawals
    sleeve_value: dict[str, pd.Series]
    sleeve_income: dict[str, pd.Series]
    sleeve_netbuy: dict[str, pd.Series]
    sleeve_members: dict[str, list[str]]
    price_notes: dict[str, str] = field(default_factory=dict)
    events: pd.DataFrame | None = None

    @property
    def asof(self) -> pd.Timestamp:
        return self.days[-1]

    def securities_value(self) -> pd.Series:
        return self.mv.sum(axis=1)

    def holdings_on(self, when=None) -> pd.DataFrame:
        """Share count, price and value per symbol on one day, zeros dropped."""
        d = self.days[-1] if when is None else self.days[self.days <= pd.Timestamp(when)][-1]
        q = self.shares.loc[d]
        q = q[q.abs() > 1e-6]
        out = pd.DataFrame({"shares": q,
                            "price": self.prices.loc[d, q.index],
                            "value": self.mv.loc[d, q.index]})
        out["sleeve"] = [sleeve_of(s) for s in out.index]
        out["weight"] = out["value"] / out["value"].sum()
        return out.sort_values("value", ascending=False)

    def cash_weight(self) -> pd.Series:
        return (self.sleeve_value["cash"] / self.total).replace([np.inf, -np.inf], np.nan)


def _snap(series: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Sum a date-indexed series onto the trading calendar, zero-filled."""
    if series.empty:
        return pd.Series(0.0, index=days)
    snapped = ledger.to_trading_day(series.index, days)
    return series.groupby(snapped).sum().reindex(days, fill_value=0.0)


def build_book(tx: pd.DataFrame, store: price_mod.PriceStore, *,
               end=None, overrides: dict[str, str] | None = None) -> Book:
    """
    Replay the ledger onto a daily trading calendar.

    `end` caps the history (the tests pin it so a later Yahoo pull cannot move a
    regression number). Events after `end` are dropped rather than snapped, so
    capping the window never piles future trades onto the last day.
    """
    events = ledger.build_events(tx)
    if events.empty:
        raise ValueError("the ledger is empty — nothing to value")

    end = pd.Timestamp(end) if end is not None else None
    days = price_mod.trading_days(store, start=config.INCEPTION, end=end)
    if len(days) == 0:
        raise ValueError("no trading calendar — fetch SPY prices first")
    events = events[events["date"] <= days[-1]].copy()

    # -- share counts ----------------------------------------------------------
    moved = events.dropna(subset=["sym"])
    moved = moved[moved["dq"].abs() > 0]
    if moved.empty:
        shares = pd.DataFrame(index=days)
    else:
        td = ledger.to_trading_day(moved["date"], days)
        shares = (moved.assign(td=td)
                  .groupby(["td", "sym"])["dq"].sum()
                  .unstack(fill_value=0.0)
                  .reindex(days, fill_value=0.0)
                  .cumsum())
        # Float accumulation leaves 1e-13 ghosts on fully exited positions.
        shares[shares.abs() < 1e-6] = 0.0

    cash = _snap(events.groupby("date")["dcash"].sum(), days).cumsum()
    flows = _snap(events.groupby("date")["flow"].sum(), days)

    # -- prices ----------------------------------------------------------------
    symbols = [c for c in shares.columns if shares[c].abs().sum() > 0]
    P, notes = price_mod.build_price_matrix(symbols, days, tx, store, held=shares)
    cols = [c for c in P.columns if c in shares.columns]
    mv = shares[cols] * P[cols]

    # -- sleeves ---------------------------------------------------------------
    members: dict[str, list[str]] = {s: [] for s in config.SLEEVES}
    for c in cols:
        members[sleeve_of(c, overrides)].append(c)

    ev = events.copy()
    ev["sleeve"] = [sleeve_of(s, overrides) if s else None for s in ev["sym"]]
    ev["td"] = ledger.to_trading_day(ev["date"], days)

    value, income, netbuy = {}, {}, {}
    for s in config.SLEEVES:
        sub = ev[ev["sleeve"] == s]
        v = mv[members[s]].sum(axis=1) if members[s] else pd.Series(0.0, index=days)
        inc = (sub.groupby("td")["income"].sum().reindex(days, fill_value=0.0)
               if not sub.empty else pd.Series(0.0, index=days))
        if s == "cash":
            # Cash is the sleeve every other sleeve settles through, so its
            # "net purchases" are every dollar that crossed its boundary:
            # total cash movement, plus money-market shares bought or
            # transferred in at $1, minus the interest it earned itself (which
            # is a return, not a transfer).
            v = v + cash
            mm = [c for c in members[s] if c in config.MONEY_MARKET]
            mm_delta = (shares[mm].sum(axis=1).diff().fillna(shares[mm].sum(axis=1).iloc[0])
                        if mm else pd.Series(0.0, index=days))
            nb = _snap(events.groupby("date")["dcash"].sum(), days) + mm_delta - inc
            # Income earned inside the cash sleeve never leaves it, so there is
            # nothing to add back on the numerator.
            inc_out = pd.Series(0.0, index=days)
        else:
            nb = (sub.groupby("td")["netbuy"].sum().reindex(days, fill_value=0.0)
                  if not sub.empty else pd.Series(0.0, index=days))
            inc_out = inc
        value[s] = v
        income[s] = inc_out
        netbuy[s] = nb

    total = mv.sum(axis=1) + cash

    return Book(days=days, shares=shares, prices=P, mv=mv, cash=cash, total=total,
                flows=flows, sleeve_value=value, sleeve_income=income,
                sleeve_netbuy=netbuy, sleeve_members=members,
                price_notes=notes, events=events)
