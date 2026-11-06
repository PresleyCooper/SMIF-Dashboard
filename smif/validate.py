"""
smif/validate.py — the checks that decide whether the rebuild is trustworthy.

There are four, and each one catches a different class of error:

  share counts      every transaction rule applied in the right direction
  sleeve values     the share counts AND the prices are right on a day the
                    custodian also valued
  fill prices       the rebuilt closes are the real, un-split-adjusted ones
  cash              nothing in the cash ledger is missing

The cash check is expected to FAIL by about $2,040 and is reported rather than
corrected. The rebuilt cash balance runs above the custodian's, which means the
export is missing an outflow — most likely a single fee or transfer that was
never exported. Plugging it with a balancing entry would make the reconciliation
green and the history wrong, and would hide the discrepancy from whoever
eventually finds the missing row. So it is surfaced, sized, and left alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, ledger


def check_share_counts(book, holdings: pd.DataFrame,
                       tolerance: float = 0.01) -> pd.DataFrame:
    """
    Rebuilt final share counts against the Portfolio sheet.

    Sub-share residues on long-exited names are reported but do not fail: a
    0.4-share DRIP remnant from 2012 that the custodian has since swept is a
    bookkeeping leftover, not a broken rule.
    """
    rebuilt = book.shares.iloc[-1]
    rebuilt = rebuilt[rebuilt.abs() > 1e-6]
    actual = holdings.set_index("symbol")["quantity"]

    rows = []
    for sym in sorted(set(rebuilt.index) | set(actual.index)):
        r = float(rebuilt.get(sym, 0.0))
        a = float(actual.get(sym, 0.0))
        diff = r - a
        rows.append({"symbol": sym, "rebuilt": r, "workbook": a, "diff": diff,
                     "ok": abs(diff) <= tolerance,
                     "sub_share": abs(diff) < 1.0 and a == 0.0})
    out = pd.DataFrame(rows)
    return out.sort_values(["ok", "symbol"]).reset_index(drop=True)


def check_sleeve_values(book, subtotals: pd.Series,
                        as_of=None) -> pd.DataFrame:
    """Sleeve values on the custodian's snapshot date against its subtotals."""
    as_of = pd.Timestamp(as_of) if as_of is not None else config.CUSTODIAN_AS_OF
    day = book.days[book.days <= as_of][-1]

    equity_target = (float(subtotals.get("Equity", np.nan))
                     + float(subtotals.get("Real Estate", 0.0)))
    targets = {
        "equity": ("Equity + Real Estate", equity_target),
        "fixed_income": ("Fixed", float(subtotals.get("Fixed", np.nan))),
        "cash": ("Cash (sweep + SNOXX)", float(subtotals.get("Cash", np.nan))),
    }
    rows = []
    for sleeve, (label, target) in targets.items():
        rebuilt = float(book.sleeve_value[sleeve].loc[day])
        rows.append({"sleeve": config.SLEEVE_LABEL[sleeve], "custodian_line": label,
                     "rebuilt": rebuilt, "custodian": target,
                     "diff": rebuilt - target})
    total_target = float(subtotals.get("Total", np.nan))
    rows.append({"sleeve": "Total portfolio", "custodian_line": "Total",
                 "rebuilt": float(book.total.loc[day]), "custodian": total_target,
                 "diff": float(book.total.loc[day]) - total_target})
    out = pd.DataFrame(rows)
    out["as_of"] = day
    return out


def check_fill_prices(book, tx: pd.DataFrame,
                      tolerance: float | None = None) -> pd.DataFrame:
    """
    Every Buy and Sell fill against the rebuilt close on its trade date.

    This is the test that actually proves the split un-adjust works: a fill is
    an observed, real-money price for a real share, so if the rebuilt series
    were still split-adjusted the ratio would come out at exactly the split
    factor and stand out immediately.
    """
    tol = config.FILL_PRICE_TOLERANCE if tolerance is None else tolerance
    t = tx.copy()
    t["Trade Date"] = pd.to_datetime(t["Trade Date"])
    t["sym"] = [ledger.resolve_symbol(s, d, a) for s, d, a
                in zip(t["Symbol"], t["Trade Date"], t["Activity"])]
    m = (t["Activity"].isin(["Buy", "Sell"])
         & (t["Quantity"].fillna(0) >= 1.0)
         & (t["Amount"].fillna(0) > 0)
         & ~t["sym"].isin(config.CASH_LIKE))
    f = t[m].copy()
    f["fill"] = f["Amount"] / f["Quantity"]

    days = book.days
    rows = []
    for _, r in f.iterrows():
        sym = r["sym"]
        d = r["Trade Date"]
        close = np.nan
        if sym in book.prices.columns:
            on_or_before = days[days <= d]
            if len(on_or_before):
                close = float(book.prices.loc[on_or_before[-1], sym])
        interpolated = sym in config.NO_YAHOO_DATA
        ratio = close / r["fill"] if (close == close and r["fill"]) else np.nan
        rows.append({"symbol": sym, "date": d, "activity": r["Activity"],
                     "quantity": float(r["Quantity"]), "fill": float(r["fill"]),
                     "close": close, "ratio": ratio,
                     "interpolated": interpolated,
                     "within_tol": bool(abs(ratio - 1) < tol) if ratio == ratio else False})
    return pd.DataFrame(rows)


def fill_price_summary(checks: pd.DataFrame,
                       tolerance: float | None = None) -> dict:
    """Headline numbers for the data-quality page."""
    tol = config.FILL_PRICE_TOLERANCE if tolerance is None else tolerance
    priced = checks[~checks["interpolated"] & checks["ratio"].notna()]
    n = len(priced)
    within = int(priced["within_tol"].sum())
    return {
        "checked": n,
        "within_tolerance": within,
        "share": within / n if n else float("nan"),
        "tolerance": tol,
        "skipped_interpolated": int(checks["interpolated"].sum()),
        "no_price": int(checks["ratio"].isna().sum()),
        "worst": (priced.reindex((priced["ratio"] - 1).abs()
                                 .sort_values(ascending=False).index)
                  .head(15).reset_index(drop=True)),
    }


def check_cash(book, subtotals: pd.Series, as_of=None) -> dict:
    """The known gap: rebuilt cash against the custodian's cash + money market."""
    as_of = pd.Timestamp(as_of) if as_of is not None else config.CUSTODIAN_AS_OF
    day = book.days[book.days <= as_of][-1]
    rebuilt = float(book.sleeve_value["cash"].loc[day])
    target = float(subtotals.get("Cash", np.nan))
    return {"as_of": day, "rebuilt": rebuilt, "custodian": target,
            "diff": rebuilt - target,
            "share_of_total": (rebuilt - target) / float(book.total.loc[day])}


def interpolated_positions(book) -> pd.DataFrame:
    """Current positions whose price is interpolated rather than observed."""
    h = book.holdings_on()
    rows = [{"symbol": s, "value": float(h.loc[s, "value"]),
             "note": book.price_notes.get(s, "")}
            for s in h.index
            if s in config.NO_YAHOO_DATA or "interpolat" in book.price_notes.get(s, "")]
    return pd.DataFrame(rows)


COLUMN_ALIASES = {
    "symbol": {"symbol", "ticker", "sym", "security", "security symbol"},
    "quantity": {"quantity", "shares", "qty", "units", "share quantity"},
    "price": {"price", "close", "last price", "market price", "unit price"},
    "value": {"value", "market value", "mkt value", "amount", "position value"},
}


def read_position_snapshot(source, sheet=None) -> pd.DataFrame:
    """
    Read an uploaded custodian position file into symbol / quantity / price /
    value.

    Column names are matched case-insensitively against a small alias table
    rather than fixed, because every custodian spells this differently and the
    alternative is telling a treasurer to rename columns before the app will
    look at their file. Rows without a symbol (subtotals, totals, headers that
    survived an export) are dropped: they are not positions.
    """
    if hasattr(source, "name") and str(source.name).lower().endswith(
            (".csv", ".txt")):
        df = pd.read_csv(source)
    else:
        df = pd.read_excel(source, sheet_name=sheet or 0)
    df.columns = [str(c).strip() for c in df.columns]

    found = {}
    for target, names in COLUMN_ALIASES.items():
        for c in df.columns:
            if c.strip().lower() in names:
                found[target] = c
                break
    if "symbol" not in found:
        raise ValueError(
            "No symbol column found. Expected one of: "
            + ", ".join(sorted(COLUMN_ALIASES["symbol"])))
    if "quantity" not in found and "value" not in found:
        raise ValueError(
            "Need a quantity or a value column to reconcile against.")

    out = pd.DataFrame({"symbol": df[found["symbol"]]})
    for target in ("quantity", "price", "value"):
        out[target] = (pd.to_numeric(df[found[target]], errors="coerce")
                       if target in found else np.nan)
    out["symbol"] = out["symbol"].astype("string").str.strip().str.upper()
    out = out[out["symbol"].notna() & (out["symbol"] != "")]
    out = out[~out["symbol"].str.contains("TOTAL|SUBTOTAL", na=False)]
    if "value" in found and "quantity" in found and out["value"].isna().all():
        out["value"] = out["quantity"] * out["price"]
    return out.reset_index(drop=True)


def reconcile_positions(book, snapshot: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """
    Rebuilt positions against an uploaded snapshot, position by position.

    Lines present on only one side are kept with zero on the other, because a
    position the fund thinks it holds and the custodian does not is the single
    most important row in the table and dropping it would hide it.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else book.days[-1]
    day = book.days[book.days <= as_of][-1]

    rebuilt = book.holdings_on(day)
    shares = rebuilt["shares"].to_dict()
    values = rebuilt["value"].to_dict()

    # The sweep balance is a cash POSITION to a custodian and a separate
    # balance to the engine. Without it on the rebuilt side the comparison is
    # not like for like, and the custodian's CASH line reconciles against
    # nothing while reporting a share difference of zero.
    sweep = float(book.cash.loc[day])
    if abs(sweep) > 1e-9:
        shares["CASH"] = shares.get("CASH", 0.0)
        values["CASH"] = values.get("CASH", 0.0) + sweep

    snap = snapshot.set_index("symbol")
    rows = []
    for sym in sorted(set(shares) | set(snap.index)):
        r_q = float(shares.get(sym, 0.0))
        r_v = float(values.get(sym, 0.0))
        s_q = float(snap["quantity"].get(sym, float("nan")))
        s_v = float(snap["value"].get(sym, float("nan")))
        if np.isnan(s_v) and not np.isnan(s_q) and sym in snap.index:
            price = float(snap["price"].get(sym, float("nan")))
            s_v = s_q * price if not np.isnan(price) else float("nan")

        share_diff = r_q - (0.0 if np.isnan(s_q) else s_q)
        value_diff = r_v - (0.0 if np.isnan(s_v) else s_v)
        rows.append({
            "symbol": sym,
            "rebuilt_shares": r_q,
            "custodian_shares": s_q,
            "share_diff": share_diff,
            "rebuilt_value": r_v,
            "custodian_value": s_v,
            "value_diff": value_diff,
            "only_in": ("rebuild" if sym not in snap.index
                        else "snapshot" if sym not in shares else ""),
        })
    out = pd.DataFrame(rows)
    # A line matches only if BOTH the share count and the value agree. Share
    # count alone would pass a cash position — which has no share count on
    # either side — however far apart the two dollar figures were.
    #
    # The value test is relative, not absolute: the custodian prices off a
    # different feed, so an eleven-cent disagreement on a $16,000 position is
    # noise and flagging it buries the lines that matter. A dollar floor keeps
    # small positions from passing on a percentage.
    tolerance = np.maximum(1.0, 0.0005 * out["custodian_value"].abs().fillna(0.0))
    out["matches"] = ((out["share_diff"].abs() < 1e-4)
                      & (out["value_diff"].abs() <= tolerance))
    out["as_of"] = day
    return out.sort_values(["matches", "symbol"]).reset_index(drop=True)


def report(book, tx: pd.DataFrame, holdings: pd.DataFrame,
           subtotals: pd.Series) -> dict:
    """Everything above in one dict — what the Methodology page renders."""
    shares = check_share_counts(book, holdings)
    fills = check_fill_prices(book, tx)
    return {
        "shares": shares,
        "share_failures": shares[~shares["ok"] & ~shares["sub_share"]],
        "sleeves": check_sleeve_values(book, subtotals),
        "cash": check_cash(book, subtotals),
        "fills": fills,
        "fill_summary": fill_price_summary(fills),
        "price_notes": book.price_notes,
        "interpolated": interpolated_positions(book),
    }
