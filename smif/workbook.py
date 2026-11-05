"""
smif/workbook.py — the three things the engine reads out of SMIF Portfolio.xlsx.

  · Portfolio sheet  — current holdings, share counts and GICS sector/industry.
    This is the reconciliation target for final share counts and the source of
    sector labels for the sector correlation view.

  · Backend sheet, columns L onward — a custodian snapshot (as-of 2026-09-22):
    every position's quantity, price and value, plus subtotals by asset class.
    This is the reconciliation target for sleeve values.

  · Data sheet — S&P 500 constituents with index weights and GICS sectors, the
    reference universe the factor z-scores are computed against.

Note on the Backend sheet: its two right-most columns (Sector / Industry) are
shifted relative to the rows they sit beside — AAPL's row carries Nvidia's
industry. They are ignored here, and sector labels come from the Portfolio
sheet, which is aligned. Reading them anyway would mislabel most of the book.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def _wb_path(path=None) -> Path:
    p = Path(path) if path else config.SEED_WORKBOOK
    if not p.exists():
        raise FileNotFoundError(
            f"workbook not found at {p} — put SMIF Portfolio.xlsx in {config.SEED_DIR}")
    return p


def read_holdings(path=None) -> pd.DataFrame:
    """
    The Portfolio sheet: symbol, quantity, GICS sector and industry.

    Subtotal rows ('Equity', 'Fixed Income', 'Cash') carry no ticker and are
    dropped; SNOXX has a ticker and is kept, since the cash sleeve holds it.
    """
    df = pd.read_excel(_wb_path(path), sheet_name="Portfolio")
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    # The sheet has two columns both literally named "GICS Industry"; pandas
    # de-duplicates the second to "GICS Industry.1". The first is the sector.
    sector_col = cols[14] if len(cols) > 14 else None
    industry_col = cols[15] if len(cols) > 15 else None
    out = pd.DataFrame({
        "name": df[cols[0]],
        "symbol": df["Ticker"].astype("string").str.strip().str.upper(),
        "quantity": pd.to_numeric(df["Quantity"], errors="coerce"),
        "sector": df[sector_col] if sector_col else None,
        "industry": df[industry_col] if industry_col else None,
    })
    out = out.dropna(subset=["symbol", "quantity"])
    return out.reset_index(drop=True)


def read_custodian_snapshot(path=None) -> tuple[pd.Timestamp, pd.DataFrame, pd.Series]:
    """
    The Backend snapshot: (as-of date, per-position frame, asset-class subtotals).

    Positions are the rows that carry a Symbol; the Subtotal / Total rows become
    the subtotal series. Both are returned because they reconcile different
    things: positions check share counts, subtotals check sleeve values.
    """
    # Columns are named positionally, not by header text: the sheet has a
    # second "Price" column outside this block, and pandas de-duplicates header
    # names across the whole row before the usecols subset is applied, so the
    # snapshot's own "Price" arrives as "Price.1".
    names = ["As of Date", "Data For", "Managed", "Asset Class", "Symbol",
             "Description", "Open Date", "Quantity", "Price", "Cost Basis",
             "Value", "Average Purchase Price", "Unrealized", "Weight",
             "Sector", "Industry"]
    raw = pd.read_excel(_wb_path(path), sheet_name="Backend", header=None,
                        skiprows=1, usecols="L:AA", names=names)
    need = ["As of Date", "Asset Class", "Symbol", "Description",
            "Quantity", "Price", "Value"]

    as_of = pd.to_datetime(raw["As of Date"], errors="coerce").dropna().max()
    df = raw[need].copy()
    df["As of Date"] = pd.to_datetime(df["As of Date"], errors="coerce")
    df = df[df["As of Date"] == as_of]

    subtotal_rows = df[df["Symbol"].isna()]
    subtotals = {}
    for _, r in subtotal_rows.iterrows():
        label = str(r["Description"]).strip()
        key = str(r["Asset Class"]).strip() if pd.notna(r["Asset Class"]) else label
        if label == "Subtotal":
            subtotals[key] = float(r["Value"])
        elif label in ("Total", "Managed Assets Total"):
            subtotals.setdefault("Total", float(r["Value"]))

    pos = df[df["Symbol"].notna()].copy()
    pos["Symbol"] = pos["Symbol"].astype(str).str.strip().str.upper()
    pos = pos.rename(columns={"Symbol": "symbol", "Asset Class": "asset_class",
                              "Description": "name", "Quantity": "quantity",
                              "Price": "price", "Value": "value"})
    pos = pos[["symbol", "asset_class", "name", "quantity", "price", "value"]]
    for c in ("quantity", "price", "value"):
        pos[c] = pd.to_numeric(pos[c], errors="coerce")
    return as_of, pos.reset_index(drop=True), pd.Series(subtotals, dtype=float)


def read_sp500(path=None) -> pd.DataFrame:
    """
    S&P 500 constituents from the Data sheet: symbol, name, weight, sector.

    Tickers arrive in Bloomberg form ('NVDA UW Equity', 'BF/B UN Equity'), so
    the exchange suffix is stripped and the class separator '/' becomes the '-'
    Yahoo uses. Weights are normalised to sum to 1 — the sheet's own total is
    1.00000007, and a factor exposure weighted by something that is not quite a
    distribution is a quiet bias.
    """
    df = pd.read_excel(_wb_path(path), sheet_name="Data", header=1, usecols="B:I")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=["Ticker"])
    sym = (df["Ticker"].astype(str).str.split().str[0]
           .str.replace("/", "-", regex=False).str.upper())
    out = pd.DataFrame({
        "symbol": sym,
        "name": df["Name"],
        "weight": pd.to_numeric(df["Weight"], errors="coerce"),
        "industry": df[[c for c in df.columns if c.startswith("GICS Ind")][0]],
        "sector": df[[c for c in df.columns if c.startswith("GICS Sector")][0]],
    })
    out = out.dropna(subset=["symbol", "weight"])
    out = out[out["weight"] > 0]
    out["weight"] = out["weight"] / out["weight"].sum()
    return out.drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def sector_map(path=None) -> dict[str, str]:
    """Symbol -> GICS sector for current holdings, from the Portfolio sheet."""
    h = read_holdings(path)
    m = {}
    for _, r in h.iterrows():
        s = r["sector"]
        if isinstance(s, str) and s.strip() and not s.startswith("#"):
            m[r["symbol"]] = s.strip()
    return m
