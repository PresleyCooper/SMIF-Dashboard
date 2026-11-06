"""
smif/config.py — every rule that is a *choice* rather than arithmetic.

Anything a future student might need to change lives here: sleeve membership,
ticker history, the names Yahoo cannot price, and where files are kept. The
engine modules import from this and hold no literals of their own, so adding a
bond fund or a renamed ticker is a one-line edit in one file.

Dates are naive Timestamps throughout the engine — the custodian export has no
timezone and neither does a daily close, so carrying one would only invite
off-by-one errors at the day boundary.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
# SMIF_DATA_DIR lets a deployment point at a writable volume without touching
# code; everything else hangs off it so there is exactly one root to redirect.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SMIF_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
LEDGER_DB = DATA_DIR / "ledger.sqlite"
PRICE_CACHE = CACHE_DIR / "prices.parquet"
SPLIT_CACHE = CACHE_DIR / "splits.parquet"
FUNDAMENTALS_CACHE = CACHE_DIR / "fundamentals.json"
UNIVERSE_CACHE = CACHE_DIR / "universe.json"
RESULTS_CACHE = CACHE_DIR / "results.pkl"

# Seed inputs. These are read once (to populate the ledger / the constituent
# list) and never written to — the canonical ledger is LEDGER_DB.
SEED_DIR = DATA_DIR / "seed"
SEED_TRANSACTIONS = SEED_DIR / "Portfolio Transaction History.xlsx"
SEED_WORKBOOK = SEED_DIR / "SMIF Portfolio.xlsx"


def ensure_dirs() -> None:
    for p in (DATA_DIR, CACHE_DIR, SEED_DIR):
        p.mkdir(parents=True, exist_ok=True)


# ── fund timeline ─────────────────────────────────────────────────────────────
INCEPTION = pd.Timestamp("2009-10-13")        # the first Deposit
PRICE_START = "2009-09-01"                    # a little history before inception

# The custodian snapshot embedded in the workbook's Backend sheet.
CUSTODIAN_AS_OF = pd.Timestamp("2026-09-22")
CUSTODIAN_SUBTOTALS = {
    "Cash": 21_722.61,          # sweep cash 1,328.61 + SNOXX 20,394.00
    "Fixed": 34_054.67,
    "Equity": 522_431.87,
    "Real Estate": 6_355.56,
    "Total": 584_564.71,
}
# Real Estate is a GICS sector, not a sleeve: EQIX sits in the equity sleeve.
CUSTODIAN_EQUITY = CUSTODIAN_SUBTOTALS["Equity"] + CUSTODIAN_SUBTOTALS["Real Estate"]

# ── sleeves ───────────────────────────────────────────────────────────────────
# Fixed income is an explicit list because there is no way to tell a bond ETF
# from an equity ETF by ticker. Add new bond funds here and nothing else has to
# change: the equity sleeve is defined as "everything priced that is not fixed
# income and not cash-like", so a name left out would silently land in equity.
FIXED_INCOME = {"IEF", "LQD", "VGIT", "VTIP"}
CASH_LIKE = {"CASH", "SNOXX"}                 # sweep cash + the money market
MONEY_MARKET = {"SNOXX"}                      # priced at a constant $1.00

SLEEVES = ("equity", "fixed_income", "cash")
SLEEVE_LABEL = {
    "equity": "Equity",
    "fixed_income": "Fixed income",
    "cash": "Cash",
    "total": "Total portfolio",
}

# ── benchmarks ────────────────────────────────────────────────────────────────
BENCHMARKS = {"equity": "SPY", "fixed_income": "AGG", "cash": None, "total": "SPY"}
BENCHMARK_FIELD = "Adj Close"                 # total return, so dividends count

# ── ticker history ────────────────────────────────────────────────────────────
# Ledger symbol -> the Yahoo ticker that carries its price history. A rename or
# merger means the surviving ticker's series IS the old one's history.
YAHOO_MAP = {
    "MHP": "SPGI",        # McGraw-Hill -> McGraw Hill Financial -> S&P Global
    "MHFI": "SPGI",
    "NU": "ES",           # Northeast Utilities -> Eversource
    "UTX": "RTX",         # United Technologies -> Raytheon Technologies
    "RDS.B": "SHEL",      # Royal Dutch Shell B -> Shell plc
    "GOLD": "B",          # the ledger's post-2019 GOLD is Barrick, now "B"
}

# Randgold traded as GOLD until 2019-01-02, when 43 Randgold shares became
# 263.5 Barrick shares. Yahoo's current GOLD is an unrelated company, so the old
# position is carried under its own synthetic symbol and priced off Barrick at
# the exchange ratio.
RANDGOLD_SYMBOL = "GOLD_RANDGOLD"
RANDGOLD_SWITCH = pd.Timestamp("2019-01-02")
RANDGOLD_PROXY = "B"
RANDGOLD_RATIO = 263.5 / 43.0                 # Barrick shares per Randgold share

# Names Yahoo has no usable history for (delisted, acquired, or the ticker has
# since been reissued to a different company). These are priced by log-linear
# interpolation between the fund's own fills, and the UI flags any position
# valued this way.
NO_YAHOO_DATA = {
    "AET",    # Aetna, acquired by CVS 2018
    "AFSI",   # AmTrust Financial, taken private 2018
    "CELG",   # Celgene, acquired by BMY 2019
    "CERN",   # Cerner, acquired by Oracle 2022
    "DO",     # Diamond Offshore
    "DOW",    # Dow Chemical 2014-2016 — today's DOW is the 2019 spin-off
    "ESI",    # ITT Educational Services, wound up 2016
    "FURX",   # Furiex Pharmaceuticals (the PPDI successor), acquired 2014
    "LCI",    # Lannett
    "MNK",    # Mallinckrodt
    "PPDI",   # Pharmaceutical Product Development
    "QCOR",   # Questcor, acquired by Mallinckrodt 2014
    "SNV",    # Synovus
    "XLNX",   # Xilinx, acquired by AMD 2022
}

# A fill is only a usable price anchor if it is a real trade, not a fractional
# DRIP top-up or a zero-proceeds write-off.
ANCHOR_MIN_QUANTITY = 0.9

# Below this a line is a remnant, not a position. A dozen long-exited names
# carry sub-share leftovers worth a dollar or two — 0.02 of a Diamond Offshore,
# -0.01 of a Synovus — which are real in the rebuild and swept by the
# custodian. They belong in the reconciliation, where they are evidence, and
# nowhere else: counted as holdings they turn 38 equity positions into 50, and
# correlated as holdings they add rows of noise to the matrix.
MATERIAL_POSITION_VALUE = 100.0

# Questcor's exit was cash plus Mallinckrodt stock, so its last "price" has to
# be constructed: 30 QCOR shares became $900 plus 26.91 MNK shares.
QCOR_EXIT = {
    "date": pd.Timestamp("2014-08-15"),
    "cash": 900.0,
    "mnk_shares": 26.91,
    "qcor_shares": 30.0,
}
# ITT Educational's final sale went through at $0 — a write-off, not a price.
DROP_ZERO_ANCHORS = {"ESI"}

# ── transaction grammar ───────────────────────────────────────────────────────
# The complete Activity vocabulary of the Schwab export. The ledger refuses a
# row whose Activity is not one of these rather than guessing at its direction.
ACTIVITIES = (
    "Deposit",
    "Withdrawal",
    "Buy",
    "Sell",
    "Income (Dividend)",
    "Income (Qualified Dividend)",
    "Income (Non-Qualified Dividend)",
    "Income (Reinvested Dividend)",
    "Income (Reinvested Qualified Dividend)",
    "Income (Reinvested Interest)",
    "Income (Short Term Gain)",
    "Income (Long Term Gain)",
    "Expense (Foreign Tax Paid)",
    "Expense (Other Expense)",
    "Return of Principal",
    "Split",
    "Credit Security",
    "Credit Security (Custodian Lot Adjustment)",
    "Debit Security",
    "Debit Security (Custodian Lot Adjustment)",
)
TX_COLUMNS = ["Trade Date", "Activity", "Description", "Symbol",
              "Quantity", "Amount", "Notes"]

# ── validation thresholds ─────────────────────────────────────────────────────
FILL_PRICE_TOLERANCE = 0.05       # a fill should land within 5% of the close
FILL_MATCH_TARGET = 0.98          # and at least this share of them should

# ── factor exposure ───────────────────────────────────────────────────────────
FACTOR_WINSOR_PCT = 1.0           # clip raw values at the 1st/99th percentile
FACTOR_Z_CLIP = 3.0
FACTOR_MIN_UNIVERSE = 30
FUNDAMENTALS_TTL_DAYS = 30
SEC_USER_AGENT = os.environ.get(
    "SMIF_SEC_USER_AGENT", "KSU SMIF Analytics (contact via repository issues)")


def today() -> pd.Timestamp:
    return pd.Timestamp(date.today())
