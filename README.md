# SMIF Portfolio Analytics

A Streamlit app that rebuilds the full daily history of the **KSU Student Managed
Investment Fund** from its Schwab transaction export, and reports what the fund
actually earned, what it is correlated with, and what it is exposed to.

Seventeen years and ~3,080 transactions go in. Out come time-weighted returns
for each sleeve, correlation matrices, a point-in-time factor exposure against
the S&P 500, and a place to keep the ledger current.

> **Status.** The engine is complete, reconciled to the custodian and covered by
> 286 tests. `streamlit run app.py` works; **Overview**, **Returns**,
> **Correlations**, **Factor exposure** and **Transactions** are live. Only
> Methodology is still a placeholder. `python scripts/check.py` prints the full validation and
> return report in the meantime.

## What it answers

| Page | Question |
|---|---|
| **Overview** | What has the fund returned, and how much of it is sitting in cash? |
| **Returns** | Over every standard period, how did each sleeve do against its benchmark? |
| **Correlations** | What is actually diversifying what? |
| **Factor exposure** | What kind of book is this — large, cheap, high quality, fast growing? |
| **Transactions** | Add this month's trades and recompute. |
| **Methodology** | Which rules produced these numbers, and where do they not reconcile? |

## How it is put together

```
smif/            the engine — no Streamlit import anywhere in here
  config.py        every rule that is a choice: sleeves, ticker history, no-data names
  store.py         the canonical SQLite ledger, de-duplication and audit trail
  ledger.py        transactions in, share/cash/income/flow events out
  prices.py        yfinance with a parquet cache, the split un-adjust, fallbacks
  nav.py           the daily book: shares, prices, sleeve values, cash, flows
  returns.py       time-weighted returns, periods, drawdown, calendar years
  correlations.py  return series, correlation, clustering, beta
  entry.py         validating a transaction before it reaches the ledger
  factors.py       cross-sectional z-scores against the S&P 500
  workbook.py      holdings, the custodian snapshot and the S&P 500 constituents
  validate.py      the four reconciliation checks
  engine.py        the one call that runs the pipeline in order
ui/              everything Streamlit, styled after the Confluence design system
tests/           286 tests: the rules on synthetic ledgers, then the real regression
scripts/         bootstrap, the factor crawl, and a command-line check
```

The engine holds no Streamlit import, so it can be tested on its own and a test
can hand `build_book` a three-row ledger and a fake price store without touching
the database or the network.

## Getting started

```bash
pip install -r requirements.txt
```

```bash
python scripts/bootstrap.py "path/to/Portfolio Transaction History.xlsx" "path/to/SMIF Portfolio.xlsx"
```

That copies both files into `data/seed/` (the originals are never modified),
loads the export into `data/ledger.sqlite`, and downloads ~17 years of daily
prices for every ticker the fund has ever held into `data/cache/prices.parquet`.
It takes a few minutes once; after that a refresh only asks for the new days.

```bash
streamlit run app.py
```

The factor page needs one more crawl, which takes about seven minutes and is
then cached for a month:

```bash
python scripts/build_factors.py
```

```bash
python -m pytest -q          # the full suite
python scripts/check.py      # the reconciliation and regression report
```

## The method, in short

**Time-weighted returns.** Flows the manager did not choose must not count, and
what qualifies as a flow depends on whose return you are measuring.

- Total portfolio — only deposits and withdrawals are flows. A dividend is not:
  the fund earned it, and it moved from the equity pocket to the cash pocket
  without leaving the fund.

  `r = (V - external flows) / V_prev - 1`

- A sleeve — the sleeve cannot see the cash pocket, so a dividend leaves it and
  has to be added back, and a purchase arrives from outside it and has to be
  taken out.

  `r = (V + income - net purchases) / V_prev - 1`

Each sleeve's growth index is based on the close **before** its first purchase,
which is what gives day one's fill-to-close move somewhere to live.

**Real prices, not adjusted ones.** The ledger holds the share counts the fund
actually owned, so the prices have to be the ones a share actually traded at.
Yahoo's close is split-adjusted, so the real close is recovered as
`close x (product of every split factor dated strictly after that date)`. Doing
it generally rather than from a list of known splits is what makes Yahoo's
fractional spin-off "splits" work too — the 1.589 it records for RTX on
2020-04-03 turns today's RTX close back into the UTX price the fund held.

**Fourteen names Yahoo cannot price** (acquired, wound up, or the ticker has
since been reissued to a different company) are interpolated log-linearly
between the fund's own fills, and every position valued that way is flagged.

## Reconciliation

Checked against the custodian snapshot of 2026-09-22 and the current holdings
sheet:

| Check | Result |
|---|---|
| Final share counts vs the Portfolio sheet | every holding matches; 12 sub-share remnants on long-exited names |
| Fixed income sleeve | $34,054.67 — the custodian's Fixed subtotal, to the cent |
| Equity sleeve | $528,798.53 vs $528,787.43 (Equity + Real Estate) — **+$11.10**, 0.002% |
| Cash | $23,764.42 vs $21,722.61 — **+$2,041.81**, a known gap |
| Buy/Sell fills within 5% of the rebuilt close | 346 / 346 |

The cash gap is reported, not corrected. The rebuild holds *more* cash than the
custodian, which means the export is missing an outflow — most likely a single
fee or transfer that was never exported. Plugging it with a balancing entry
would make the reconciliation green and the history wrong, and would hide the
discrepancy from whoever eventually finds the missing row. It is 0.35% of the
fund and it is pinned by a test, so if it ever changes size that is a new fact.

## Returns through the 2026-11-04 close

| Period | Equity | SPY | Total portfolio | SPY (from inception) |
|---|---|---|---|---|
| All time | +620.21% | +804.41% | +501.15% | +863.68% |
| 10Y | +319.21% | +330.55% | +267.14% | +330.55% |
| 5Y | +59.51% | +75.59% | +55.55% | +75.59% |
| 3Y | +69.90% | +82.92% | +63.02% | +82.92% |
| 1Y | +21.95% | +14.79% | +19.08% | +14.79% |
| YTD | +18.68% | +13.33% | +16.11% | +13.33% |

Equity is measured from 2010-01-20, the close before the fund's first stock
purchase; the total portfolio from 2009-10-13, the first deposit. Fixed income,
bought on 2026-09-22, is +0.54% against AGG's +0.18%.

Every cell is asserted by `tests/test_returns.py` against the pinned
2026-11-04 close, to a tolerance of 0.15 percentage points — wide enough to
absorb Yahoo restating a dividend somewhere in seventeen years, far too tight to
absorb a transaction rule being wrong.

## Not built

Performance attribution (allocation vs selection) is deliberately out of scope.
The structure leaves room for it as its own page.
