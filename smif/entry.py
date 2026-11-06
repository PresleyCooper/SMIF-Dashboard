"""
smif/entry.py — validating a transaction before it reaches the ledger.

Every rule here exists because the alternative is a number that looks fine. A
sell of more shares than the fund owns does not raise anything: it produces a
negative position, which gets priced, which quietly subtracts from the equity
sleeve for the rest of history. A buy with no amount produces free shares. A
misspelled ticker produces a new position with no price and a silent hole in
the NAV.

So validation is a gate, not a convenience. Errors block the save; warnings are
things a human can legitimately mean — a new ticker, a back-dated correction —
and are shown with a confirmation rather than refused.

One deliberate omission: this does NOT check that cash stays positive. The
rebuilt cash balance already disagrees with the custodian by about $2,000
because the export is missing an outflow, and a rule that refused any entry
taking cash negative would start rejecting legitimate trades on the strength
of a known gap in the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config, ledger, nav, store

# Activities that move shares and therefore need a symbol and a quantity.
SHARE_ACTIVITIES = {
    "Buy", "Sell", "Split",
    "Credit Security", "Credit Security (Custodian Lot Adjustment)",
    "Debit Security", "Debit Security (Custodian Lot Adjustment)",
    "Income (Reinvested Dividend)", "Income (Reinvested Qualified Dividend)",
}
# Activities that reduce a position, so the fund must hold enough.
REDUCING_ACTIVITIES = {"Sell", "Debit Security",
                       "Debit Security (Custodian Lot Adjustment)"}
# Activities that do not belong to a security at all.
CASH_ACTIVITIES = {"Deposit", "Withdrawal"}
# Amount may legitimately be negative: a fee reversal.
SIGNED_AMOUNT = {"Expense (Foreign Tax Paid)", "Expense (Other Expense)"}


@dataclass(frozen=True)
class Issue:
    level: str          # "error" blocks the save; "warning" asks first
    field: str
    message: str

    @property
    def blocking(self) -> bool:
        return self.level == "error"


def _num(v) -> float | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def validate_row(row: dict, *, holdings: pd.Series | None = None,
                 known_symbols: set[str] | None = None,
                 existing_keys=None,
                 new_symbol_confirmed: bool = False,
                 today: pd.Timestamp | None = None) -> list[Issue]:
    """
    Check one proposed transaction.

    `holdings` is {symbol: shares} as of now, used to stop a sell from
    exceeding the position. `known_symbols` is every symbol the ledger has ever
    seen; anything else is a warning until confirmed.
    """
    issues: list[Issue] = []
    today = today or config.today()
    known = {s.upper() for s in (known_symbols or set())}
    held = holdings if holdings is not None else pd.Series(dtype=float)

    activity = str(row.get("Activity") or "").strip()
    symbol = str(row.get("Symbol") or "").strip().upper()
    qty = _num(row.get("Quantity"))
    amt = _num(row.get("Amount"))

    # -- date ------------------------------------------------------------------
    raw_date = row.get("Trade Date")
    date = None
    if raw_date in (None, ""):
        issues.append(Issue("error", "Trade Date", "A trade date is required."))
    else:
        try:
            date = pd.Timestamp(raw_date)
        except (ValueError, TypeError):
            issues.append(Issue("error", "Trade Date", "Not a readable date."))
    if date is not None:
        if date > today:
            issues.append(Issue("error", "Trade Date",
                                f"{date:%d %b %Y} is in the future."))
        elif date < config.INCEPTION:
            issues.append(Issue("warning", "Trade Date",
                                f"Before the fund's first deposit "
                                f"({config.INCEPTION:%d %b %Y})."))

    # -- activity --------------------------------------------------------------
    if not activity:
        issues.append(Issue("error", "Activity", "Pick an activity."))
        return issues
    if activity not in config.ACTIVITIES:
        issues.append(Issue("error", "Activity",
                            f"{activity!r} is not one of the Schwab activity "
                            f"types."))
        return issues

    # -- symbol ----------------------------------------------------------------
    if activity in CASH_ACTIVITIES:
        if symbol and symbol != "CASH":
            issues.append(Issue("warning", "Symbol",
                                f"A {activity} normally carries CASH, not "
                                f"{symbol}."))
    elif not symbol:
        if activity.startswith("Expense"):
            issues.append(Issue("warning", "Symbol",
                                "No symbol: this expense will be attributed "
                                "through its Description, or left out of every "
                                "sleeve if that fails to match."))
        else:
            issues.append(Issue("error", "Symbol",
                                f"A {activity} needs a symbol."))
    elif symbol not in known and symbol not in config.CASH_LIKE:
        if new_symbol_confirmed:
            issues.append(Issue("warning", "Symbol",
                                f"{symbol} is new to the ledger and will need "
                                f"a sleeve."))
        else:
            issues.append(Issue("error", "Symbol",
                                f"{symbol} has never appeared in the ledger. "
                                f"Confirm it is a new position, or correct the "
                                f"spelling."))

    # -- quantity --------------------------------------------------------------
    if activity in SHARE_ACTIVITIES and symbol != "CASH":
        if qty is None or qty == 0:
            issues.append(Issue("error", "Quantity",
                                f"A {activity} needs a quantity."))
        elif qty < 0:
            issues.append(Issue("error", "Quantity",
                                "Quantity is always positive; the activity "
                                "sets the direction."))
    if activity in REDUCING_ACTIVITIES and symbol and qty:
        owned = float(held.get(symbol, 0.0))
        if qty > owned + 1e-6:
            issues.append(Issue(
                "error", "Quantity",
                f"The fund holds {owned:,.4f} {symbol}; this would take it to "
                f"{owned - qty:,.4f}."))

    # -- amount ----------------------------------------------------------------
    needs_amount = (activity in {"Buy", "Sell", "Deposit", "Withdrawal",
                                 "Return of Principal"}
                    or activity.startswith("Income")
                    or activity.startswith("Expense"))
    share_paying = activity in {"Income (Reinvested Qualified Dividend)"} or (
        activity == "Income (Reinvested Dividend)" and symbol != "CASH")
    if share_paying:
        needs_amount = False

    if needs_amount:
        if amt is None:
            issues.append(Issue("error", "Amount",
                                f"A {activity} needs an amount."))
        elif amt == 0:
            issues.append(Issue("warning", "Amount", "An amount of zero."))
        elif amt < 0 and activity not in SIGNED_AMOUNT:
            issues.append(Issue(
                "error", "Amount",
                "Amount is always positive; the activity sets the direction. "
                "Only an expense may be negative, as a fee reversal."))

    if activity in {"Split"} and amt:
        issues.append(Issue("warning", "Amount",
                            "A split carries no cash; this amount is ignored."))
    if activity.startswith(("Credit Security", "Debit Security")) and amt:
        issues.append(Issue(
            "warning", "Amount",
            "On a security transfer the Amount is cost basis, not market "
            "value. It is recorded but never used as a price."))

    # -- duplicate -------------------------------------------------------------
    if existing_keys is not None and date is not None:
        key = store.dedupe_key(date, activity, symbol or None, qty, amt)
        if existing_keys.get(key, 0) > 0:
            issues.append(Issue(
                "warning", "",
                "The ledger already holds a row with this date, activity, "
                "symbol, quantity and amount. Saving will add a second one."))

    return issues


def blocking(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.blocking]


def warnings(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if not i.blocking]


def current_holdings(book) -> pd.Series:
    """{symbol: shares} as of the last close — what a sell is checked against."""
    q = book.shares.iloc[-1]
    return q[q.abs() > 1e-9]


def needs_classification(symbol: str, overrides: dict[str, str] | None = None
                         ) -> bool:
    """
    Whether the app has to ask which sleeve a symbol belongs to.

    Only when the config does not already decide. Everything unknown would
    otherwise land in equity by default, which is the right fallback and the
    wrong thing to do silently to a bond fund.
    """
    s = str(symbol).strip().upper()
    if s in config.CASH_LIKE or s in config.FIXED_INCOME:
        return False
    return s not in (overrides or {})


def preview_effect(row: dict) -> dict:
    """
    What this row will do, in the engine's own terms.

    Shown beside the form so the rules are visible at the point of entry
    rather than only in the methodology — "this adds 10 shares and removes
    $500" is a claim the person typing can check.
    """
    frame = pd.DataFrame([{
        "Trade Date": pd.Timestamp(row.get("Trade Date")),
        "Activity": row.get("Activity"),
        "Description": row.get("Description") or "",
        "Symbol": row.get("Symbol") or None,
        "Quantity": _num(row.get("Quantity")),
        "Amount": _num(row.get("Amount")),
        "Notes": row.get("Notes") or "",
    }])
    try:
        ev = ledger.build_events(frame)
    except ValueError as exc:
        return {"error": str(exc)}
    if ev.empty:
        return {"error": "no effect"}
    e = ev.iloc[0]
    return {
        "symbol": e["sym"], "shares": float(e["dq"]), "cash": float(e["dcash"]),
        "external_flow": float(e["flow"]), "income": float(e["income"]),
        "net_purchase": float(e["netbuy"]),
        "sleeve": nav.sleeve_of(e["sym"]) if e["sym"] else None,
    }
