"""Build the factor snapshot: universe prices, then fundamentals.

  python scripts/build_factors.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smif import config, engine, factors, workbook


def main():
    config.ensure_dirs()
    sp = workbook.read_sp500()
    a = engine.load()
    held = [s for s in a.book.holdings_on(
        min_value=config.MATERIAL_POSITION_VALUE).index
        if s not in config.CASH_LIKE]
    tickers = sorted(set(sp["symbol"]) | set(held))
    print(f"universe {len(sp)} + holdings -> {len(tickers)} tickers", flush=True)

    t0 = time.time()
    print("1/2 price panel...", flush=True)
    panel = factors.fetch_universe_prices(
        tickers, end=a.asof,
        progress=lambda i, n, t: print(f"   {i}/{n} {t}", flush=True))
    print(f"   panel {panel.shape} in {time.time()-t0:.0f}s", flush=True)

    print("2/2 fundamentals...", flush=True)
    out = factors.build_snapshot(
        tickers, price_panel=panel,
        progress=lambda i, n, t: (
            print(f"   {i}/{n} {t}", flush=True) if i % 25 == 0 else None))
    print(f"done: n={out['n']} fetched={out['fetched']} failed={out['failed']} "
          f"in {time.time()-t0:.0f}s", flush=True)
    for k, v in sorted(out["coverage"].items()):
        print(f"   {k:14s} {v}/{out['n']}", flush=True)


if __name__ == "__main__":
    main()
