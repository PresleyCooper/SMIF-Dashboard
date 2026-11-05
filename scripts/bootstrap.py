"""One-off: copy the source files in, seed the ledger, fetch prices.

Run with:  python scripts/bootstrap.py "<transactions.xlsx>" "<SMIF Portfolio.xlsx>"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smif import config, engine, store


def main(tx_path, wb_path):
    config.ensure_dirs()
    print("installing seed files...")
    print(engine.install_seed(tx_path, wb_path))
    print("seeding ledger...")
    print(store.seed_from_export())
    tx = store.read_transactions()
    print(f"ledger rows: {len(tx)}  {tx['Trade Date'].min().date()} -> {tx['Trade Date'].max().date()}")
    print("fetching prices (this takes a few minutes the first time)...")

    def progress(i, n, t):
        print(f"  {i}/{n} ... {t}", flush=True)

    print(engine.refresh_prices(tx, progress=progress))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
