"""Print the validation report and the regression table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
pd.set_option("display.width", 220)
from smif import engine, config

END = pd.Timestamp("2026-11-04")
a = engine.load(end=END)
print("as of", a.asof.date(), "| trading days", len(a.book.days))
v = a.validation()

print("\n--- share counts ---")
bad = v["share_failures"]
print("failures:", len(bad))
if len(bad): print(bad.to_string(index=False))
sub = v["shares"][v["shares"]["sub_share"]]
print("sub-share leftovers:", len(sub))
if len(sub): print(sub[["symbol","rebuilt","workbook","diff"]].to_string(index=False))

print("\n--- sleeve values vs custodian ---")
print(v["sleeves"].to_string(index=False))
print("\ncash gap:", v["cash"])

print("\n--- fill prices ---")
fs = v["fill_summary"]
print({k: fs[k] for k in ("checked","within_tolerance","share","no_price","skipped_interpolated")})
print(fs["worst"][["symbol","date","fill","close","ratio"]].to_string(index=False))

print("\n--- regression ---")
print(engine.regression_table(a, end=END).round(2).to_string(index=False))

fi = a.tracks["fixed_income"]
print("\nfixed income base", fi.base.date(), "cum %", round((fi.index.iloc[-1]-1)*100,3),
      "AGG %", round((fi.benchmark.iloc[-1]-1)*100,3))
print("\nprice notes:")
for k, val in sorted(a.book.price_notes.items()):
    print(f"  {k}: {val}")
