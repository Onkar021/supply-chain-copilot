"""CLI:  python -m src.cli "Heavy rains are delaying all freight leaving Sao Paulo for the next 5 days" [--lang pt] [--emails 2]
        python -m src.cli --samples      (regenerates the two sample reports in samples/)"""
import argparse, re
from . import config as C
from .orchestrator import Pipeline

SCENARIOS = ["Heavy rains are delaying all freight leaving Sao Paulo for the next 5 days",
             "A truckers' strike is blocking bulky furniture and home appliance shipments out of Minas Gerais and Parana for the next 10 days"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?"); ap.add_argument("--lang", default="en", choices=["en", "pt"])
    ap.add_argument("--emails", type=int, default=2); ap.add_argument("--samples", action="store_true")
    a = ap.parse_args()
    p = Pipeline()
    if a.samples:
        C.OUT_DIR.parent.joinpath("samples").mkdir(exist_ok=True)
        for i, q in enumerate(SCENARIOS, 1):
            r = p.run(q, a.lang, a.emails)
            C.ROOT.joinpath("samples", f"scenario_{i}.md").write_text(r.to_markdown(), encoding="utf-8")
            print(f"wrote samples/scenario_{i}.md")
        return
    if not a.query: ap.error("provide a disruption query or --samples")
    r = p.run(a.query, a.lang, a.emails); md = r.to_markdown(); print(md)
    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / (re.sub(r"\W+", "_", a.query)[:40] + ".md")).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
