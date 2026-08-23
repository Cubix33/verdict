"""End-to-end evaluation of the VERDICT pipeline on a labelled corpus.

Reports the metrics from PLAN.md section 13, with one addition the project
cares about more than accuracy:

  false_accusation_rate  share of authentic documents called tampered

That is the number the reported bug was about. A detector that flags
everything scores well on recall and is worthless.

Split discipline
----------------
Calibration documents and evaluation documents come from disjoint *source*
images. A forgery derived from a source used for calibration is excluded too -
the calibrator saw that page's texture, its scanner and its fonts, so scoring
against its forged twin would measure memorisation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verdict import calibrate as cal_mod  # noqa: E402
from verdict.ledger import Ledger  # noqa: E402
from verdict.orchestrator import Orchestrator  # noqa: E402

CORPUS = ROOT / "testdata" / "corpus"
TAMPER_VERDICTS = {"tamper_detected"}


def score(results: list[dict]) -> dict:
    n_auth = sum(1 for r in results if r["truth"] == "authentic")
    n_forged = sum(1 for r in results if r["truth"] == "forged")

    tp = sum(1 for r in results if r["pred"] in TAMPER_VERDICTS and r["truth"] == "forged")
    fp = sum(1 for r in results if r["pred"] in TAMPER_VERDICTS and r["truth"] == "authentic")
    fn = sum(1 for r in results if r["pred"] not in TAMPER_VERDICTS and r["truth"] == "forged")

    # A detection only counts as bonded if at least one claim survived the
    # Verifier. This is the project's headline metric: it refuses to credit a
    # correct answer that could not prove itself.
    btp = sum(1 for r in results if r["pred"] in TAMPER_VERDICTS
              and r["truth"] == "forged" and r["bonds_held"] > 0)

    def f1(a, b, c):
        return round(2 * a / max(2 * a + b + c, 1), 3)

    bonds_total = sum(r["bonds_total"] for r in results)
    bonds_broken = sum(r["bonds_broken"] for r in results)

    return {
        "n_authentic": n_auth, "n_forged": n_forged,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "recall": round(tp / max(n_forged, 1), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "answer_f1": f1(tp, fp, fn),
        "bonded_f1": f1(btp, fp, fn),
        "false_accusation_rate": round(fp / max(n_auth, 1), 3),
        "bond_break_rate": round(bonds_broken / max(bonds_total, 1), 3),
        "shuffle_survivals": sum(r.get("shuffle_survivors", 0) for r in results),
        "refusal_rate": round(sum(1 for r in results
                                  if r["pred"] == "insufficient_evidence")
                              / max(len(results), 1), 3),
        "mean_latency_s": round(sum(r["latency_s"] for r in results)
                                / max(len(results), 1), 3),
    }


def by_mode(results: list[dict]) -> dict:
    modes: dict[str, dict] = {}
    for r in results:
        if r["truth"] != "forged":
            continue
        m = r.get("op") or "unknown"
        d = modes.setdefault(m, {"n": 0, "detected": 0, "bonded": 0})
        d["n"] += 1
        if r["pred"] in TAMPER_VERDICTS:
            d["detected"] += 1
            if r["bonds_held"] > 0:
                d["bonded"] += 1
    for d in modes.values():
        d["recall"] = round(d["detected"] / max(d["n"], 1), 3)
    return modes


def run(customer: str, limit: int | None, exclude_sources: set[str],
        verbose: bool) -> tuple[list[dict], dict]:
    labels = json.loads((CORPUS / "labels.json").read_text())
    cal = cal_mod.load(customer)

    ledger = Ledger(db=str(ROOT / "data" / "eval.db"))
    ledger.reset()
    orch = Orchestrator(customer_id=customer, ledger=ledger, calibration=cal)

    items = [(rel, meta) for rel, meta in sorted(labels.items())
             if meta.get("source") not in exclude_sources]
    if limit:
        auth = [i for i in items if i[1]["label"] == "authentic"][:limit]
        forged = [i for i in items if i[1]["label"] == "forged"][:limit]
        items = auth + forged

    results = []
    for i, (rel, meta) in enumerate(items, 1):
        path = CORPUS / rel
        t0 = time.time()
        try:
            v = orch.run(str(path), claim_id=f"EVAL-{i:04d}")
        except Exception as e:
            print(f"\n  ERROR on {rel}: {type(e).__name__}: {e}")
            continue
        claims = v["claims"]
        results.append({
            "file": rel, "truth": meta["label"], "op": meta.get("op"),
            "pred": v["verdict"], "confidence": v["confidence"],
            "bonds_total": len(claims),
            "bonds_held": sum(1 for c in claims if c["bond_status"] == "holds"),
            "bonds_broken": sum(1 for c in claims if c["bond_status"] == "broken"),
            "shuffle_survivors": v.get("controls", {}).get("shuffle_survivors", 0),
            "latency_s": round(time.time() - t0, 3),
        })
        if verbose:
            r = results[-1]
            mark = "ok " if ((r["pred"] in TAMPER_VERDICTS) ==
                             (r["truth"] == "forged")) else "MISS"
            print(f"  [{mark}] {rel:52} {r['pred']:22} "
                  f"bonds {r['bonds_held']}/{r['bonds_total']}")
        else:
            print("." if (results[-1]["pred"] in TAMPER_VERDICTS) ==
                  (results[-1]["truth"] == "forged") else "x", end="", flush=True)
    print()
    return results, score(results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", default="sroie")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--exclude-sources", default="",
                    help="comma-separated source filenames used for calibration")
    ap.add_argument("--out", default=str(ROOT / "testdata" / "eval_results.json"))
    args = ap.parse_args()

    exclude = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}
    results, s = run(args.customer, args.limit, exclude, args.verbose)

    print(f"\n{'metric':26}value")
    print("-" * 40)
    for k, v in s.items():
        print(f"{k:26}{v}")

    print(f"\n{'forgery mode':22}{'n':>5}{'detected':>10}{'recall':>9}{'bonded':>9}")
    print("-" * 55)
    for m, d in sorted(by_mode(results).items()):
        print(f"{m:22}{d['n']:5}{d['detected']:10}{d['recall']:9}{d['bonded']:9}")

    pathlib.Path(args.out).write_text(json.dumps(
        {"summary": s, "by_mode": by_mode(results), "results": results}, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
