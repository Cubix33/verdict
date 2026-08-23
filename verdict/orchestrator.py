"""Wires the agents together and enforces the separation of powers.

  Triage      plans nothing, spends nothing, and may only raise risk
  Prober      plans and spends, but has no authority to decide
  Adjudicator decides, but never sees the image
  Verifier    validates, but shares no state with the Adjudicator

The final verdict is not the Adjudicator's. A verdict only stands on claims
whose bonds survived the Verifier, which is why a run can end in
`undetermined` even though the Adjudicator was confident - and why
`tamper_detected` here means something a reader can check.
"""
from __future__ import annotations

import pathlib
import time
import uuid

import cv2

from . import calibrate as cal_mod
from .agents import adjudicator, prober, triage
from .agents.verifier import verify_claims
from .ledger import Ledger
from .types import Verdict


class Orchestrator:
    def __init__(self, customer_id: str = "default", ledger: Ledger | None = None,
                 budget: float = 150.0, run_controls: bool = True,
                 use_probes: bool = True, calibration=None):
        self.customer_id = customer_id
        self.cal = calibration if calibration is not None else cal_mod.load(customer_id)
        self.ledger = ledger if ledger is not None else Ledger()
        self.budget = budget
        self.run_controls = run_controls
        self.use_probes = use_probes
        self.phash_index: dict[str, str] = {}

    def run(self, image_path: str, claim_id: str | None = None,
            claim_text: str | None = None) -> dict:
        t0 = time.time()
        claim_id = claim_id or f"CLM-{uuid.uuid4().hex[:8]}"
        path = str(pathlib.Path(image_path))
        trace: list[dict] = []

        img = cv2.imread(path)
        if img is None:
            return self._finish(claim_id, "insufficient_evidence", None, [], {},
                                t0, [], trace + [{"agent": "input",
                                                  "error": "imread_failed"}])

        # 1. TRIAGE -----------------------------------------------------
        tri = triage.process_claim(path, claim_text, self.phash_index, claim_id)
        if tri.get("phash"):
            self.phash_index[claim_id] = tri["phash"]
        trace.append({"agent": "triage", "decision": tri["decision"],
                      "risk": tri["risk"], "flags": tri["flags"],
                      "reason": tri["reason"]})

        # 2. PROBER -----------------------------------------------------
        # Probing runs even on low-risk documents. Triage cannot see pixels,
        # so a clean triage is not evidence of authenticity - and a verdict of
        # `authentic` has to be earned by probes that looked and found nothing.
        if not self.use_probes:
            crop = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
            self.ledger.add(claim_id, "EV-1", "downsample", {}, None, crop,
                            "whole document at 448x448", {}, 0.0)
            pr = {"created_cards": [{"card_id": "EV-1"}], "budget_spent": 0.0}
        else:
            pr = prober.process_probes(path, claim_id, self.ledger, self.cal,
                                       triage=tri, budget=self.budget)
        cards = self.ledger.cards(claim_id)
        trace.append({"agent": "prober", "n_cards": len(cards),
                      "budget_spent": pr.get("budget_spent", 0.0),
                      "regions_probed": len(pr.get("regions", []))})

        # 3. ADJUDICATOR (starved: it receives `cards`, never `img`) -----
        adj = adjudicator.adjudicate(cards, self.cal)
        trace.append({"agent": "adjudicator", "verdict": adj["verdict"],
                      "n_claims": len(adj["claims"]), "sees_image": False,
                      "reasoning": adj["reasoning"]})

        # 4. VERIFIER ----------------------------------------------------
        claims = adj["claims"]
        controls: dict = {}
        if claims and self.run_controls:
            ver = verify_claims(claims, cards, self.ledger, claim_id, self.cal, img)
            trace.append({"agent": "verifier", "holds": ver["holds"],
                          "broken": ver["broken"],
                          "broken_reasons": [c["detail"].get("reason")
                                             for c in ver["claims"]
                                             if c["overall"] == "broken"]})
            controls = {
                "blank": all(c["detail"]["checks"].get("blank_control", True)
                             for c in ver["claims"]),
                "shuffle": all(c["detail"]["checks"].get("shuffle_control", True)
                               for c in ver["claims"]),
                "provenance": all(c["detail"]["checks"].get("provenance", True)
                                  for c in ver["claims"]),
                "shuffle_survivors": sum(c["detail"].get("shuffle_survivors", 0)
                                         for c in ver["claims"]),
            }
        elif claims:
            for c in claims:
                c.bond_status = "untested"

        # 5. VERDICT -----------------------------------------------------
        # Only bonded claims count. An unbonded accusation is struck from the
        # record even if it happens to be true.
        held = [c for c in claims if c.bond_status == "holds"]
        tamper_held = [c for c in held if c.bond_detail.get("kind") != "provenance"]

        if tamper_held:
            verdict = "tamper_detected"
            confidence = adj["confidence"]
        elif adj["verdict"] == "authentic":
            verdict, confidence = "authentic", adj["confidence"]
        elif claims and not held:
            # The Adjudicator accused; nothing survived the bond test.
            verdict, confidence = "undetermined", None
        elif adj["verdict"] == "insufficient_evidence":
            verdict, confidence = "insufficient_evidence", None
        else:
            verdict, confidence = "undetermined", None

        self.ledger.record_claim(claim_id, tri.get("phash"), verdict, path)
        return self._finish(claim_id, verdict, confidence, claims, controls,
                            t0, [c.id for c in cards], trace,
                            triage=tri, adjudication=adj)

    def _finish(self, claim_id, verdict, confidence, claims, controls, t0,
                chain, trace, triage=None, adjudication=None) -> dict:
        cost = {"vlm_calls": 0, "probe_units": sum(
            c.cost_units for c in self.ledger.cards(claim_id))}
        v = Verdict(claim_id=claim_id, verdict=verdict, confidence=confidence,
                    claims=claims, controls=controls, cost=cost,
                    latency_s=round(time.time() - t0, 3), evidence_chain=chain)
        d = v.to_dict()
        d["trace"] = trace
        d["customer_id"] = self.customer_id
        if triage is not None:
            d["triage"] = triage
        if adjudication is not None:
            d["adjudicator_reasoning"] = adjudication.get("reasoning", [])
            d["adjudicator_verdict"] = adjudication.get("verdict")
        return d


def analyze(image_path: str, customer_id: str = "default", **kw) -> dict:
    """One-shot convenience entry point."""
    return Orchestrator(customer_id=customer_id, **kw).run(image_path)


def _main() -> int:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Run the VERDICT pipeline on one image.")
    ap.add_argument("image")
    ap.add_argument("--customer", default="default")
    ap.add_argument("--budget", type=float, default=90.0)
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()

    try:
        result = analyze(args.image, customer_id=args.customer,
                         budget=args.budget, run_controls=not args.no_controls)
    except FileNotFoundError as e:
        print(e)
        return 2

    print(_json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
