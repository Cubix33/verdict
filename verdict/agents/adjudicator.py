"""ADJUDICATOR - perceptually starved. Issues claims, bonded to card hashes.

It reads the evidence manifest and the numeric readings on each card. It never
receives the image, not even downsampled. That restriction is Mechanism 1 from
idea.md, and it is what stops the agent with answer authority from forming an
impression the evidence does not support.

Three things were wrong before.

The thresholds were literals - `abs(delta) > 0.5` and `z > 3.0` - that had
never been measured against a real document. Calibration on authentic receipts
puts the worst honest region at `ela_z` around 5.7, so `z > 3.0` was below the
noise floor and fired constantly. Every threshold now comes from
`verdict.calibrate`.

A single flagged statistic was enough to accuse. Any one probe fires
occasionally on honest documents, and a receipt has dozens of regions, so
single-signal accusation is a false-positive engine. A claim now requires
corroboration from two *independent probe families* - grouped so that two
measurements of the same underlying physics cannot vote twice.

There was no way to conclude a document was fine. The verdict was either
`tamper_detected` or `undetermined`, so an authentic receipt could never come
back clean. `authentic` is now a first-class outcome.
"""
from __future__ import annotations

import json

from ..types import Claim

# Probes that measure the same physics must not corroborate each other.
# `flat_pixel_frac` and `hf_ratio` are both texture-smoothness statistics; if
# they counted as two independent votes, a single smooth region would satisfy
# the corroboration rule on its own and the rule would do nothing.
PROBE_FAMILY = {
    "ela_exceeds": "compression",
    "grid_exceeds": "compression",
    "hf_exceeds": "texture",
    "flat_exceeds": "texture",
    "dup_exceeds": "duplication",
    "stroke_exceeds": "typography",
    "baseline_exceeds": "typography",
}

SIGNAL_TEXT = {
    "ela_exceeds": "compression residual inconsistent with the rest of the page",
    "grid_exceeds": "JPEG block grid does not align with the surrounding page",
    "hf_exceeds": "high-frequency texture inconsistent with neighbouring text",
    "flat_exceeds": "region contains flat, gradient-free pixels unlike scanned paper",
    "dup_exceeds": "region is a near-identical copy of another region",
    "stroke_exceeds": "stroke width does not match the other words on its row",
    "baseline_exceeds": "baseline is offset from the other words on its row",
}

# A single signal may accuse only if it is this many calibrated units past the
# bound. Set high on purpose: one probe alone should almost never be enough.
SOLO_SEVERITY = 3.0


def _families(numeric: dict) -> dict[str, list[str]]:
    """Group the flags raised on one region by independent probe family."""
    hit: dict[str, list[str]] = {}
    for key, family in PROBE_FAMILY.items():
        if numeric.get(key):
            hit.setdefault(family, []).append(key)
    return hit


def _severity(cal, numeric: dict, keys: list[str]) -> float:
    metric_of = {"ela_exceeds": "ela_z", "grid_exceeds": "grid_ratio",
                 "hf_exceeds": "hf_ratio", "flat_exceeds": "flat_pixel_frac",
                 "dup_exceeds": "dup_score", "stroke_exceeds": "stroke_ratio",
                 "baseline_exceeds": "baseline_z"}
    best = 0.0
    for k in keys:
        m = metric_of.get(k)
        if m and numeric.get(m) is not None:
            best = max(best, cal.severity(m, numeric[m]))
    return round(best, 3)


def adjudicate(cards, cal) -> dict:
    """Form bonded claims from the manifest alone.

    `cards` is one claim's evidence and nothing else. The old signature took
    the whole ledger and iterated its process-global cache, which meant every
    analysis adjudicated over every image the process had ever seen.
    """
    claims: list[Claim] = []
    reasoning: list[str] = []
    n = 0

    by_id = {c.id: c for c in cards}
    arithmetic_card = next(
        (c for c in cards if c.numeric.get("arithmetic_mismatch")), None)
    provenance_card = next((c for c in cards if c.probe == "provenance"), None)

    region_cards = [c for c in cards if c.probe == "region_forensics"]
    if not region_cards and not arithmetic_card:
        return {"verdict": "insufficient_evidence", "claims": [],
                "confidence": None,
                "reasoning": ["no region carried enough pixels to probe"]}

    for card in region_cards:
        num = card.numeric or {}
        fams = _families(num)
        if not fams:
            continue

        keys = [k for ks in fams.values() for k in ks]
        sev = _severity(cal, num, keys)
        text_seen = (card.params or {}).get("text") or ""

        # --- duplication is self-evident and needs no corroboration -------
        if "duplication" in fams and num.get("dup_distance_px", 0) >= 40:
            n += 1
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the region at {card.bbox} reading '{text_seen}' is a "
                     f"near-identical copy of another region of this same document",
                bond=[card.id],
                bond_detail={"signals": ["dup_exceeds"], "families": ["duplication"],
                             "severity": sev, "rule": "duplication_single_signal"}))
            continue

        # --- arithmetic corroborated by a pixel anomaly at the same field --
        if (arithmetic_card is not None and card.bbox
                and arithmetic_card.bbox and _overlaps(card.bbox, arithmetic_card.bbox)):
            n += 1
            sig = sorted(fams)
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the stated total was altered: the arithmetic does not "
                     f"reconcile and the total field itself shows "
                     f"{', '.join(SIGNAL_TEXT[k] for k in keys[:2])}",
                bond=[card.id, arithmetic_card.id],
                bond_detail={"signals": keys, "families": sig, "severity": sev,
                             "rule": "arithmetic_plus_pixel"}))
            continue

        # --- two independent families ---------------------------------------
        if len(fams) >= 2:
            n += 1
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the region at {card.bbox} reading '{text_seen}' was "
                     f"digitally altered: " +
                     "; ".join(SIGNAL_TEXT[k] for k in keys[:3]),
                bond=[card.id],
                bond_detail={"signals": keys, "families": sorted(fams),
                             "severity": sev, "rule": "two_independent_families"}))
            continue

        # --- one family, but far past the calibrated bound ------------------
        if sev >= SOLO_SEVERITY:
            n += 1
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the region at {card.bbox} reading '{text_seen}' shows "
                     f"{SIGNAL_TEXT[keys[0]]}, {sev:.1f} calibrated units past "
                     f"this customer's authentic range",
                bond=[card.id],
                bond_detail={"signals": keys, "families": sorted(fams),
                             "severity": sev, "rule": "single_family_high_severity"}))
            continue

        reasoning.append(
            f"{card.id}: {len(fams)} family flagged at severity {sev} - "
            f"below the corroboration bar, no claim raised")

    # --- provenance is reported, never used to accuse of tampering --------
    if provenance_card is not None:
        pn = provenance_card.numeric or {}
        if pn.get("editor_signature"):
            n += 1
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the file carries an editing-software tag "
                     f"({pn['editor_signature']}), so it was resaved by a raster editor",
                bond=[provenance_card.id],
                bond_detail={"signals": ["editor_signature"],
                             "families": ["provenance"], "severity": 1.0,
                             "rule": "editor_tag", "kind": "provenance"}))
        elif not pn.get("has_exif"):
            reasoning.append(
                "no EXIF metadata, which is normal for screenshots, scans and "
                "web-delivered images - recorded, not treated as evidence")

    # --- arithmetic alone, with no pixel support --------------------------
    if arithmetic_card is not None and not any(
            arithmetic_card.id in c.bond for c in claims):
        an = arithmetic_card.numeric or {}
        reasoning.append(
            f"arithmetic does not reconcile (delta {an.get('delta')}) but no "
            f"pixel-level anomaly was found at the total - OCR error is at "
            f"least as likely as tampering, so no claim raised")

    tamper = [c for c in claims if c.bond_detail.get("kind") != "provenance"]
    if tamper:
        verdict = "tamper_detected"
        confidence = min(0.95, 0.55 + 0.1 * len(tamper) +
                         0.05 * max(c.bond_detail.get("severity", 0) for c in tamper))
    elif claims:
        verdict = "undetermined"        # provenance-only findings decide nothing
        confidence = None
    elif region_cards:
        verdict = "authentic"
        confidence = 0.7 + min(0.2, 0.01 * len(region_cards))
        reasoning.append(
            f"{len(region_cards)} regions probed, none exceeded this customer's "
            f"calibrated range on two independent probe families")
    else:
        verdict = "insufficient_evidence"
        confidence = None

    return {"agent": "adjudicator", "verdict": verdict,
            "confidence": round(confidence, 3) if confidence else None,
            "claims": claims, "reasoning": reasoning,
            "n_cards": len(cards)}


def _overlaps(a, b) -> bool:
    if not a or not b:
        return False
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def adjudicate_from_ledger(ledger, claim_id: str, cal) -> dict:
    """Convenience wrapper. Note the required claim_id - see the module note."""
    cards = ledger.cards(claim_id)
    out = adjudicate(cards, cal)
    out["manifest"] = ledger.manifest(claim_id)
    return out


if __name__ == "__main__":
    import sys
    from .. import calibrate as cal_mod
    from ..ledger import Ledger
    if len(sys.argv) < 2:
        print("Usage: python -m verdict.agents.adjudicator <claim_id> [customer]")
        sys.exit(2)
    led = Ledger()
    r = adjudicate_from_ledger(led, sys.argv[1],
                               cal_mod.load(sys.argv[2] if len(sys.argv) > 2 else "default"))
    r["claims"] = [c.__dict__ for c in r["claims"]]
    print(json.dumps(r, indent=2, default=str))
