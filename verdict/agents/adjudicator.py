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

from ..probes.ocr import parse_amount
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

METRIC_OF = {"ela_exceeds": "ela_z", "grid_exceeds": "grid_ratio",
             "hf_exceeds": "hf_ratio", "flat_exceeds": "flat_pixel_frac",
             "dup_exceeds": "dup_score", "stroke_exceeds": "stroke_ratio",
             "baseline_exceeds": "baseline_z"}


def _evaluate_region(cal, numeric: dict):
    """Apply both calibration tiers to a region's raw statistics.

    The Adjudicator re-derives exceedance from the raw values on the card
    rather than trusting the Prober's `*_exceeds` flags - the flag is a
    display convenience, and an agent with answer authority should not
    inherit another agent's thresholding.

    Returns (families_screen, families_strong, tiers) where the families
    dicts map family -> [signal keys] and tiers maps metric -> the highest
    tier it crossed ("strong" beats "screen").
    """
    fams_screen: dict[str, list[str]] = {}
    fams_strong: dict[str, list[str]] = {}
    tiers: dict[str, str] = {}
    for key, family in PROBE_FAMILY.items():
        metric = METRIC_OF[key]
        value = numeric.get(metric)
        if value is None:
            continue
        if cal.exceeds(metric, value, tier="strong"):
            fams_strong.setdefault(family, []).append(key)
            fams_screen.setdefault(family, []).append(key)
            tiers[metric] = "strong"
        elif cal.exceeds(metric, value, tier="screen"):
            fams_screen.setdefault(family, []).append(key)
            tiers[metric] = "screen"
    return fams_screen, fams_strong, tiers


def _severity(cal, numeric: dict, keys: list[str], tier: str = "screen") -> float:
    best = 0.0
    for k in keys:
        m = METRIC_OF.get(k)
        if m and numeric.get(m) is not None:
            best = max(best, cal.severity(m, numeric[m], tier=tier))
    return round(best, 3)


def _is_motivated(card) -> bool:
    """Would a forger have had a reason to touch this region?

    Forgers edit amounts; they do not redraw the shop's name. The screen-tier
    corroboration rule therefore only applies to regions where a motive
    exists: text that parses as a money amount, the field the arithmetic
    check implicated, or a region the pixel scan nominated that carries no
    readable text at all (an erased field reads as nothing).

    Header and prose regions can still be accused - but only past the strong
    tier. Without this distinction the corroboration rule accuses logos and
    bold headlines, which are the most *legitimately* unusual regions on any
    receipt: measured on the full corpus, every surviving false positive was
    a screen-tier two-family claim on decorative text.
    """
    params = card.params or {}
    why = params.get("why") or ""
    text = (params.get("text") or "").strip()
    if not text:
        # Unreadable regions are only motivated when the dead-grain scan
        # nominated them (an erased field reads as nothing AND has no paper
        # grain). Any other no-text region is usually a logo or a stamp -
        # decorative content that is legitimately unusual on every axis.
        return why.startswith("dead_grain")
    if parse_amount(text) is not None:
        return True                      # money amount: the forger's target
    # A tampered amount often no longer OCRs cleanly ('46881' with the
    # decimals lost, a retyped total reading as one long digit run) -
    # requiring a perfect money parse here would exempt precisely the fields
    # whose glyphs the forger just replaced. Digit-dominated text keeps the
    # gate open for mangled numbers while still excluding headers and prose,
    # which read as letters. Long digit runs (phone numbers, till ids) stay
    # in on purpose: a retyped amount is indistinguishable from them once its
    # decimal point is lost, and measured on the corpus the length cap that
    # tried to separate the two blocked a true forgery while letting the one
    # digit-run false positive through anyway.
    digits = sum(ch.isdigit() for ch in text)
    if digits >= 3 and digits >= 0.5 * len(text):
        return True
    return why == "arithmetic_target"


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
        fams, fams_strong, tiers = _evaluate_region(cal, num)
        if not fams:
            continue

        keys = [k for ks in fams.values() for k in ks]
        sev = _severity(cal, num, keys)
        text_seen = (card.params or {}).get("text") or ""

        # --- duplication may accuse alone, but only at the strong tier -----
        # Receipts legitimately repeat short strings ("0.00", "1.00") in the
        # same font and rasteriser, so screen-tier duplication is ordinary; at
        # screen tier it counts as one family like everything else. A match
        # beyond any authentic document's best self-match is another matter.
        if "duplication" in fams_strong and num.get("dup_distance_px", 0) >= 40:
            n += 1
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the region at {card.bbox} reading '{text_seen}' is a "
                     f"near-identical copy of another region of this same document",
                bond=[card.id],
                bond_detail={"signals": ["dup_exceeds"], "families": ["duplication"],
                             "severity": _severity(cal, num, ["dup_exceeds"], tier="strong"),
                             "tiers": tiers, "rule": "duplication_strong_tier"}))
            continue

        # --- arithmetic corroborated by a pixel anomaly at the same field --
        # Duplication cannot be the corroborating family here: printing the
        # total twice (subtotal line, total line, amount tendered) is the
        # single most common legitimate receipt layout, so a high dup score
        # AT the total is expected, not incriminating.
        arith_fams = {f: ks for f, ks in fams.items() if f != "duplication"}
        if (arithmetic_card is not None and card.bbox and arith_fams
                and arithmetic_card.bbox and _overlaps(card.bbox, arithmetic_card.bbox)):
            arith_keys = [k for ks in arith_fams.values() for k in ks]
            n += 1
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the stated total was altered: the arithmetic does not "
                     f"reconcile and the total field itself shows "
                     f"{', '.join(SIGNAL_TEXT[k] for k in arith_keys[:2])}",
                bond=[card.id, arithmetic_card.id],
                bond_detail={"signals": arith_keys, "families": sorted(arith_fams),
                             "severity": sev, "tiers": tiers,
                             "rule": "arithmetic_plus_pixel"}))
            continue

        # --- two independent families at the screen tier -------------------
        # Each screen bound passes ~2% of honest regions; two *independent*
        # families crossing on the same region is what makes this rare enough
        # to say out loud. Restricted to motivated regions (amounts, the
        # arithmetic target, unreadable patches) because decorative text -
        # logos, bold headers - is legitimately unusual on every axis at once
        # and a forger has no reason to have touched it.
        if len(fams) >= 2 and _is_motivated(card):
            n += 1
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the region at {card.bbox} reading '{text_seen}' was "
                     f"digitally altered: " +
                     "; ".join(SIGNAL_TEXT[k] for k in keys[:3]),
                bond=[card.id],
                bond_detail={"signals": keys, "families": sorted(fams),
                             "severity": sev, "tiers": tiers,
                             "rule": "two_independent_families"}))
            continue

        # --- one family, but past the strong (per-document-extreme) tier ---
        # The strong bound is set on the distribution of each authentic
        # document's WORST region, so a value past it is beyond anything the
        # calibration corpus produced even once - individually damning, no
        # corroboration needed.
        if fams_strong:
            strong_keys = [k for ks in fams_strong.values() for k in ks]
            n += 1
            claims.append(Claim(
                id=f"C-{n}",
                text=f"the region at {card.bbox} reading '{text_seen}' shows "
                     f"{SIGNAL_TEXT[strong_keys[0]]}, beyond the worst region of "
                     f"any document in this customer's authentic corpus",
                bond=[card.id],
                bond_detail={"signals": strong_keys, "families": sorted(fams_strong),
                             "severity": _severity(cal, num, strong_keys, tier="strong"),
                             "tiers": tiers, "rule": "single_family_strong_tier"}))
            continue

        reasoning.append(
            f"{card.id}: {len(fams)} family at screen tier (severity {sev}) - "
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
