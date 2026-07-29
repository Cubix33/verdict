"""Adjudicator (starved) — produces grammar-constrained claims from the evidence manifest.

MVP behaviour:
- Reads ledger.manifest() text and ledger._cache to see numeric fields
- If any EvidenceCard numeric contains 'delta' with abs(delta) > 0.5 -> emit claim 'total_field_altered' bonded to that card
- If any card numeric contains 'z_score' > 3.0 -> emit 'local_ela_anomaly' bonded to that card

The Adjudicator only uses the manifest / metadata and never loads full images.
"""
from __future__ import annotations
from typing import List
from ..ledger import Ledger
from ..types import Claim


def adjudicate_from_ledger(ledger: Ledger) -> dict:
    manifest = ledger.manifest()
    # starved: only use manifest text + numeric visible in each card object
    claims: List[Claim] = []
    for cid, card in ledger._cache.items():
        num = card.numeric or {}
        # check cross_field delta
        if 'delta' in num and num.get('delta') is not None:
            try:
                delta = float(num.get('delta') or 0.0)
            except Exception:
                delta = 0.0
            if abs(delta) > 0.5:
                claims.append(Claim(id=f"C-{cid}", text="total field digitally altered", bond=[card.id]))
        # check ELA z-score
        if 'z_score' in num and num.get('z_score') is not None:
            try:
                z = float(num.get('z_score') or 0.0)
            except Exception:
                z = 0.0
            if z > 3.0:
                claims.append(Claim(id=f"C-ELA-{cid}", text="localized ELA anomaly", bond=[card.id]))
    verdict = {"verdict": "undetermined", "claims": [c.__dict__ for c in claims], "manifest": manifest}
    if claims:
        verdict['verdict'] = 'tamper_detected'
    return verdict


if __name__ == '__main__':
    import json, sys
    ledger = Ledger(root='data', db='data/ledger.db')
    r = adjudicate_from_ledger(ledger)
    print(json.dumps(r, indent=2))
