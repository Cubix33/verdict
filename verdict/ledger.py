"""Append-only, content-addressed evidence store backed by SQLite and a crops directory."""
from __future__ import annotations
import json, pathlib, sqlite3, time
import cv2
from .types import EvidenceCard, sha256_bytes


class Ledger:
    def __init__(self, root: str = "data", db: str = "data/ledger.db"):
        self.root = pathlib.Path(root)
        (self.root / "crops").mkdir(parents=True, exist_ok=True)
        dbp = pathlib.Path(db)
        dbp.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(dbp), check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS cards(
            sha TEXT PRIMARY KEY, claim_id TEXT, card_id TEXT, probe TEXT,
            params TEXT, bbox TEXT, px INTEGER, obs TEXT, numeric TEXT,
            crop_path TEXT, cost REAL, ts REAL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS claims(
            claim_id TEXT PRIMARY KEY, phash TEXT, verdict TEXT, ts REAL)""")
        self.db.commit()
        self._cache: dict[str, EvidenceCard] = {}

    def add(self, claim_id: str, card_id: str, probe: str, params: dict,
            bbox, crop_bgr, observation: str, numeric: dict, cost: float) -> EvidenceCard:
        ok, buf = cv2.imencode('.png', crop_bgr)
        assert ok, 'crop encode failed'
        sha = sha256_bytes(buf.tobytes())
        path = self.root / 'crops' / f"{sha[:16]}.png"
        if not path.exists():
            path.write_bytes(buf.tobytes())
        card = EvidenceCard(id=card_id, probe=probe, params=params, bbox=bbox,
                            px_on_target=int(crop_bgr.shape[0] * crop_bgr.shape[1]),
                            crop_sha256=sha, crop_path=str(path),
                            observation=observation, numeric=numeric, cost_units=cost)
        self.db.execute("INSERT OR IGNORE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (sha, claim_id, card_id, probe, json.dumps(params),
                         json.dumps(bbox), card.px_on_target, observation,
                         json.dumps(numeric), str(path), cost, card.ts))
        self.db.commit()
        self._cache[card_id] = card
        return card

    def get(self, card_id: str) -> EvidenceCard | None:
        return self._cache.get(card_id)

    def verify_provenance(self, card_id: str) -> bool:
        """A card must exist AND its bytes must still hash to its recorded sha."""
        c = self._cache.get(card_id)
        if c is None or not c.crop_path:
            return False
        return sha256_bytes(pathlib.Path(c.crop_path).read_bytes()) == c.crop_sha256

    def manifest(self, ids=None) -> str:
        cards = [c for k, c in self._cache.items() if ids is None or k in ids]
        return "\n".join(c.manifest_line() for c in sorted(cards, key=lambda c: c.id))
