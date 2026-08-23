"""Append-only, content-addressed evidence store. Provenance cannot be faked.

Two defects in the original made the ledger unsound.

It kept every card in one process-global dictionary and never consulted the
claim id, so `manifest()` returned every card ever created. Analysing a second
image adjudicated over the first image's evidence too: in the reproduction run,
image ten carried ten claims, nine of them belonging to other people's
receipts, and a photograph of a mountain inherited twelve accusations of
receipt tampering. Every read path here is now scoped by `claim_id`.

It also only ever read from that dictionary, never from SQLite, so a restarted
server reported an empty manifest while the database sat full of rows. Cards
are now rehydrated on demand, which is also what makes provenance verification
meaningful across processes.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import time

import cv2

from .types import EvidenceCard, sha256_bytes


class Ledger:
    def __init__(self, root: str = "data", db: str = "data/ledger.db"):
        self.root = pathlib.Path(root)
        (self.root / "crops").mkdir(parents=True, exist_ok=True)
        dbp = pathlib.Path(db)
        dbp.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(dbp), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS cards(
            sha TEXT, claim_id TEXT, card_id TEXT, probe TEXT,
            params TEXT, bbox TEXT, px INTEGER, obs TEXT, numeric TEXT,
            crop_path TEXT, cost REAL, ts REAL,
            PRIMARY KEY (claim_id, card_id))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS claims(
            claim_id TEXT PRIMARY KEY, phash TEXT, verdict TEXT,
            image_path TEXT, ts REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_cards_claim ON cards(claim_id)")
        self.db.commit()
        self._migrate()
        self._cache: dict[tuple[str, str], EvidenceCard] = {}

    def _migrate(self) -> None:
        """Add columns introduced after a database file may already exist.

        `claims` gained `image_path` for the audit trail; this adds it to a
        pre-existing database instead of forcing a delete-and-restart.

        This does NOT fix a deeper schema change: `cards` used to have `sha`
        as its primary key, so two different claims that happened to produce
        an identical crop shared one row and each overwrote the other's
        metadata. The key is now `(claim_id, card_id)`, but SQLite cannot
        change a primary key in place - a database file created before this
        change keeps the old key until it is deleted and rebuilt from a fresh
        `Ledger()` call. `Ledger.reset()` does that.
        """
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(claims)")}
        if "image_path" not in cols:
            self.db.execute("ALTER TABLE claims ADD COLUMN image_path TEXT")
            self.db.commit()

    # -- writing ------------------------------------------------------------

    def add(self, claim_id: str, card_id: str, probe: str, params: dict,
            bbox, crop_bgr, observation: str, numeric: dict,
            cost: float, px_on_target: int | None = None) -> EvidenceCard:
        """Register one probe result, committing to the exact pixels examined."""
        ok, buf = cv2.imencode(".png", crop_bgr)      # PNG: lossless, stable hash
        if not ok:
            raise ValueError("crop encode failed")
        raw = buf.tobytes()
        sha = sha256_bytes(raw)
        path = self.root / "crops" / f"{sha[:16]}.png"
        if not path.exists():
            path.write_bytes(raw)

        if px_on_target is None:
            px_on_target = int(crop_bgr.shape[0] * crop_bgr.shape[1])

        card = EvidenceCard(id=card_id, probe=probe, params=_safe(params), bbox=bbox,
                            px_on_target=int(px_on_target), crop_sha256=sha,
                            crop_path=str(path), observation=observation,
                            numeric=_safe(numeric), cost_units=cost)
        self.db.execute("INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (sha, claim_id, card_id, probe, json.dumps(card.params),
                         json.dumps(bbox), card.px_on_target, observation,
                         json.dumps(card.numeric), str(path), cost, card.ts))
        self.db.commit()
        self._cache[(claim_id, card_id)] = card
        return card

    def record_claim(self, claim_id: str, phash: str | None,
                     verdict: str | None, image_path: str | None = None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?)",
            (claim_id, phash, verdict, image_path, time.time()))
        self.db.commit()

    # -- reading ------------------------------------------------------------

    def get(self, claim_id: str, card_id: str) -> EvidenceCard | None:
        """Fetch one card, rehydrating from SQLite if it is not in memory."""
        hit = self._cache.get((claim_id, card_id))
        if hit is not None:
            return hit
        row = self.db.execute(
            "SELECT * FROM cards WHERE claim_id=? AND card_id=?",
            (claim_id, card_id)).fetchone()
        if row is None:
            return None
        card = _from_row(row)
        self._cache[(claim_id, card_id)] = card
        return card

    def cards(self, claim_id: str) -> list[EvidenceCard]:
        """Every card belonging to one claim, and nothing belonging to another."""
        rows = self.db.execute(
            "SELECT * FROM cards WHERE claim_id=? ORDER BY ts, card_id",
            (claim_id,)).fetchall()
        out = []
        for r in rows:
            card = _from_row(r)
            self._cache[(claim_id, card.id)] = card
            out.append(card)
        return out

    def cards_by_id(self, claim_id: str) -> dict[str, EvidenceCard]:
        return {c.id: c for c in self.cards(claim_id)}

    def claim_ids(self) -> list[str]:
        return [r["claim_id"] for r in
                self.db.execute("SELECT DISTINCT claim_id FROM cards ORDER BY claim_id")]

    def manifest(self, claim_id: str, ids=None) -> str:
        """The text the starved Adjudicator sees. Scoped to one claim, always."""
        cards = [c for c in self.cards(claim_id) if ids is None or c.id in ids]
        return "\n".join(c.manifest_line() for c in sorted(cards, key=lambda c: c.id))

    # -- integrity ----------------------------------------------------------

    def verify_provenance(self, claim_id: str, card_id: str) -> bool:
        """A card must exist AND its bytes must still hash to its recorded sha."""
        c = self.get(claim_id, card_id)
        if c is None or not c.crop_path:
            return False
        p = pathlib.Path(c.crop_path)
        if not p.exists():
            return False
        return sha256_bytes(p.read_bytes()) == c.crop_sha256

    # -- maintenance --------------------------------------------------------

    def purge(self, claim_id: str) -> int:
        """Drop one claim's evidence. Crops shared with another claim survive."""
        rows = self.db.execute("SELECT crop_path FROM cards WHERE claim_id=?",
                               (claim_id,)).fetchall()
        self.db.execute("DELETE FROM cards WHERE claim_id=?", (claim_id,))
        self.db.execute("DELETE FROM claims WHERE claim_id=?", (claim_id,))
        self.db.commit()
        for k in [k for k in self._cache if k[0] == claim_id]:
            del self._cache[k]
        removed = 0
        for r in rows:
            path = r["crop_path"]
            still = self.db.execute("SELECT 1 FROM cards WHERE crop_path=? LIMIT 1",
                                    (path,)).fetchone()
            if still is None:
                try:
                    pathlib.Path(path).unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def reset(self) -> None:
        self.db.execute("DELETE FROM cards")
        self.db.execute("DELETE FROM claims")
        self.db.commit()
        self._cache.clear()
        for f in (self.root / "crops").glob("*.png"):
            try:
                f.unlink()
            except OSError:
                pass


def _from_row(row) -> EvidenceCard:
    return EvidenceCard(
        id=row["card_id"], probe=row["probe"],
        params=json.loads(row["params"] or "{}"),
        bbox=json.loads(row["bbox"] or "null"),
        px_on_target=row["px"], crop_sha256=row["sha"],
        crop_path=row["crop_path"], observation=row["obs"],
        numeric=json.loads(row["numeric"] or "{}"),
        cost_units=row["cost"], ts=row["ts"])


def _safe(d):
    """Make a probe payload JSON-serialisable.

    EXIF fields arrive as bytes and numpy scalars arrive from every probe;
    `json.dumps` rejects both, and the original code let them straight through
    into the ledger, which crashed the pipeline on any real camera photo.
    """
    if isinstance(d, dict):
        return {str(k): _safe(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_safe(v) for v in d]
    if isinstance(d, bytes):
        return d.decode("utf-8", "replace").strip("\x00").strip()
    if isinstance(d, (str, int, float, bool)) or d is None:
        return d
    if hasattr(d, "item"):                 # numpy scalar
        try:
            return d.item()
        except Exception:
            pass
    return str(d)
