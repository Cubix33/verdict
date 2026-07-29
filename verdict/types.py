"""Core types for the VERDICT system: EvidenceCard, Claim, Verdict, and helpers."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, field, asdict
from typing import Any


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class EvidenceCard:
    """One probe result. The hash commits to the EXACT pixel bytes examined."""
    id: str
    probe: str
    params: dict
    bbox: list | None
    px_on_target: int
    crop_sha256: str            # sha256 of the PNG-encoded crop -> the bond anchor
    crop_path: str | None
    observation: str            # short, factual, machine-generated
    numeric: dict               # z-scores, ratios etc. -- what the Adjudicator reasons over
    cost_units: float
    ts: float = field(default_factory=time.time)

    def manifest_line(self) -> str:
        """The ONLY representation the starved Adjudicator ever sees."""
        n = " ".join(f"{k}={v}" for k, v in self.numeric.items())
        return (f"{self.id}  {self.probe:<14} px={self.px_on_target:<6} "
                f"sha={self.crop_sha256[:8]}  {n}  | {self.observation}")


@dataclass
class Claim:
    id: str
    text: str
    bond: list[str]
    bond_status: str = "untested"   # untested | holds | broken
    bond_detail: dict = field(default_factory=dict)


@dataclass
class Verdict:
    claim_id: str
    verdict: str                # tamper_detected | authentic | synthetic_evidence
                                # | insufficient_evidence | undetermined
    confidence: float | None
    claims: list[Claim]
    controls: dict
    cost: dict
    latency_s: float
    evidence_chain: list[str]

    def to_dict(self):
        d = asdict(self); d["claims"] = [asdict(c) for c in self.claims]; return d

    @property
    def bonded_claims(self):
        return [c for c in self.claims if c.bond_status == "holds"]

    @property
    def is_bonded(self):
        """A verdict only counts if it has at least one surviving bonded claim."""
        return len(self.bonded_claims) > 0
