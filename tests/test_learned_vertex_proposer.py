"""Smoke tests for LearnedVertexProposer's graceful-degradation contract.

The PR #142 contract: the proposer must return a benign Proposal
(no exception) when EITHER the trained model file is absent OR the
optional `rembg` dependency is missing. Devin's audit caught the
original import ordering as a real bug — these tests pin the
behavior so it can't regress.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools.propose_geometry_labels import (  # noqa: E402
    LearnedVertexProposer,
    Proposal,
)


def _make_target(side: str = "A"):
    """Minimal LabelTarget stand-in: just the attributes propose() reads
    before the model-state / rembg branches kick in. We never reach the
    rembg call in these tests."""
    target = types.SimpleNamespace()
    target.image = None
    target.load = lambda: None
    target.gt_face_quads = {"U": [], "R": [], "F": []} if side == "A" else {"D": [], "L": [], "B": []}
    return target


def test_absent_model_returns_no_trained_model(monkeypatch):
    """When runs/vertex_regressor.pkl is absent, propose() must return
    Proposal(notes={"reason": "no_trained_model"}) — no exception."""
    # Pin the class state to None and patch _get_model_state to return None
    # (simulating an absent .pkl regardless of what's actually on disk).
    monkeypatch.setattr(LearnedVertexProposer, "_model_state", None)
    monkeypatch.setattr(LearnedVertexProposer, "_get_model_state", classmethod(lambda cls: None))

    proposal = LearnedVertexProposer().propose(_make_target())

    assert isinstance(proposal, Proposal)
    assert proposal.notes.get("reason") == "no_trained_model"


def test_absent_rembg_returns_rembg_unavailable(monkeypatch):
    """When the model is present but `rembg` import fails, propose() must
    return Proposal(notes={"reason": "rembg_unavailable"}) — no
    ModuleNotFoundError leaks out. This is the exact failure mode
    Devin demonstrated on PR #142 head SHA 92c7d34."""
    # Make the model state look present (any non-None dict suffices —
    # we never reach the actual prediction step because rembg is gone).
    fake_state = {"model": object(), "model_name": "fake", "training_samples": 0}
    monkeypatch.setattr(LearnedVertexProposer, "_get_model_state", classmethod(lambda cls: fake_state))

    # Force `from rembg import remove` to ImportError. sys.modules-level
    # block is the standard pytest pattern for absent optional deps.
    monkeypatch.setitem(sys.modules, "rembg", None)

    proposal = LearnedVertexProposer().propose(_make_target())

    assert isinstance(proposal, Proposal)
    assert proposal.notes.get("reason") == "rembg_unavailable"


def test_absent_model_takes_precedence_over_absent_rembg(monkeypatch):
    """When BOTH model and rembg are absent, the proposer should report
    no_trained_model (the more actionable signal — train the model)
    rather than rembg_unavailable (a separate install fix)."""
    monkeypatch.setattr(LearnedVertexProposer, "_get_model_state", classmethod(lambda cls: None))
    monkeypatch.setitem(sys.modules, "rembg", None)

    proposal = LearnedVertexProposer().propose(_make_target())

    assert proposal.notes.get("reason") == "no_trained_model"
