"""The provisional neuroscience brief must remain sourced and read-only."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from orion_core.neuroscience_research import (
    CLAIMS,
    QUESTION,
    SOURCES,
    pilot_evidence,
    pilot_report,
    validate_catalog,
)


def test_every_pilot_claim_has_reviewable_source_data() -> None:
    validate_catalog()
    rows = pilot_evidence()
    assert {row["claim_id"] for row in rows} == {claim.claim_id for claim in CLAIMS}
    for row in rows:
        assert row["source_id"] and row["pmid"] and row["doi"]
        assert row["url"] == f'https://pubmed.ncbi.nlm.nih.gov/{row["pmid"]}/'
        assert row["publication_date"] and row["accessed_date"]
        assert row["supporting_data"] and row["limitation"]

    brief = pilot_report()
    assert QUESTION in brief
    assert "no live search in this run" in brief
    assert "It has run no simulation" in brief
    for source in SOURCES:
        assert source.url in brief
        assert source.publication_date in brief
        assert source.accessed_date in brief
        assert source.supporting_data in brief


def test_catalog_rejects_missing_support_or_dangling_claim() -> None:
    with pytest.raises(ValueError, match="Incomplete source"):
        validate_catalog((replace(SOURCES[0], supporting_data=""),), (CLAIMS[0],))
    with pytest.raises(ValueError, match="Unsupported claim"):
        validate_catalog(SOURCES, (replace(CLAIMS[0], source_ids=("MISSING",)),))


def test_question_scope_is_explicit() -> None:
    assert QUESTION in pilot_report(topic="STDP")
    with pytest.raises(ValueError, match="only the provisional STDP question"):
        pilot_report(topic="How does human memory work?")


def test_research_route_reads_existing_mission_without_changing_it() -> None:
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    class MissionStore:
        reads = 0

        def load(self):
            self.reads += 1
            return {"missions": {"Neuroscience Research": {"brief": "user data"}}}

        def attach(self, *_args, **_kwargs):
            raise AssertionError("The pilot must not write to missions")

    class Harness(KnowledgeDispatchMixin):
        research = None
        missions = MissionStore()

    result = asyncio.run(Harness().research_tool({"action": "neuroscience_pilot"}))
    assert result.ok
    assert "existing mission record found" in result.text
    assert result.evidence and len(result.evidence) == len(CLAIMS)
    assert Harness.missions.reads == 1

    wrong = asyncio.run(Harness().research_tool(
        {"action": "neuroscience_pilot", "topic": "unrelated topic"}))
    assert not wrong.ok
    assert "unrelated topic" not in wrong.text
