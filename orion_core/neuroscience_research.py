"""A bounded, source-backed literature pilot for the Neuroscience Research mission.

This is a reviewed PubMed abstract snapshot, not a live literature search or a
scientific validation service. It performs no network access and writes no
mission, memory, or research state. New questions need a separate source review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping
from urllib.parse import urlparse


MISSION_NAME = "Neuroscience Research"
QUESTION = (
    "In two foundational experimental studies, how did relative spike timing "
    "relate to synaptic efficacy changes?"
)
SNAPSHOT_DATE = "2026-09-28"


@dataclass(frozen=True)
class LiteratureSource:
    source_id: str
    pmid: str
    title: str
    authors: str
    publication_date: str
    accessed_date: str
    doi: str
    url: str
    preparation: str
    supporting_data: str


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    statement: str
    source_ids: tuple[str, ...]
    limitation: str


# Supporting data paraphrase the cited PubMed abstracts. They are deliberately
# narrow; the source ID and the experimental preparation travel with every claim.
SOURCES = (
    LiteratureSource(
        source_id="S1",
        pmid="8985014",
        title="Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs",
        authors="Markram H, Lübke J, Frotscher M, Sakmann B",
        publication_date="1997-01-10",
        accessed_date=SNAPSHOT_DATE,
        doi="10.1126/science.275.5297.213",
        url="https://pubmed.ncbi.nlm.nih.gov/8985014/",
        preparation="Paired whole-cell recordings from neocortical pyramidal neurons",
        supporting_data=(
            "The abstract reports both increases and decreases in mean EPSP "
            "amplitude as the timing of the postsynaptic action potential "
            "relative to the EPSP changed."
        ),
    ),
    LiteratureSource(
        source_id="S2",
        pmid="9852584",
        title=(
            "Synaptic modifications in cultured hippocampal neurons: dependence "
            "on spike timing, synaptic strength, and postsynaptic cell type"
        ),
        authors="Bi GQ, Poo MM",
        publication_date="1998-12-15",
        accessed_date=SNAPSHOT_DATE,
        doi="10.1523/JNEUROSCI.18-24-10464.1998",
        url="https://pubmed.ncbi.nlm.nih.gov/9852584/",
        preparation="Dissociated rat hippocampal neuron cultures",
        supporting_data=(
            "The abstract reports potentiation when postsynaptic spiking followed "
            "presynaptic activation within 20 ms, and depression when it preceded "
            "presynaptic activation within 20 ms. It also reports dependence on "
            "initial synaptic strength, NMDA receptors, and postsynaptic cell type."
        ),
    ),
)

CLAIMS = (
    EvidenceClaim(
        claim_id="C1",
        statement=(
            "In the neocortical preparation, relative AP/EPSP timing was associated "
            "with both increases and decreases in EPSP amplitude."
        ),
        source_ids=("S1",),
        limitation="The abstract does not establish a universal timing window.",
    ),
    EvidenceClaim(
        claim_id="C2",
        statement=(
            "In the cultured rat hippocampal preparation, a postsynaptic spike "
            "within 20 ms after presynaptic activation was associated with LTP; "
            "the reverse order within 20 ms was associated with LTD."
        ),
        source_ids=("S2",),
        limitation="This is the reported protocol and preparation, not a general human rule.",
    ),
    EvidenceClaim(
        claim_id="C3",
        statement=(
            "The cultured-neuron results also depended on initial synaptic "
            "strength and postsynaptic cell type."
        ),
        source_ids=("S2",),
        limitation="The abstract does not quantify every cell-type effect.",
    ),
)


def validate_catalog(
    sources: tuple[LiteratureSource, ...] = SOURCES,
    claims: tuple[EvidenceClaim, ...] = CLAIMS,
) -> None:
    """Reject broken provenance before presenting any scientific statement."""
    if not sources or not claims:
        raise ValueError("The neuroscience pilot needs sources and claims.")
    by_id = {source.source_id: source for source in sources}
    if len(by_id) != len(sources):
        raise ValueError("Duplicate neuroscience source ID.")
    seen_claims: set[str] = set()
    for source in sources:
        fields = (source.source_id, source.pmid, source.title, source.authors,
                  source.doi, source.url, source.preparation, source.supporting_data)
        if not all(str(value).strip() for value in fields):
            raise ValueError(f"Incomplete source {source.source_id}.")
        for value in (source.publication_date, source.accessed_date):
            date.fromisoformat(value)
        parsed = urlparse(source.url)
        if (parsed.scheme != "https" or parsed.hostname != "pubmed.ncbi.nlm.nih.gov"
                or parsed.path != f"/{source.pmid}/"):
            raise ValueError(f"Invalid PubMed URL for {source.source_id}.")
    for claim in claims:
        if (not claim.claim_id or claim.claim_id in seen_claims
                or not claim.statement.strip() or not claim.limitation.strip()
                or not claim.source_ids or any(source_id not in by_id
                                                for source_id in claim.source_ids)):
            raise ValueError(f"Unsupported claim {claim.claim_id}.")
        seen_claims.add(claim.claim_id)


def _mission_state(missions: Any) -> str:
    """Read the existing mission record without seeding or changing it."""
    if missions is None:
        return "mission engine unavailable; pilot remains unlinked"
    try:
        data = missions.load()
        records = data.get("missions", {}) if isinstance(data, Mapping) else {}
        if isinstance(records, Mapping) and MISSION_NAME in records:
            return "existing mission record found"
    except Exception:
        pass
    return "mission record unavailable; no mission was created"


def _accepts_topic(topic: str) -> bool:
    text = " ".join(str(topic or "").casefold().split()).rstrip("?.")
    return text in {
        "",
        "stdp",
        "spike-timing-dependent plasticity",
        "spike timing dependent plasticity",
        " ".join(QUESTION.casefold().split()).rstrip("?."),
    }


def pilot_evidence() -> list[dict[str, str]]:
    """Field-level evidence for ToolResult consumers."""
    validate_catalog()
    by_id = {source.source_id: source for source in SOURCES}
    return [
        {
            "claim_id": claim.claim_id,
            "claim": claim.statement,
            "source_id": source_id,
            "pmid": by_id[source_id].pmid,
            "doi": by_id[source_id].doi,
            "url": by_id[source_id].url,
            "publication_date": by_id[source_id].publication_date,
            "accessed_date": by_id[source_id].accessed_date,
            "supporting_data": by_id[source_id].supporting_data,
            "limitation": claim.limitation,
        }
        for claim in CLAIMS for source_id in claim.source_ids
    ]


def pilot_report(*, missions: Any = None, topic: str = "") -> str:
    """Render the fixed question as a cited brief through the research tool."""
    if not _accepts_topic(topic):
        raise ValueError(
            "This source snapshot answers only the provisional STDP question. "
            "Use ordinary research for a different topic."
        )
    validate_catalog()
    lines = [
        "# Neuroscience Research — provisional literature pilot",
        "",
        f"Mission: {MISSION_NAME} ({_mission_state(missions)}).",
        f"Question: {QUESTION}",
        f"Source snapshot reviewed: {SNAPSHOT_DATE}; no live search in this run.",
        "",
        "## Evidence-linked answer",
        "",
    ]
    for claim in CLAIMS:
        source_refs = ", ".join(f"[{source_id}]" for source_id in claim.source_ids)
        lines.append(f"- {claim.claim_id}. {claim.statement} {source_refs}")
        lines.append(f"  Limit: {claim.limitation}")
    lines += [
        "",
        "## Sources and supporting abstract data",
        "",
    ]
    for source in SOURCES:
        lines += [
            f"- [{source.source_id}] {source.authors}. [{source.title}]({source.url}). "
            f"Published {source.publication_date}; accessed {source.accessed_date}. "
            f"PMID {source.pmid}; DOI {source.doi}.",
            f"  Preparation: {source.preparation}.",
            f"  Abstract data (paraphrased): {source.supporting_data}",
        ]
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These are results reported for the listed experimental preparations. "
        "The pilot has not assessed replication, later contradictory studies, "
        "in-vivo generality, or relevance to human memory. It has run no simulation.",
    ]
    return "\n".join(lines)


__all__ = ["MISSION_NAME", "QUESTION", "SNAPSHOT_DATE", "SOURCES", "CLAIMS",
           "EvidenceClaim", "LiteratureSource", "pilot_evidence", "pilot_report",
           "validate_catalog"]
