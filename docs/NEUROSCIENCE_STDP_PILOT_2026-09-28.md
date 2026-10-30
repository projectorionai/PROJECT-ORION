# Neuroscience Research: provisional literature pilot

**Mission:** Neuroscience Research  
**Question:** In two foundational experimental studies, how did relative spike timing relate to synaptic efficacy changes?  
**Status:** Curated PubMed abstract snapshot, reviewed 28 September 2026. This is a provisional question until the user selects a research direction.  
**Expected output:** A brief in which each claim has a source ID, PMID, DOI, publication date, access date, experimental preparation and supporting abstract data.  
**Success criterion:** The `research` tool's `neuroscience_pilot` action returns the brief with claim-level evidence, rejects unrelated questions, and leaves the mission ledger untouched.

The runtime action uses the existing `research` dispatcher route. Call it with `{"action":"neuroscience_pilot"}` or `{"action":"neuroscience_pilot","topic":"STDP"}`. It reads the existing mission record when available. The source snapshot is bundled in [`neuroscience_research.py`](../orion_core/neuroscience_research.py); it does not claim a live search or write to mission or memory stores.

## Evidence-linked answer

- **C1.** Markram and colleagues reported both increases and decreases in EPSP amplitude depending on the timing of postsynaptic action potentials relative to EPSPs in their neocortical preparation. [S1]
- **C2.** Bi and Poo reported potentiation when postsynaptic spikes followed presynaptic activation within 20 ms, and depression for the opposite order within 20 ms, in dissociated rat hippocampal cultures. [S2]
- **C3.** Their reported effects also depended on initial synaptic strength and postsynaptic cell type. [S2]

These claims are limited to the preparations and protocols in the cited papers. They do not establish a universal rule for human synapses or memory. No simulation has been run.

## Sources and supporting abstract data

- **S1.** Markram H, Lübke J, Frotscher M, Sakmann B. “[Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs](https://pubmed.ncbi.nlm.nih.gov/8985014/).” *Science*, published **1997-01-10**; accessed **2026-09-28**. PMID **8985014**; DOI **10.1126/science.275.5297.213**. The PubMed abstract describes paired recordings from pyramidal neurons and reports differential up- and down-regulation of average EPSP amplitude with relative AP/EPSP timing.
- **S2.** Bi GQ, Poo MM. “[Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type](https://pubmed.ncbi.nlm.nih.gov/9852584/).” *Journal of Neuroscience*, published **1998-12-15**; accessed **2026-09-28**. PMID **9852584**; DOI **10.1523/JNEUROSCI.18-24-10464.1998**. The PubMed abstract identifies dissociated rat hippocampal cultures, the two 20 ms timing windows, and dependence on initial synaptic strength and postsynaptic cell type.

## Scope of the next research pass

The pilot demonstrates provenance and a bounded answer. It has not searched for replication, later contradictory experiments, in-vivo evidence, or human applicability. A broader review should retrieve and inspect those sources before drawing a wider conclusion. Simulator selection remains deferred until a question that requires one is chosen.
