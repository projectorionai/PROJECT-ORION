"""
NeuroKnowledgeBase — ORION's resident expertise in neuroscience and neural
engineering.

Two jobs:

    1. A curated, offline corpus of substantive facts across neuroanatomy,
       neurophysiology, computational neuroscience, brain–computer interfaces,
       neural signal processing and neuroprosthetics.  It is seeded into the
       MemoryAgent's KNOWLEDGE tier at startup (idempotently) so it surfaces in
       ``prompt_context`` for the live model *and* can be recalled with zero
       API calls when the cloud is unavailable.

    2. A local retrieval + answer path (``answer``) the offline LocalBrain uses
       to discuss the domain without any provider — so ORION remains a genuine
       specialist even with the network down.

The corpus is deliberately mechanism-first and hedged where evidence is
uncertain, matching ORION's briefed scientific manner.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Persona reinforcement appended to the system instruction.
PERSONA_BOOST = (
    "You carry deep, working expertise in neuroscience and neural engineering: "
    "neuroanatomy, neurophysiology (membrane biophysics, action potentials, "
    "synaptic transmission, neurotransmitter systems), systems and cognitive "
    "neuroscience, computational neuroscience (Hodgkin–Huxley, integrate-and-fire, "
    "cable theory, plasticity rules), brain–computer interfaces (EEG, ECoG, "
    "intracortical microelectrode arrays such as the Utah array, Neuralink's "
    "N1 threads), neural signal processing (spike detection and sorting, LFP, "
    "population decoding), and neuroprosthetics. Reason from mechanism, state "
    "the level of evidence, distinguish established findings from speculation, "
    "and be ready to critique methods, design experiments and generate "
    "hypotheses. Use correct terminology precisely but explain it plainly."
)

# topic slug → (title, fact).  Facts are concise and citable-in-spirit.
_CORPUS: dict[str, tuple[str, str]] = {
    "neuron": ("The neuron",
        "The neuron is the brain's signalling unit: dendrites integrate inputs, "
        "the soma sums them, and the axon propagates all-or-none action "
        "potentials to synaptic terminals. The human brain holds on the order of "
        "86 billion neurons and a comparable number of glia."),
    "glia": ("Glial cells",
        "Glia outnumber or roughly equal neurons and are not mere support: "
        "astrocytes regulate the extracellular milieu and tripartite synapses, "
        "oligodendrocytes (and Schwann cells in the periphery) myelinate axons, "
        "and microglia are the brain's resident immune cells."),
    "action_potential": ("The action potential",
        "An action potential is a ~1 ms, ~100 mV depolarising spike. At threshold "
        "(around −55 mV) voltage-gated sodium channels open and drive the rising "
        "phase; their inactivation plus delayed potassium efflux repolarises the "
        "membrane, overshooting to a brief hyperpolarisation. It is all-or-none "
        "and regenerated along the axon."),
    "resting_potential": ("Resting membrane potential",
        "The resting potential (~ −70 mV) is set by selective K+ permeability and "
        "the Na+/K+ ATPase, and is well described by the Goldman–Hodgkin–Katz "
        "equation. It is a steady state, not equilibrium — the pump continuously "
        "counters leak."),
    "synapse": ("Synaptic transmission",
        "At a chemical synapse, a presynaptic action potential opens voltage-gated "
        "Ca2+ channels; calcium triggers vesicle fusion and neurotransmitter "
        "release, which binds postsynaptic receptors — ionotropic (fast, e.g. "
        "AMPA/NMDA, GABA-A) or metabotropic (slower, G-protein coupled)."),
    "neurotransmitters": ("Neurotransmitter systems",
        "Glutamate is the main excitatory transmitter, GABA the main inhibitory "
        "one. Modulatory systems — dopamine (reward, motor control), serotonin "
        "(mood, arousal), acetylcholine (attention, plasticity), noradrenaline "
        "(vigilance) — reconfigure whole circuits rather than carrying fast "
        "point-to-point signals."),
    "plasticity": ("Synaptic plasticity",
        "Learning is thought to be stored as changes in synaptic weight. "
        "Long-term potentiation and depression, and spike-timing-dependent "
        "plasticity (pre-before-post strengthens, the reverse weakens), implement "
        "Hebb's principle — 'cells that fire together wire together' — largely "
        "through NMDA-receptor-gated calcium signalling."),
    "cortex": ("Cerebral cortex",
        "The neocortex is a ~2–4 mm sheet of six laminae organised into columns. "
        "Broadly: occipital lobe for vision, temporal for audition and memory, "
        "parietal for somatosensation and spatial processing, frontal for motor "
        "control and executive function. The primary motor cortex and "
        "somatosensory cortex are somatotopically mapped (the homunculus)."),
    "hippocampus": ("Hippocampus",
        "The hippocampus is critical for forming new declarative memories and for "
        "spatial navigation; it contains place cells, and the neighbouring "
        "entorhinal cortex contains grid cells (O'Keefe, and Moser & Moser — "
        "Nobel Prize 2014). Bilateral damage causes profound anterograde amnesia "
        "(the patient H.M.)."),
    "hodgkin_huxley": ("Hodgkin–Huxley model",
        "The 1952 Hodgkin–Huxley model quantitatively describes the squid giant "
        "axon action potential with voltage- and time-dependent Na+ and K+ "
        "conductances (gating variables m, h, n). It remains the foundation of "
        "biophysical neuron modelling and won the 1963 Nobel Prize."),
    "integrate_fire": ("Integrate-and-fire neurons",
        "The leaky integrate-and-fire model abstracts a neuron as an RC circuit "
        "that integrates input current and emits a spike when the membrane "
        "crosses threshold, then resets. It is cheap enough for large-scale "
        "network simulation and underlies much of computational neuroscience and "
        "spiking neural networks."),
    "cable_theory": ("Cable theory",
        "Cable theory treats dendrites and axons as leaky cables, giving the "
        "length constant λ (how far voltage spreads passively) and the time "
        "constant τ. It explains dendritic integration, signal attenuation and "
        "why myelination and node spacing set conduction velocity."),
    "bci": ("Brain–computer interfaces",
        "A BCI records neural activity, decodes intended state or movement, and "
        "acts on it — restoring communication or control. The pipeline is "
        "acquisition → feature extraction → decoding (often a Kalman filter or "
        "neural network) → effector, ideally with feedback closing the loop."),
    "eeg": ("EEG",
        "Electroencephalography records summed cortical postsynaptic potentials "
        "at the scalp: non-invasive and millisecond-fast but spatially blurred "
        "(centimetres) and noisy. Standard rhythms: delta, theta, alpha, beta, "
        "gamma. It powers affordable non-invasive BCIs (e.g. P300 spellers, "
        "motor-imagery control)."),
    "ecog": ("ECoG",
        "Electrocorticography places electrodes on the cortical surface "
        "(subdural). It trades invasiveness for far better spatial resolution and "
        "high-gamma signal than EEG, and is a leading modality for speech and "
        "motor BCIs in humans."),
    "utah_array": ("Utah array",
        "The Utah array is a 4×4 mm silicon block of ~100 microelectrodes (~1 mm "
        "shanks) that penetrates cortex to record single-unit and multi-unit "
        "activity. It is the workhorse of intracortical BCIs (e.g. BrainGate), "
        "though chronic gliosis can degrade signals over months to years."),
    "neuralink": ("Neuralink",
        "Neuralink's N1 implant uses thousands of thin, flexible polymer "
        "electrode 'threads' inserted by a robotic 'sewing machine' to minimise "
        "vascular damage, with wireless, battery-powered readout. Its stated aim "
        "is high-channel-count intracortical BCIs; first-in-human implants began "
        "in 2024 for cursor control. Flexible threads target the longevity "
        "problem rigid arrays face."),
    "spike_sorting": ("Spike detection and sorting",
        "Extracellular electrodes pick up spikes from several nearby neurons. "
        "Spike sorting detects threshold crossings and clusters waveforms (by PCA "
        "features, template matching, or modern ML) to attribute spikes to "
        "putative single units — essential but error-prone, and drift over time "
        "is a core challenge."),
    "lfp": ("Local field potential",
        "The LFP is the low-frequency (<~300 Hz) extracellular signal dominated "
        "by synchronised synaptic currents in a local population. It is more "
        "stable chronically than single units and carries oscillatory and "
        "population information useful for decoding."),
    "decoding": ("Neural decoding",
        "Decoding maps neural activity to behaviour or intent. Classic motor BCIs "
        "use the population vector or a Kalman filter over firing rates to "
        "reconstruct intended velocity; modern systems use recurrent and "
        "transformer networks. The mirror problem, encoding, asks how stimuli map "
        "to activity."),
    "neuroprosthetics": ("Neuroprosthetics",
        "Neuroprosthetics substitute or restore neural function: cochlear "
        "implants (the most successful, electrically stimulating the auditory "
        "nerve), retinal implants, deep brain stimulation for Parkinson's and "
        "essential tremor, and motor prostheses driving robotic limbs or "
        "reanimating muscle via functional electrical stimulation."),
    "dbs": ("Deep brain stimulation",
        "DBS delivers high-frequency electrical stimulation via implanted "
        "electrodes (e.g. subthalamic nucleus or globus pallidus) to treat "
        "Parkinson's disease, essential tremor and dystonia, and is under study "
        "for OCD and depression. Its mechanism is still debated — likely a mix of "
        "local inhibition and network modulation."),
    "connectome": ("Connectomics",
        "The connectome is the complete wiring diagram of a nervous system. C. "
        "elegans (302 neurons) is fully mapped; the fruit-fly brain connectome "
        "was completed recently. Mammalian connectomes remain partial — the scale "
        "gap (a cubic millimetre of cortex holds ~50,000 neurons and a billion "
        "synapses) is enormous."),
    "neurovascular": ("Neurovascular coupling / fMRI",
        "Active neurons increase local blood flow (the BOLD signal fMRI measures). "
        "fMRI gives millimetre spatial resolution over the whole brain but only "
        "second-scale temporal resolution, and it is an indirect, haemodynamic "
        "proxy for neural activity, not a direct electrical measure."),
    "stimulation": ("Neural stimulation",
        "Beyond recording, BCIs can write in: intracortical microstimulation can "
        "evoke artificial sensation (restoring touch to prosthetic hands), and "
        "non-invasive TMS and tDCS modulate cortical excitability for research "
        "and therapy. Safe charge-density limits and electrode materials are key "
        "engineering constraints."),
}


class NeuroKnowledgeBase:
    """Curated neuroscience/neural-engineering corpus with local retrieval."""

    SEED_MARKER_KEY = "neuro_corpus_seeded_v1"

    def __init__(self, telemetry: Any | None = None) -> None:
        self.telemetry = telemetry
        # Pre-tokenise each entry once for cheap keyword scoring.
        self._tokens: dict[str, set[str]] = {
            slug: self._tokenise(f"{title} {fact}")
            for slug, (title, fact) in _CORPUS.items()
        }

    # ── seeding into the KNOWLEDGE tier ───────────────────────────────────────

    def seed(self, memory: Any) -> int:
        """Idempotently write the corpus into persistent KNOWLEDGE memory."""
        try:
            existing = memory.recall("knowledge", limit=200)
            if any(r.get("key_ref") == self.SEED_MARKER_KEY for r in existing):
                return 0
        except Exception:
            pass
        written = 0
        for slug, (title, fact) in _CORPUS.items():
            try:
                memory.remember("knowledge", f"neuro_{slug}", f"{title}: {fact}")
                written += 1
            except Exception:
                continue
        try:
            memory.remember("knowledge", self.SEED_MARKER_KEY,
                            f"Neuroscience corpus seeded ({written} entries).")
        except Exception:
            pass
        if self.telemetry is not None:
            self.telemetry.log.info("KNOWLEDGE", f"neuroscience corpus seeded", entries=written)
        return written

    # ── retrieval ─────────────────────────────────────────────────────────────

    def topics(self) -> list[str]:
        return [title for title, _ in _CORPUS.values()]

    def search(self, query: str, limit: int = 3) -> list[tuple[str, str]]:
        q = self._tokenise(query)
        if not q:
            return []
        scored: list[tuple[int, str]] = []
        for slug, tokens in self._tokens.items():
            overlap = len(q & tokens)
            # Boost direct title-word hits.
            title = _CORPUS[slug][0].lower()
            if any(word in title for word in q):
                overlap += 2
            if overlap:
                scored.append((overlap, slug))
        scored.sort(reverse=True)
        return [_CORPUS[slug] for _, slug in scored[:limit]]

    def answer(self, query: str) -> Optional[str]:
        """A spoken-ready answer assembled from the best matches, or None."""
        hits = self.search(query, limit=2)
        if not hits:
            return None
        if len(hits) == 1 or self._tokenise(query) & self._tokenise(hits[0][0]):
            return hits[0][1]
        primary, secondary = hits[0], hits[1]
        return f"{primary[1]} Relatedly, {secondary[1]}"

    def is_neuro_query(self, text: str) -> bool:
        return bool(self._NEURO_RE.search(text))

    _NEURO_RE = re.compile(
        r"(?i)\b(neuro\w*|neural|neuron|brain|cortex|cortical|synap\w*|axon|dendrit\w*|"
        r"action potential|spik\w*|bci|brain.?computer|eeg|ecog|electrode|neuralink|"
        r"utah array|hippocamp\w*|dopamine|serotonin|glutamate|gaba|plasticity|"
        r"hodgkin|prosthe\w*|stimulation|dbs|connectome|glia|myelin|membrane potential)\b"
    )

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        stop = {"the", "a", "an", "of", "and", "or", "is", "are", "to", "in", "on",
                "for", "with", "what", "how", "why", "does", "do", "tell", "me",
                "about", "explain", "that", "this", "it", "you", "can"}
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if len(w) >= 3 and w not in stop}
