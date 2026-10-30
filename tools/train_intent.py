"""Train an isolated intent-routing candidate and promote only measured gains.

This trains ORION's local action classifier, not a cloud model's weights.
Tool declarations supply training text; the fixed routing holdout supplies
evaluation only. Run with ORION closed; --apply is an explicit promotion gate.
Use --public-only to audit built-in tools without personal state or plugins.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

OOD_CASES = (
    "hello", "thanks very much", "good evening", "that was interesting",
    "how are you feeling", "tell me a harmless joke", "why is the sky blue",
    "what does curiosity mean", "explain a metaphor", "what is a prime number",
    "tell me a fictional bedtime story", "what is your favourite colour",
    "I am just thinking aloud", "hmm", "I agree with that", "that makes sense",
    "the purple teapot dreams quietly", "silver marmalade hums backwards",
    "flibbertigibbet quux frobnicate", "zqxv blorp zzzzzz",
    "teleport a chair to Mars", "reverse the orbit of Saturn",
    "make gravity disappear", "turn my laptop into a physical bicycle",
    "manufacture a star in this room", "travel backwards through time",
    "create unlimited electricity from nothing", "materialise a dragon",
    "thank you and goodbye", "no action needed", "I was only chatting",
    "please wait while I decide",
)


def evaluate(model, cases, scorer=None):
    hits = {k: 0 for k in (1, 3, 10)}
    correct_first = []
    for text, tool, also in cases:
        scores = model.top(text, k=10) if scorer is None else sorted(
            scorer(model, text).items(), key=lambda row: -row[1])[:10]
        names = [name for name, _ in scores]
        accepted = {tool, *also}
        for k in hits:
            hits[k] += bool(accepted.intersection(names[:k]))
        correct_first.append(bool(accepted.intersection(names[:1])))
    return {"cases": len(cases), "hits": hits,
            "accuracy": {k: round(n / max(1, len(cases)), 4) for k, n in hits.items()}}, correct_first


def cold_routing_scores(model, text, declarations):
    """Evaluate lexical + learned routing without loading semantic models or
    queuing missing tools. Spelling correction and warm semantic routing are
    outside this offline gate; no evaluation call may train either model.
    """
    from orion_core.intent_brain import (
        LEARNED_BOOST,
        TRUSTED_CONFIDENCE,
        TRUSTED_EXAMPLES,
    )
    from orion_core.tool_resolver import lexical_scores

    scores = dict(lexical_scores(text, declarations))
    probabilities = model.probabilities(text)
    top_lexical = max(scores.values(), default=0.0)
    for declaration in declarations:
        name = declaration["name"]
        probability = probabilities.get(name, 0.0)
        boost = LEARNED_BOOST * probability
        if (probability >= TRUSTED_CONFIDENCE
                and model.user_counts.get(name, 0) >= TRUSTED_EXAMPLES
                and model.resembles_user_phrasing(text, name)):
            boost = max(boost, probability * (top_lexical + 1.0))
        scores[name] = scores.get(name, 0.0) + boost
    return scores


def evaluate_ood(model):
    values = [max(model.probabilities(text).values(), default=0.0) for text in OOD_CASES]
    if any(not math.isfinite(value) or not 0 <= value <= 1.000001 for value in values):
        raise ValueError("Non-finite or invalid OOD confidence")
    return {"cases": len(values), "mean_max_probability": sum(values) / len(values),
            "max_probability": max(values),
            "at_least_0_5": sum(value >= 0.5 for value in values),
            "at_least_0_8": sum(value >= 0.8 for value in values),
            "limitation": "No unknown class; action confidence is not an execution decision"}


def promotion_gate(before, after, regressions, fused_before, fused_after,
                   fused_regressions, ood_before, ood_after):
    reasons = []
    if not before["cases"]:
        reasons.append("empty holdout")
    if any(after["hits"][k] < before["hits"][k] for k in before["hits"]):
        reasons.append("standalone aggregate regression")
    if not any(after["hits"][k] > before["hits"][k] for k in before["hits"]):
        reasons.append("no standalone improvement")
    if regressions:
        reasons.append("standalone top1 regression")
    if any(fused_after["hits"][k] < fused_before["hits"][k] for k in fused_before["hits"]):
        reasons.append("fused aggregate regression")
    if fused_regressions:
        reasons.append("fused top1 regression")
    for threshold in ("at_least_0_5", "at_least_0_8"):
        if ood_after[threshold] > ood_before[threshold]:
            reasons.append("ood confidence regression: " + threshold)
    return not reasons, reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0,
                        help="Reproducible initialization and training seed (default: 0)")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--public-only", action="store_true",
                        help="Use built-in schemas only; exclude personal history and installed plugins")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20:
        parser.error("--epochs must be between 1 and 20")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    if args.public_only and args.apply:
        parser.error("--public-only cannot be combined with --apply")
    from orion_core.atomic_io import atomic_write_bytes, atomic_write_text
    from orion_core.constants import CONFIG_DIR
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.intent_brain import STATE_FILE, IntentBrain, _tool_texts
    from orion_core.tool_resolver import vocabulary_for

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    active = None if args.public_only else CONFIG_DIR / "brain" / STATE_FILE
    baseline_dir, candidate_dir = output / "baseline", output / "candidate"
    if baseline_dir.exists() or candidate_dir.exists():
        parser.error("Use a fresh output directory; existing candidates are immutable audit evidence")
    for directory in (baseline_dir, candidate_dir):
        directory.mkdir()
        if active is not None and active.exists():
            shutil.copy2(active, directory / STATE_FILE)
    original_digest = (hashlib.sha256(active.read_bytes()).hexdigest()
                       if active is not None and active.exists() else None)
    declarations = list(TOOL_DECLARATIONS)
    if not args.public_only:
        from orion_core.plugin_manifest import discover_manifests
        manifests, problems = discover_manifests(CONFIG_DIR / "custom_tools")
        if problems:
            raise RuntimeError(f"Fix invalid plugin manifests before training: {', '.join(problems)}")
        for manifest in manifests:
            if manifest.name not in {d["name"] for d in declarations}:
                declarations.append({"name": manifest.name, "description": manifest.description})
    vocabulary, descriptions = _tool_texts()
    vocabulary = {d["name"]: vocabulary.get(d["name"], vocabulary_for(d["name"])) for d in declarations}
    descriptions = {d["name"]: d.get("description", "") for d in declarations}
    baseline, candidate = IntentBrain(baseline_dir, seed=args.seed), IntentBrain(candidate_dir, seed=args.seed)
    if not baseline.seeded:
        baseline.seed(vocabulary, descriptions)
        if not baseline.save():
            raise RuntimeError("Could not save the baseline")
        shutil.copy2(baseline.path, candidate.path)
        candidate = IntentBrain(candidate_dir, seed=args.seed)
    holdout_path = Path(__file__).resolve().parents[1] / "tests/data/intent_routing_cases.py"
    spec = importlib.util.spec_from_file_location("routing_holdout", holdout_path)
    corpus = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corpus)
    cases = corpus.holdout_cases()
    before, old_first = evaluate(baseline, cases)
    scorer = lambda model, text: cold_routing_scores(model, text, declarations)
    fused_before, old_fused_first = evaluate(baseline, cases, scorer)
    ood_before = evaluate_ood(baseline)
    began = time.monotonic()
    rows = candidate.seed(vocabulary, descriptions, epochs=args.epochs)
    if not candidate.save():
        raise RuntimeError("Could not save the candidate")
    after, new_first = evaluate(candidate, cases)
    fused_after, new_fused_first = evaluate(candidate, cases, scorer)
    ood_after = evaluate_ood(candidate)
    regressions = sum(old and not new for old, new in zip(old_first, new_first))
    fused_regressions = sum(old and not new for old, new in zip(old_fused_first, new_fused_first))
    eligible, gate_reasons = promotion_gate(before, after, regressions, fused_before, fused_after,
                                          fused_regressions, ood_before, ood_after)
    promoted = False
    if args.apply and eligible:
        current_digest = hashlib.sha256(active.read_bytes()).hexdigest() if active.exists() else None
        if current_digest != original_digest:
            raise RuntimeError("Active weights changed during training; stop ORION and rerun")
        active.parent.mkdir(parents=True, exist_ok=True)
        if active.exists():
            shutil.copy2(active, output / "previous-active.npz")
        atomic_write_bytes(active, candidate.path.read_bytes())
        promoted = True
    report = {"public_only": args.public_only,
              "training_source": ("native schemas and public vocabulary only" if args.public_only else
                                  "native and installed plugin schemas plus existing successful user history"),
              "evaluation_source": "fixed odd-index routing holdout; excluded from training",
              "training_rows": rows, "epochs": args.epochs, "seed": args.seed,
              "seconds": round(time.monotonic() - began, 2),
              "actions_before": len(baseline.labels), "actions_after": len(candidate.labels),
              "before": before, "after": after, "top1_regressions": regressions,
              "cold_routing": {"before": fused_before, "after": fused_after,
                               "top1_regressions": fused_regressions,
                               "scope": "lexical + neural, no semantic or spelling correction"},
              "ood": {"before": ood_before, "after": ood_after},
              "eligible": eligible, "promoted": promoted,
              "promotion_gate_reasons": gate_reasons,
              "baseline_sha256": hashlib.sha256(baseline.path.read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(candidate.path.read_bytes()).hexdigest(),
              "holdout_sha256": hashlib.sha256(holdout_path.read_bytes()).hexdigest(),
              "ood_sha256": hashlib.sha256(json.dumps(OOD_CASES).encode()).hexdigest(),
              "scope": "Local language-to-action classifier; no claim of general intelligence"}
    atomic_write_text(output / "training.json", json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
