"""Provider-compatible declarations for ORION's local planning instruments."""

from __future__ import annotations


def declarations() -> list[dict]:
    return [
        {
            "name": "project_rehearsal",
            "description": (
                "Rehearse a project's deadline before committing: critical path, slack, "
                "Monte Carlo completion ranges and what-if task delays or speedups. "
                "Use user-supplied or explicitly agreed duration estimates; ask for missing "
                "estimates instead of inventing them. All durations and deadline share one "
                "unit. Assumes independent task durations and unlimited parallel workers; "
                "results are modelled scenarios, not promises. Local, offline, saves nothing."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "tasks": {
                        "type": "ARRAY",
                        "description": "1 to 60 tasks; dependencies are task ids, with no cycles.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "STRING", "description": "Unique task id."},
                                "name": {"type": "STRING", "description": "Readable task label."},
                                "optimistic": {"type": "NUMBER", "description": "Shortest duration, 0 to 1000000."},
                                "likely": {"type": "NUMBER", "description": "Most likely duration, between optimistic and pessimistic."},
                                "pessimistic": {"type": "NUMBER", "description": "Longest duration, at most 1000000."},
                                "depends_on": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Tasks that must finish before this starts."},
                            },
                            "required": ["id", "optimistic", "likely", "pessimistic"],
                        },
                    },
                    "unit": {"type": "STRING", "description": "Common duration unit, e.g. hours or days; default days."},
                    "deadline": {"type": "NUMBER", "description": "Optional elapsed duration available from project start, not a calendar date."},
                    "trials": {"type": "INTEGER", "description": "Simulation trials, 1 to 5000; default 2000."},
                    "seed": {"type": "INTEGER", "description": "Random seed for reproducible results; default 0."},
                    "what_if": {
                        "type": "ARRAY",
                        "description": "Optional task duration multipliers to compare against the same baseline samples.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "STRING", "description": "Existing task id."},
                                "multiplier": {"type": "NUMBER", "description": "Positive factor, at most 1000; 2 doubles duration, 0.5 halves it."},
                            },
                            "required": ["id", "multiplier"],
                        },
                    },
                },
                "required": ["tasks"],
            },
        },
        {
            "name": "context_capsule",
            "description": (
                "Save or recall a named project handoff: intent, context, exact next action, "
                "blockers, assumptions and references. Use for 'park this work', 'save a "
                "handoff' or 'where should I pick this project up'. User-curated local "
                "checkpoints; references are stored as text and never opened. Save without "
                "id creates a new checkpoint; only an explicit id updates one. List first "
                "if the id is unknown. Archive hides a capsule but preserves direct recall."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "action": {"type": "STRING", "enum": ["save", "list", "show", "archive"], "description": "Default list."},
                    "id": {"type": "STRING", "description": "Exact checkpoint id from list/save, needed for show/archive or a deliberate update."},
                    "title": {"type": "STRING", "description": "Required for a new checkpoint, at most 160 characters."},
                    "intent": {"type": "STRING", "description": "What you are trying to accomplish; up to 2000 characters."},
                    "context": {"type": "STRING", "description": "Current state and useful findings; up to 12000 characters."},
                    "next_action": {"type": "STRING", "description": "Concrete next step, required for a new checkpoint; up to 2000 characters."},
                    "blockers": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Up to 16 blockers, each at most 1000 characters."},
                    "assumptions": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Up to 16 assumptions to revisit, each at most 1000 characters."},
                    "references": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Up to 24 user-specified references; stored text only, at most 2048 characters each."},
                    "query": {"type": "STRING", "description": "Search saved capsules, up to 200 characters."},
                    "limit": {"type": "INTEGER", "description": "List size, 1 to 50; default 10."},
                    "include_archived": {"type": "BOOLEAN", "description": "Include archived capsules in list; default false."},
                },
            },
        },
        {
            "name": "experiment_notebook",
            "description": (
                "Test whether a change helps: preregister a hypothesis and measurement, "
                "log baseline and trial observations, then compare the measured results. "
                "Useful for study methods, focus routines and project experiments. Create "
                "the design before observations; it stays immutable. Never invent data. "
                "Analysis reports sample sizes and changes, not causal proof or statistical "
                "significance. Local offline notebook; archive keeps records readable."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "action": {"type": "STRING", "enum": ["create", "list", "show", "record", "analyse", "archive"], "description": "Default list."},
                    "id": {"type": "STRING", "description": "Exact experiment id from create/list, required except create/list."},
                    "title": {"type": "STRING", "description": "Experiment name, required for create."},
                    "hypothesis": {"type": "STRING", "description": "Prediction to test, required for create."},
                    "metric": {"type": "STRING", "description": "What is measured consistently, required for create."},
                    "unit": {"type": "STRING", "description": "Measurement unit, e.g. minutes or correct answers."},
                    "direction": {"type": "STRING", "enum": ["higher", "lower"], "description": "Which direction is better, required for create."},
                    "target_improvement": {"type": "NUMBER", "description": "Optional positive target improvement in absolute metric units, not a percentage."},
                    "phase": {"type": "STRING", "enum": ["baseline", "trial"], "description": "Observation phase, required for record."},
                    "value": {"type": "NUMBER", "description": "Actual finite measured value, required for record."},
                    "note": {"type": "STRING", "description": "Optional observation conditions or caveats."},
                    "limit": {"type": "INTEGER", "description": "List size, 1 to 50; default 20."},
                    "include_archived": {"type": "BOOLEAN", "description": "Include archived experiments in list; default false."},
                },
            },
        },
    ]
