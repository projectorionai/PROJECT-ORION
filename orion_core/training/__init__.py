"""
Teaching a local model to be ORION - local brain, stage 3.

Stage 1 put a local model in front of ORION's tools (``local_agent.py``).
Stage 2 measured it. This package is the data half of stage 3: it turns what
ORION already knows about how the owner talks to him - plus the routing and
safety knowledge the project already encodes - into a chat-format
tool-calling dataset that a QLoRA run can learn from.

What lives here, and what does not
----------------------------------
* ``redact``    - strips emails, phone numbers, keys, tokens, card numbers and
                  profile paths from every string before it is written.
* ``toolcalls`` - the message shapes (the same system prompt, tool schemas and
                  tool-result envelope ``LocalAgent`` uses at run time) and the
                  validation that every training call is a call the live agent
                  would actually accept.
* ``argsynth``  - plausible arguments for a spoken request, from rules, for the
                  examples where the real arguments were never recorded.
* ``templates`` - hand-written, British-English outcomes, safety refusals,
                  follow-ups and chat.
* ``sources``   - read-only miners for ORION's own stores. Nothing is opened
                  for writing.
* ``dataset``   - builds, de-duplicates, balances, splits and writes it all.

Nothing here imports a machine-learning library, downloads anything or talks to
the network. Training itself is ``tools/train_lora.py``, run by the owner, in
the neural-training environment, after he has approved the downloads.

Privacy: a dataset is private data. It is written under the training directory
(``paths.training_root``) - never inside the project folder and never inside a
cloud-synced folder - and the files in it are named in a manifest by hash, not
by content.
"""

from __future__ import annotations

#: Written into every manifest; bumped when the row format or the rules change.
DATASET_FORMAT = "orion-local-brain-dataset-v1"
BUILDER_VERSION = "dataset-builder-1"

__all__ = ["BUILDER_VERSION", "DATASET_FORMAT"]
