"""
CodeChangelog — ORION reads his own source and says what actually changed.

The problem
-----------
``changelog.py`` is a hand-curated list of ``Release`` objects. It is prose
somebody wrote once, so it can only ever describe changes somebody remembered
to write down — and by definition it knows nothing about what ORION did to
himself last night. The other half of the picture, ``change_awareness.py``,
watches files and can tell you *that* ``audio.py`` changed. Neither can tell
you *what* changed, and "audio.py was modified" is not an answer to "what did
you just do?".

The approach
------------
ORION indexes his own code STRUCTURALLY — every module, class, function, its
signature, its docstring, and a hash of its body — and keeps the last index on
disk. Asked what changed, he re-indexes and compares:

    added        a function/class that did not exist before
    removed      one that no longer exists
    resigned     same name, different parameters (a contract change)
    rewritten    same signature, different body (a behaviour change)
    redocumented only the docstring moved

Then he explains each one **in the words of the code itself** — the docstring
is the author's own statement of intent, so quoting it is both the most
accurate available description and free of invention. Where a docstring says
*why*, ORION says why.

Why AST and not a text diff
---------------------------
A textual diff of this codebase is mostly noise: reflowed comments, moved
imports, renamed locals. Parsing means a change is reported when the CODE
changed — a reordered function is not a change, a new parameter is, and
"rewritten" genuinely means the body does something different.

Why not git
-----------
Git would be the obvious source, but this working tree's history is a single
sterilised commit and everything since is uncommitted, so ``git diff HEAD``
reports the entire project as new — true, and useless. Indexing works from
whatever is actually on disk right now, which is the thing ORION is actually
running.

Nothing here invents. A change with no docstring is reported as a change with
no stated purpose, not given an imagined one.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

INDEX_PATH = CONFIG_DIR / "code_index.json"

#: Directories that are not ORION's own thinking and would only add noise.
SKIP_PARTS = {"__pycache__", ".venv", "node_modules", "_quarantine",
              ".git", "build", "dist"}


def _first_sentence(text: str | None, limit: int = 220) -> str:
    """The first real sentence of a docstring, flattened to one line."""
    if not text:
        return ""
    flat = " ".join(str(text).split())
    for stop in (". ", " — ", "; "):
        index = flat.find(stop)
        if 0 < index <= limit:
            return flat[:index].strip()
    # Strip a trailing full stop: callers punctuate their own sentences, and
    # keeping it produced "…to the output device..".
    return flat[:limit].strip().rstrip(".").strip()


def _signature(node: ast.AST) -> str:
    """A readable parameter list, so a contract change is visible as one."""
    args = getattr(node, "args", None)
    if args is None:
        return ""
    parts: list[str] = []
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    defaults = list(args.defaults)
    pad = len(positional) - len(defaults)
    for index, arg in enumerate(positional):
        text = arg.arg
        if index >= pad:
            text += "=" + _short(defaults[index - pad])
        parts.append(text)
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if default is not None:
            text += "=" + _short(default)
        parts.append(text)
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def _short(node: ast.AST) -> str:
    try:
        rendered = ast.unparse(node)
    except Exception:
        return "…"
    return rendered if len(rendered) <= 24 else rendered[:21] + "…"


def _body_hash(node: ast.AST) -> str:
    """Fingerprint the BODY only, with the docstring stripped.

    Stripping it is what separates "rewritten" from "redocumented": improving a
    comment is not a behaviour change and should never be reported as one.
    """
    body = list(getattr(node, "body", []))
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    try:
        text = "\n".join(ast.dump(n, annotate_fields=False) for n in body)
    except Exception:
        text = repr(body)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _class_hash(node: ast.ClassDef) -> str:
    """Fingerprint a class's own shape, not its methods' bodies.

    A method body changing is reported as that method changing. Folding it into
    the class hash as well made every touched class read as "rewritten" too,
    which is noise: one edit produced two entries saying different things about
    the same change.
    """
    parts: list[str] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(f"def {item.name}{_signature(item)}")
        elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            continue                       # the docstring is tracked separately
        else:
            try:
                parts.append(ast.dump(item, annotate_fields=False))
            except Exception:
                parts.append(repr(item))
    blob = "\n".join(parts) + "|" + ",".join(
        b.id for b in node.bases if isinstance(b, ast.Name))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class Symbol:
    """One function, method or class as ORION understands it."""

    kind: str            # "function" | "method" | "class"
    name: str            # dotted within its module: "Class.method"
    signature: str
    doc: str
    body: str            # hash
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "signature": self.signature,
                "doc": self.doc, "body": self.body, "line": self.line}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Symbol":
        return Symbol(str(data.get("kind", "function")), str(data.get("name", "")),
                      str(data.get("signature", "")), str(data.get("doc", "")),
                      str(data.get("body", "")), int(data.get("line", 0)))


@dataclass
class ModuleIndex:
    path: str
    doc: str = ""
    symbols: dict[str, Symbol] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "doc": self.doc,
                "symbols": {k: v.to_dict() for k, v in self.symbols.items()}}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ModuleIndex":
        return ModuleIndex(
            str(data.get("path", "")), str(data.get("doc", "")),
            {k: Symbol.from_dict(v)
             for k, v in (data.get("symbols") or {}).items()})


@dataclass
class Change:
    """One thing that changed, and what the code says it is for."""

    kind: str            # added | removed | resigned | rewritten | redocumented
    module: str
    symbol: str
    purpose: str = ""    # from the docstring — never invented
    detail: str = ""     # e.g. the signature before and after
    line: int = 0

    #: How much a reader should care. Contract changes and removals can break
    #: callers; a new helper cannot.
    WEIGHT = {"removed": 4, "resigned": 3, "added": 2,
              "rewritten": 2, "redocumented": 1}

    @property
    def weight(self) -> int:
        return self.WEIGHT.get(self.kind, 1)

    def sentence(self) -> str:
        verb = {
            "added": "Added",
            "removed": "Removed",
            "resigned": "Changed the signature of",
            "rewritten": "Rewrote",
            "redocumented": "Re-documented",
        }.get(self.kind, "Changed")
        line = f"{verb} {self.symbol} in {self.module}"
        if self.detail:
            line += f" ({self.detail})"
        if self.purpose:
            line += f" — {self.purpose}"
        return line + "."

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "module": self.module, "symbol": self.symbol,
                "purpose": self.purpose, "detail": self.detail, "line": self.line}


class CodeIndex:
    """A structural snapshot of ORION's own source."""

    def __init__(self, modules: dict[str, ModuleIndex] | None = None,
                 built_at: float | None = None) -> None:
        self.modules: dict[str, ModuleIndex] = modules or {}
        self.built_at: float = built_at if built_at is not None else time.time()

    # ── building ──────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, root: Path | str, limit: int = 4000) -> "CodeIndex":
        """Index every Python module under *root*. Never raises on one bad file."""
        root = Path(root)
        modules: dict[str, ModuleIndex] = {}
        for path in sorted(root.rglob("*.py")):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if len(modules) >= limit:
                break
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (OSError, SyntaxError, ValueError):
                continue          # a file ORION cannot read is simply not indexed
            try:
                rel = str(path.relative_to(root.parent)).replace("\\", "/")
            except ValueError:
                rel = str(path).replace("\\", "/")
            module = ModuleIndex(rel, _first_sentence(ast.get_docstring(tree)))
            for symbol in cls._walk(tree):
                module.symbols[symbol.name] = symbol
            modules[rel] = module
        return cls(modules)

    @staticmethod
    def _walk(tree: ast.Module) -> Iterable[Symbol]:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield Symbol("function", node.name, _signature(node),
                             _first_sentence(ast.get_docstring(node)),
                             _body_hash(node), node.lineno)
            elif isinstance(node, ast.ClassDef):
                yield Symbol("class", node.name, "",
                             _first_sentence(ast.get_docstring(node)),
                             _class_hash(node), node.lineno)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield Symbol("method", f"{node.name}.{item.name}",
                                     _signature(item),
                                     _first_sentence(ast.get_docstring(item)),
                                     _body_hash(item), item.lineno)

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {"built_at": self.built_at,
                "modules": {k: v.to_dict() for k, v in self.modules.items()}}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CodeIndex":
        return CodeIndex(
            {k: ModuleIndex.from_dict(v)
             for k, v in (data.get("modules") or {}).items()},
            float(data.get("built_at") or 0.0))

    def save(self, path: Path = INDEX_PATH) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps(self.to_dict()), encoding="utf-8")
            return True
        except OSError:
            return False

    @staticmethod
    def load(path: Path = INDEX_PATH) -> "CodeIndex | None":
        try:
            return CodeIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    # ── the comparison ────────────────────────────────────────────────────────

    def diff(self, previous: "CodeIndex") -> list[Change]:
        """Every semantic difference from *previous*, most consequential first."""
        changes: list[Change] = []

        for path, module in self.modules.items():
            before = previous.modules.get(path)
            if before is None:
                changes.append(Change(
                    "added", path, "the whole module",
                    module.doc,
                    f"{len(module.symbols)} definition"
                    f"{'s' if len(module.symbols) != 1 else ''}", 1))
                continue
            for name, symbol in module.symbols.items():
                old = before.symbols.get(name)
                if old is None:
                    changes.append(Change("added", path, name, symbol.doc,
                                          symbol.signature, symbol.line))
                elif old.signature != symbol.signature:
                    changes.append(Change(
                        # ASCII arrow deliberately: this text reaches the log,
                        # a console that may be cp1252, and the voice channel.
                        # A pretty arrow is worth nothing to any of the three
                        # and raises UnicodeEncodeError on the second.
                        "resigned", path, name, symbol.doc,
                        f"{old.signature or '()'} -> {symbol.signature or '()'}",
                        symbol.line))
                elif old.body != symbol.body:
                    changes.append(Change("rewritten", path, name, symbol.doc,
                                          "", symbol.line))
                elif old.doc != symbol.doc:
                    changes.append(Change("redocumented", path, name, symbol.doc,
                                          "", symbol.line))
            for name, old in before.symbols.items():
                if name not in module.symbols:
                    changes.append(Change("removed", path, name, old.doc, "",
                                          old.line))

        for path, before in previous.modules.items():
            if path not in self.modules:
                changes.append(Change("removed", path, "the whole module",
                                      before.doc, "", 1))

        changes.sort(key=lambda c: (-c.weight, c.module, c.symbol))
        return changes


class CodeChangelog:
    """Answers "what did you change?" from the code, not from a written note."""

    #: Read aloud, more than this is a monologue. The written form is complete.
    SPOKEN_LIMIT = 6

    def __init__(self, root: Path | str, index_path: Path = INDEX_PATH) -> None:
        self.root = Path(root)
        self.index_path = index_path

    # ── snapshots ─────────────────────────────────────────────────────────────

    def snapshot(self) -> CodeIndex:
        """Record the current code as the new baseline."""
        index = CodeIndex.build(self.root)
        index.save(self.index_path)
        return index

    def changes_since_snapshot(self) -> tuple[list[Change], CodeIndex | None]:
        previous = CodeIndex.load(self.index_path)
        current = CodeIndex.build(self.root)
        if previous is None:
            return [], None
        return current.diff(previous), previous

    # ── narration ─────────────────────────────────────────────────────────────

    def narrate(self, changes: list[Change], previous: CodeIndex | None,
                spoken: bool = False) -> str:
        """Explain the changes in prose, grouped so it reads as a story.

        Every purpose quoted here is the code's own docstring. Where there is
        none, ORION says so rather than inventing a reason.
        """
        if previous is None:
            return ("I have no earlier snapshot of my own code to compare "
                    "against, so I have taken one now. Ask me again after I "
                    "have changed something and I will tell you exactly what.")
        if not changes:
            since = self._ago(previous.built_at)
            return f"Nothing in my code has changed since I last looked{since}."

        counts: dict[str, int] = {}
        for change in changes:
            counts[change.kind] = counts.get(change.kind, 0) + 1
        summary = ", ".join(
            f"{count} {self._plural(kind, count)}"
            for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]))
        modules = {c.module for c in changes}
        head = (f"Since I last looked{self._ago(previous.built_at)} I have made "
                f"{len(changes)} change{'s' if len(changes) != 1 else ''} across "
                f"{len(modules)} file{'s' if len(modules) != 1 else ''}: {summary}.")

        if spoken:
            lines = [head]
            for change in changes[:self.SPOKEN_LIMIT]:
                lines.append(self._spoken_sentence(change))
            if len(changes) > self.SPOKEN_LIMIT:
                lines.append(f"There are {len(changes) - self.SPOKEN_LIMIT} "
                             "more; the full list is on screen.")
            return " ".join(lines)

        lines = [head, ""]
        by_module: dict[str, list[Change]] = {}
        for change in changes:
            by_module.setdefault(change.module, []).append(change)
        ordered = sorted(by_module.items(),
                         key=lambda kv: (-max(c.weight for c in kv[1]), kv[0]))
        for module, group in ordered:
            lines.append(f"{module}")
            for change in group:
                lines.append(f"  • {change.sentence()}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _spoken_sentence(self, change: Change) -> str:
        """One change, phrased for the ear rather than the eye."""
        where = change.module.rsplit("/", 1)[-1].removesuffix(".py")
        verb = {
            "added": "I added", "removed": "I removed",
            "resigned": "I changed the arguments of",
            "rewritten": "I rewrote", "redocumented": "I re-documented",
        }.get(change.kind, "I changed")
        name = change.symbol.replace("_", " ").replace(".", "'s ")
        sentence = f"{verb} {name} in {where}"
        if change.purpose:
            sentence += self._join_purpose(change.purpose)
        return sentence + "."

    #: Docstrings that begin with one of these are a VERB phrase describing
    #: behaviour ("Write a PCM chunk…"), so they join with "which".
    _VERB_STARTS = (
        "write", "read", "return", "returns", "open", "close", "build",
        "make", "create", "run", "start", "stop", "cap", "caps", "trim",
        "trims", "check", "checks", "handle", "handles", "convert", "converts",
        "record", "records", "report", "reports", "decide", "decides",
        "resolve", "resolves", "answer", "answers", "explain", "explains",
        "fix", "fixes", "parse", "parses", "gate", "gates", "pause", "pauses",
        "find", "finds", "give", "gives", "keep", "keeps", "turn", "turns",
        "move", "moves", "apply", "applies", "emit", "emits", "wait", "waits",
    )

    @classmethod
    def _join_purpose(cls, purpose: str) -> str:
        """Attach a docstring to a sentence so it actually reads as English.

        Docstrings come in two shapes. A verb phrase ("Write a PCM chunk to the
        output device") reads correctly after "which". A noun phrase ("The
        output device") does not — "I rewrote Speaker, which the output device"
        is gibberish, which is exactly what the first version produced.
        """
        text = purpose.strip().rstrip(".")
        if not text:
            return ""
        head, _, tail = text.partition(" ")
        first = head.lower().rstrip(",:")
        if first in cls._VERB_STARTS:
            # Docstrings are written in the imperative ("Write a PCM chunk…").
            # Spoken inside "…, which …" that needs the third person, or ORION
            # says "which write a PCM chunk", which is not English.
            return f", which {cls._third_person(first)}" + (f" {tail}" if tail else "")
        return f" — {text}"

    @staticmethod
    def _third_person(verb: str) -> str:
        """write -> writes, catch -> catches, verify -> verifies, caps -> caps."""
        if verb.endswith("s") and not verb.endswith("ss"):
            return verb                       # already third person
        if verb.endswith(("sh", "ch", "x", "z", "ss", "o")):
            return verb + "es"
        if len(verb) > 1 and verb.endswith("y") and verb[-2] not in "aeiou":
            return verb[:-1] + "ies"
        return verb + "s"

    @staticmethod
    def _plural(kind: str, count: int) -> str:
        words = {"added": "addition", "removed": "removal",
                 "resigned": "signature change", "rewritten": "rewrite",
                 "redocumented": "documentation update"}
        word = words.get(kind, kind)
        return word if count == 1 else word + "s"

    @staticmethod
    def _ago(when: float) -> str:
        if not when:
            return ""
        seconds = max(0.0, time.time() - when)
        if seconds < 90:
            return " a moment ago"
        if seconds < 3600:
            return f" {int(seconds // 60)} minutes ago"
        if seconds < 86400:
            hours = int(seconds // 3600)
            return f" {hours} hour{'s' if hours != 1 else ''} ago"
        days = int(seconds // 86400)
        return f" {days} day{'s' if days != 1 else ''} ago"

    # ── the question ──────────────────────────────────────────────────────────

    def report(self, spoken: bool = False, update: bool = True) -> str:
        """What has changed in my own code since I last looked?"""
        changes, previous = self.changes_since_snapshot()
        text = self.narrate(changes, previous, spoken=spoken)
        if update:
            # Take a fresh baseline so the next question means "since now"
            # rather than repeating the same answer for ever.
            self.snapshot()
        return text

    def explain(self, name: str) -> str:
        """What is this function/class for, in its own words?

        The other half of "look in your own code and tell me": not what
        changed, but what a given piece of ORION actually does.
        """
        needle = str(name or "").strip().lower()
        if not needle:
            return "Which part of me would you like me to explain?"
        index = CodeIndex.build(self.root)
        hits: list[tuple[str, Symbol]] = []
        for path, module in index.modules.items():
            for symbol in module.symbols.values():
                if needle in symbol.name.lower():
                    hits.append((path, symbol))
        if not hits:
            return f"I have nothing called '{name}' in my own code."
        hits.sort(key=lambda pair: len(pair[1].name))
        lines = []
        for path, symbol in hits[:6]:
            head = f"{symbol.name}{symbol.signature} — {symbol.kind} in {path}"
            lines.append(head)
            lines.append(f"    {symbol.doc}" if symbol.doc
                         else "    (no docstring: this one does not say what it is for)")
        if len(hits) > 6:
            lines.append(f"…and {len(hits) - 6} more matches.")
        return "\n".join(lines)


__all__ = ["Change", "CodeChangelog", "CodeIndex", "ModuleIndex", "Symbol",
           "INDEX_PATH"]
