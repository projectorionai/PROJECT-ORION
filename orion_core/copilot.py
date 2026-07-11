"""
Developer Copilot (Phase 7) — whole-repository understanding.

Four co-operating pieces:

    ProjectIndexer   — walks a repository, classifies files by language, and
                       extracts symbols (Python via AST; other languages via
                       language-aware regex): modules, classes, functions,
                       line counts and TODO markers.

    DependencyMapper — parses imports (Python AST + JS/TS regex) into an
                       internal module dependency graph, separating internal
                       edges from external packages.

    SystemGraph      — the resulting directed graph: who-imports-what, reverse
                       dependencies (impact analysis) and cycle detection.

    CodebaseMemory   — persists a compact index digest into the MemoryAgent's
                       PROJECT tier, so ORION *remembers* repositories across
                       sessions and can resume with architectural context.

DeveloperCopilot is the façade the dispatcher calls.  Deep analysis tasks
(refactor, architecture review, bug hunt, test generation, documentation)
assemble indexed context and route it through the ProviderRouter with an
engineering persona; when no text provider is configured the assembled
context is returned so the live model can act on it directly.
"""

from __future__ import annotations

import ast
import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .memory import MemoryAgent, MemoryTier
from .utils import first_line

CODE_LANGS = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".jsx": "JavaScript", ".cs": "C#", ".rs": "Rust", ".go": "Go", ".java": "Java",
    ".rb": "Ruby", ".php": "PHP", ".c": "C", ".cpp": "C++", ".h": "C/C++ header",
    ".html": "HTML", ".css": "CSS", ".sql": "SQL",
}
IGNORED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", "target", "bin",
    "obj", ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode",
    "legacy",
}


@dataclass
class ModuleInfo:
    path: str                       # repo-relative
    language: str
    lines: int
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)      # raw import targets
    internal_deps: list[str] = field(default_factory=list)  # resolved repo modules
    todos: int = 0


@dataclass
class RepoIndex:
    root: str
    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    total_lines: int = 0
    total_todos: int = 0
    key_files: list[str] = field(default_factory=list)

    def summary(self) -> str:
        langs = ", ".join(
            f"{lang}: {count}" for lang, count in
            sorted(self.languages.items(), key=lambda kv: kv[1], reverse=True)
        )
        return (
            f"{len(self.modules)} module(s), ~{self.total_lines} lines. "
            f"Languages: {langs}. TODO/FIXME: {self.total_todos}. "
            f"Key files: {', '.join(self.key_files[:8]) or 'none'}."
        )


# ──────────────────────────────────────────────────────────────────────────────
# INDEXER
# ──────────────────────────────────────────────────────────────────────────────

class ProjectIndexer:
    KEY_NAMES = {
        "readme.md", "pyproject.toml", "package.json", "cargo.toml", "go.mod",
        "requirements.txt", "setup.py", "tsconfig.json", "dockerfile", "makefile",
    }
    MAX_FILES = 4000
    MAX_FILE_BYTES = 800_000

    def index(self, root: Path) -> RepoIndex:
        root = root.resolve()
        index = RepoIndex(root=str(root))
        if not root.is_dir():
            return index
        scanned = 0
        for path in root.rglob("*"):
            if scanned >= self.MAX_FILES:
                break
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            scanned += 1
            if path.name.lower() in self.KEY_NAMES or path.suffix.lower() == ".csproj":
                index.key_files.append(str(path.relative_to(root)))
            lang = CODE_LANGS.get(path.suffix.lower())
            if lang is None:
                continue
            try:
                if path.stat().st_size > self.MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            rel = str(path.relative_to(root)).replace("\\", "/")
            info = self._analyse(rel, lang, text)
            index.modules[rel] = info
            index.languages[lang] = index.languages.get(lang, 0) + 1
            index.total_lines += info.lines
            index.total_todos += info.todos
        return index

    def _analyse(self, rel: str, lang: str, text: str) -> ModuleInfo:
        info = ModuleInfo(path=rel, language=lang, lines=text.count("\n") + 1)
        info.todos = len(re.findall(r"(?i)\b(?:todo|fixme|hack|xxx)\b", text))
        if lang == "Python":
            self._analyse_python(info, text)
        else:
            self._analyse_generic(info, lang, text)
        return info

    def _analyse_python(self, info: ModuleInfo, text: str) -> None:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info.classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only module- and class-level defs, not nested closures.
                info.functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    info.imports.append(node.module)

    def _analyse_generic(self, info: ModuleInfo, lang: str, text: str) -> None:
        for m in re.finditer(r"(?m)^\s*(?:export\s+)?class\s+([A-Za-z_]\w*)", text):
            info.classes.append(m.group(1))
        for m in re.finditer(
            r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)", text
        ):
            info.functions.append(m.group(1))
        for m in re.finditer(r"""(?m)^\s*import\s+.*?from\s+['"]([^'"]+)['"]""", text):
            info.imports.append(m.group(1))
        for m in re.finditer(r"""require\(\s*['"]([^'"]+)['"]\s*\)""", text):
            info.imports.append(m.group(1))


# ──────────────────────────────────────────────────────────────────────────────
# DEPENDENCY MAPPER + SYSTEM GRAPH
# ──────────────────────────────────────────────────────────────────────────────

class DependencyMapper:
    """Resolve raw imports to internal repo modules and external packages."""

    def map(self, index: RepoIndex) -> "SystemGraph":
        graph = SystemGraph()
        # Build a lookup from dotted/py-path module names to repo-relative files.
        py_lookup: dict[str, str] = {}
        for rel in index.modules:
            if rel.endswith(".py"):
                dotted = rel[:-3].replace("/", ".")
                py_lookup[dotted] = rel
                py_lookup[dotted.rsplit(".", 1)[-1]] = rel  # bare module name
        for rel, info in index.modules.items():
            graph.add_node(rel)
            for target in info.imports:
                resolved = self._resolve(target, py_lookup)
                if resolved and resolved != rel:
                    info.internal_deps.append(resolved)
                    graph.add_edge(rel, resolved)
                else:
                    graph.add_external(rel, target.split(".")[0])
        return graph

    def _resolve(self, target: str, py_lookup: dict[str, str]) -> Optional[str]:
        if target in py_lookup:
            return py_lookup[target]
        # Try progressively shorter prefixes (package.__init__ style).
        parts = target.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in py_lookup:
                return py_lookup[candidate]
        return None


class SystemGraph:
    """Directed module dependency graph with impact + cycle queries."""

    def __init__(self) -> None:
        self.edges: dict[str, set[str]] = {}          # module → internal deps
        self.reverse: dict[str, set[str]] = {}        # module → dependents
        self.external: dict[str, set[str]] = {}       # module → external pkgs

    def add_node(self, node: str) -> None:
        self.edges.setdefault(node, set())
        self.reverse.setdefault(node, set())

    def add_edge(self, src: str, dst: str) -> None:
        self.add_node(src)
        self.add_node(dst)
        self.edges[src].add(dst)
        self.reverse[dst].add(src)

    def add_external(self, src: str, pkg: str) -> None:
        self.external.setdefault(src, set()).add(pkg)

    def dependencies_of(self, module: str) -> list[str]:
        return sorted(self.edges.get(module, set()))

    def dependents_of(self, module: str) -> list[str]:
        """Reverse deps — the blast radius if *module* changes."""
        return sorted(self.reverse.get(module, set()))

    def impact(self, module: str) -> set[str]:
        """Transitive set of modules affected by a change to *module*."""
        seen: set[str] = set()
        stack = [module]
        while stack:
            current = stack.pop()
            for dependent in self.reverse.get(current, set()):
                if dependent not in seen:
                    seen.add(dependent)
                    stack.append(dependent)
        return seen

    def external_packages(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pkgs in self.external.values():
            for pkg in pkgs:
                counts[pkg] = counts.get(pkg, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def find_cycles(self, limit: int = 20) -> list[list[str]]:
        cycles: list[list[str]] = []
        colour: dict[str, int] = {}  # 0=white,1=grey,2=black
        stack: list[str] = []

        def _dfs(node: str) -> None:
            if len(cycles) >= limit:
                return
            colour[node] = 1
            stack.append(node)
            for nxt in self.edges.get(node, set()):
                c = colour.get(nxt, 0)
                if c == 0:
                    _dfs(nxt)
                elif c == 1 and nxt in stack:
                    idx = stack.index(nxt)
                    cycles.append(stack[idx:] + [nxt])
            stack.pop()
            colour[node] = 2

        for node in list(self.edges):
            if colour.get(node, 0) == 0:
                _dfs(node)
        return cycles

    def hotspots(self, top: int = 8) -> list[tuple[str, int]]:
        """Modules with the most dependents — architectural load-bearing files."""
        ranked = sorted(
            ((m, len(deps)) for m, deps in self.reverse.items()),
            key=lambda kv: kv[1], reverse=True,
        )
        return [(m, n) for m, n in ranked if n > 0][:top]


# ──────────────────────────────────────────────────────────────────────────────
# CODEBASE MEMORY
# ──────────────────────────────────────────────────────────────────────────────

class CodebaseMemory:
    """Persist repo digests into the PROJECT memory tier for cross-session recall."""

    def __init__(self, memory: MemoryAgent) -> None:
        self.memory = memory

    def remember(self, index: RepoIndex, graph: SystemGraph) -> None:
        project = self.memory._project_slug(Path(index.root).name)
        digest = index.summary()
        hotspots = ", ".join(f"{Path(m).name}({n})" for m, n in graph.hotspots(5))
        self.memory.remember(MemoryTier.PROJECT, "codebase_digest", digest, project=project)
        if hotspots:
            self.memory.remember(MemoryTier.PROJECT, "codebase_hotspots", hotspots, project=project)

    def recall(self, project: str) -> list[dict[str, str]]:
        return self.memory.recall(MemoryTier.PROJECT, project=project, limit=20)


# ──────────────────────────────────────────────────────────────────────────────
# COPILOT FAÇADE
# ──────────────────────────────────────────────────────────────────────────────

class DeveloperCopilot:
    """Whole-repo engineering partner over the indexer/graph/memory + provider."""

    def __init__(self, bus: OrionBus, memory: MemoryAgent, router: Any,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.router = router
        self.telemetry = telemetry
        self.indexer = ProjectIndexer()
        self.mapper = DependencyMapper()
        self.codebase_memory = CodebaseMemory(memory)
        self._cache: dict[str, tuple[RepoIndex, SystemGraph]] = {}

    async def _index(self, root: Path) -> tuple[RepoIndex, SystemGraph]:
        key = str(root.resolve())
        if key in self._cache:
            return self._cache[key]
        index = await asyncio.to_thread(self.indexer.index, root)
        graph = await asyncio.to_thread(self.mapper.map, index)
        self._cache[key] = (index, graph)
        return index, graph

    def _resolve_root(self, path: str) -> Path:
        p = Path(path).expanduser() if path.strip() else BASE_DIR
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    # ── analysis ──────────────────────────────────────────────────────────────

    async def analyse_repository(self, path: str = "") -> ToolResult:
        root = self._resolve_root(path)
        if not root.is_dir():
            return ToolResult(f"Repository not found: {root}", ok=False)
        index, graph = await self._index(root)
        if not index.modules:
            return ToolResult(f"No recognised source files under {root}.")
        await asyncio.to_thread(self.codebase_memory.remember, index, graph)
        cycles = graph.find_cycles(limit=5)
        hotspots = graph.hotspots(6)
        ext = list(graph.external_packages().items())[:10]
        lines = [
            f"Repository analysis: {root}",
            index.summary(),
            "Architectural hotspots (most depended-on): "
            + (", ".join(f"{m} <-{n}" for m, n in hotspots) or "none"),
            "Top external packages: " + (", ".join(f"{p}×{n}" for p, n in ext) or "none"),
        ]
        if cycles:
            lines.append("Import cycles detected: " + "; ".join(" → ".join(c) for c in cycles[:3]))
        else:
            lines.append("No import cycles detected.")
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("copilot.modules", float(len(index.modules)))
        return ToolResult("\n".join(lines))

    async def find_symbol(self, name: str, path: str = "") -> ToolResult:
        index, _ = await self._index(self._resolve_root(path))
        name_l = name.strip().lower()
        hits: list[str] = []
        for rel, info in index.modules.items():
            for cls in info.classes:
                if name_l in cls.lower():
                    hits.append(f"class {cls}  —  {rel}")
            for fn in info.functions:
                if name_l in fn.lower():
                    hits.append(f"def {fn}  —  {rel}")
        if not hits:
            return ToolResult(f"No class or function matching '{name}'.")
        return ToolResult(f"Symbols matching '{name}':\n" + "\n".join(hits[:40]))

    async def dependency_report(self, module: str, path: str = "") -> ToolResult:
        index, graph = await self._index(self._resolve_root(path))
        match = self._match_module(module, index)
        if match is None:
            return ToolResult(f"No module matching '{module}' in the index.", ok=False)
        deps = graph.dependencies_of(match)
        dependents = graph.dependents_of(match)
        impact = graph.impact(match)
        return ToolResult(
            f"Dependency report for {match}:\n"
            f"Imports ({len(deps)}): {', '.join(deps) or 'none internal'}\n"
            f"Imported by ({len(dependents)}): {', '.join(dependents) or 'none'}\n"
            f"Change impact (transitive dependents): {len(impact)} module(s)"
            + (": " + ", ".join(sorted(impact)[:15]) if impact else "")
        )

    def _match_module(self, module: str, index: RepoIndex) -> Optional[str]:
        module_l = module.strip().lower().replace("\\", "/")
        if module_l in index.modules:
            return module_l
        for rel in index.modules:
            if module_l in rel.lower() or Path(rel).stem.lower() == module_l:
                return rel
        return None

    # ── LLM-backed engineering tasks (context-grounded) ───────────────────────

    async def engineering_task(self, task: str, path: str = "", focus: str = "") -> ToolResult:
        """
        Refactor / architecture-review / bug-hunt / test-gen / doc-gen, grounded
        in the live index.  Routes through the provider with an engineering
        persona; if none is configured, returns the assembled context so the
        live model can act.
        """
        index, graph = await self._index(self._resolve_root(path))
        context = self._context_brief(index, graph, focus)
        persona = (
            "SPECIALIST MODE — Principal Engineer with whole-repository context. "
            "Use the provided repository index (modules, symbols, dependency graph, "
            "hotspots) to answer precisely. Reference real file paths. For refactors "
            "propose concrete, minimal diffs; for reviews cite specific modules; for "
            "test generation target the highest-impact hotspots; never invent files "
            "that are not in the index."
        )
        prompt = f"Task: {task}\n\n{context}"
        if focus.strip():
            prompt += f"\n\nFocus: {focus.strip()}"
        if not self.router.has_text_fallback():
            return ToolResult(
                "[Engineering context — no text provider configured; the live model "
                f"should act on this directly]\n{context}"
            )
        try:
            profile, answer = await self.router.generate_text(prompt, system_extra=persona)
            return ToolResult(f"[Developer Copilot via {profile.name}]\n{answer}")
        except Exception as exc:
            return ToolResult(f"Copilot could not reach a provider: {first_line(exc)}", ok=False)

    def _context_brief(self, index: RepoIndex, graph: SystemGraph, focus: str) -> str:
        lines = [f"REPOSITORY INDEX ({index.root})", index.summary(), "", "MODULES:"]
        focus_l = focus.strip().lower()
        shown = 0
        for rel, info in index.modules.items():
            if focus_l and focus_l not in rel.lower() and shown > 40:
                continue
            if shown >= 60:
                break
            symbols = ", ".join((info.classes + info.functions)[:8])
            deps = ", ".join(Path(d).stem for d in info.internal_deps[:6])
            lines.append(f"- {rel} [{info.language}, {info.lines} ln]"
                         + (f" symbols: {symbols}" if symbols else "")
                         + (f" → {deps}" if deps else ""))
            shown += 1
        hotspots = graph.hotspots(6)
        if hotspots:
            lines.append("\nHOTSPOTS: " + ", ".join(f"{m}<-{n}" for m, n in hotspots))
        return "\n".join(lines)
