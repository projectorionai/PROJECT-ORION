"""
Cybersecurity knowledge & training subsystem (Section 11).

A versioned, structured curriculum (modules → topics/labs/projects) with search
and retrieval, per-user progress tracking, safe lab-boundary metadata, and a
default-deny permission gate for any security-related system/network action.

This subsystem organises KNOWLEDGE and guides SAFE practice — it does not grant
uncontrolled offensive capability.  Offensive-security modules are marked as
requiring explicit authorisation and isolation, carry legal reminders, and any
action against a target is gated by :func:`validate_target_scope`, which
default-denies ambiguous or external targets.

Adding curriculum content: append a ``Module`` to ``CURRICULUM_MODULES`` with
its topics/labs/projects and the correct :class:`SafetyClass`.  Bump
``CURRICULUM_VERSION``.  Anything offensive MUST set ``isolation_required`` and
``authorization_required`` and provide a ``legal_reminder``.  Validation is via
``validate_curriculum()`` and the test-suite.

NOTE ON "TRAINING": adding curriculum files does NOT retrain or change any base
model.  This subsystem provides retrieval, structure, progress and safe-practice
guidance only; :func:`training_capability_note` states this plainly for the UI.
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from .atomic_io import atomic_write_text

try:
    from .constants import CONFIG_DIR
except Exception:  # pragma: no cover
    CONFIG_DIR = Path(".")

CURRICULUM_VERSION = "1.0.0"
PROGRESS_PATH = CONFIG_DIR / "cyber_progress.json"


class SafetyClass(str, Enum):
    """Separation of concern for every curriculum item."""

    GENERAL_EDUCATION = "general_education"
    DEFENSIVE_ANALYSIS = "defensive_analysis"
    SECURE_CODING = "secure_coding"
    AUTHORIZED_LAB = "authorized_lab"          # isolated/authorised simulation only
    POTENTIALLY_DESTRUCTIVE = "potentially_destructive"  # gated, default-deny


# Items at or above this class require an isolated, authorised environment.
_REQUIRES_ISOLATION = {SafetyClass.AUTHORIZED_LAB, SafetyClass.POTENTIALLY_DESTRUCTIVE}

_LEGAL_REMINDER = (
    "Authorisation required: only perform this against systems you own or have "
    "explicit written permission to test. Unauthorised access is illegal. Use "
    "isolated lab environments; never touch third-party or production systems.")


@dataclass
class Lab:
    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)
    safety: SafetyClass = SafetyClass.GENERAL_EDUCATION

    @property
    def isolation_required(self) -> bool:
        return self.safety in _REQUIRES_ISOLATION


@dataclass
class Project:
    name: str
    description: str = ""
    languages: list[str] = field(default_factory=list)
    safety: SafetyClass = SafetyClass.SECURE_CODING


@dataclass
class Module:
    number: int
    title: str
    topics: list[str] = field(default_factory=list)
    labs: list[Lab] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    prerequisites: list[int] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    safety: SafetyClass = SafetyClass.GENERAL_EDUCATION
    authorization_required: bool = False
    isolation_required: bool = False
    legal_reminder: str = ""

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")

    def searchable_text(self) -> str:
        parts = [self.title, *self.topics, *self.tools,
                 *(l.name for l in self.labs), *(p.name for p in self.projects)]
        return " ".join(parts).lower()

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["safety"] = self.safety.value
        for lab in d["labs"]:
            lab["safety"] = lab["safety"].value if isinstance(lab["safety"], SafetyClass) else lab["safety"]
        for proj in d["projects"]:
            proj["safety"] = proj["safety"].value if isinstance(proj["safety"], SafetyClass) else proj["safety"]
        d["slug"] = self.slug
        return d


def _offensive(number: int, title: str, **kw) -> Module:
    """Helper: build an offensive/lab module with the safety defaults enforced."""
    kw.setdefault("safety", SafetyClass.AUTHORIZED_LAB)
    return Module(number=number, title=title, authorization_required=True,
                  isolation_required=True, legal_reminder=_LEGAL_REMINDER, **kw)


# ── the curriculum (all 20 modules) ──────────────────────────────────────────

CURRICULUM_MODULES: list[Module] = [
    Module(1, "Computer Science Foundations",
           topics=["CPU", "RAM", "Cache", "Registers", "Motherboards", "Storage", "Buses",
                   "Binary", "Hexadecimal", "Octal", "Two's complement", "Bitwise operations",
                   "Processes", "Threads", "Scheduling", "Memory management", "Paging",
                   "Virtual memory", "Arrays", "Linked lists", "Trees", "Graphs", "Hash tables",
                   "Queues", "Stacks", "Sorting", "Searching", "Dynamic programming",
                   "Graph algorithms"],
           labs=[Lab("Binary calculator"), Lab("Hex converter"), Lab("Mini shell"),
                 Lab("Process monitor")],
           projects=[Project("Simple CPU simulator", languages=["Python", "C++"]),
                     Project("Memory-allocation visualizer", languages=["Python"])],
           tools=["LeetCode", "HackerRank"],
           objectives=["Understand architecture, binary, OS internals, data structures and algorithms"]),

    Module(2, "Programming Mastery",
           topics=["Variables", "Functions", "Object-oriented programming", "Async programming",
                   "APIs", "Automation", "DOM", "Node.js", "Express", "Pointers", "Memory",
                   "Buffers", "Structs", "Templates", "STL", "Ownership", "Borrowing", "Lifetimes"],
           projects=[
               Project("Password manager", languages=["Python"], safety=SafetyClass.SECURE_CODING),
               Project("Authorized localhost network scanner", languages=["Python"],
                       safety=SafetyClass.AUTHORIZED_LAB),
               Project("Authorized localhost port scanner", languages=["Python"],
                       safety=SafetyClass.AUTHORIZED_LAB),
               Project("Secure login system", languages=["JavaScript"], safety=SafetyClass.SECURE_CODING),
               Project("Web dashboard", languages=["JavaScript"]),
               Project("TCP chat server", languages=["C"]),
               Project("Mini HTTP server", languages=["C"]),
               Project("Game-engine fundamentals", languages=["C++"]),
               Project("Secure network tools for authorized lab environments", languages=["Rust"],
                       safety=SafetyClass.AUTHORIZED_LAB)],
           tools=["Python", "JavaScript", "C", "C++", "Rust"],
           prerequisites=[1],
           objectives=["Master Python, JavaScript, C, C++ and Rust with secure-by-default projects"]),

    Module(3, "Linux Mastery",
           topics=["File system", "Permissions", "Services", "Shell scripting", "Automation",
                   "System calls", "Proc filesystem", "Kernel concepts", "SELinux", "AppArmor",
                   "Firewalls"],
           labs=[Lab("Ubuntu", safety=SafetyClass.GENERAL_EDUCATION),
                 Lab("Debian", safety=SafetyClass.GENERAL_EDUCATION),
                 Lab("Kali in isolated labs", safety=SafetyClass.AUTHORIZED_LAB)],
           projects=[Project("Automated backup system", languages=["Bash"])],
           prerequisites=[1],
           safety=SafetyClass.DEFENSIVE_ANALYSIS,
           isolation_required=True, legal_reminder=_LEGAL_REMINDER),

    Module(4, "Networking",
           topics=["TCP/IP", "OSI", "Routing", "NAT", "DNS records", "Resolution", "Caching",
                   "HTTP", "HTTPS", "Requests", "Responses", "TLS", "VLANs", "VPNs", "Subnetting"],
           labs=[Lab("Analyze authorized traffic captures and fixtures",
                     tools=["Wireshark", "TCPDump"], safety=SafetyClass.DEFENSIVE_ANALYSIS)],
           tools=["Wireshark", "TCPDump"],
           prerequisites=[1],
           objectives=["Understand TCP/IP, DNS, HTTP(S)/TLS, packet analysis and network design"]),

    Module(5, "Databases",
           topics=["PostgreSQL", "MySQL", "MongoDB", "Redis",
                   "SQL injection prevention and isolated demonstrations", "Access controls",
                   "Encryption"],
           labs=[Lab("CRM database", safety=SafetyClass.GENERAL_EDUCATION)],
           prerequisites=[2],
           safety=SafetyClass.SECURE_CODING,
           objectives=["SQL/NoSQL fundamentals and database security (injection prevention)"]),

    Module(6, "Web Development",
           topics=["HTML", "CSS", "JavaScript", "APIs", "Authentication", "Sessions", "JWT",
                   "OAuth 2.0", "OpenID Connect", "Secure coding", "Validation", "Logging"],
           prerequisites=[2],
           safety=SafetyClass.SECURE_CODING,
           objectives=["Build secure frontend/backend with modern auth and safe defaults"]),

    Module(7, "Secure Software Engineering",
           topics=["Monoliths", "Microservices", "Factory", "Observer", "Singleton",
                   "STRIDE", "DREAD"],
           tools=["GitHub Actions", "Jenkins", "Docker", "Kubernetes"],
           prerequisites=[6],
           safety=SafetyClass.SECURE_CODING,
           objectives=["Architecture, design patterns, CI/CD, DevOps and threat modelling"]),

    Module(8, "Cybersecurity Fundamentals",
           topics=["CIA triad", "Zero Trust", "Defense in depth", "NIST", "ISO 27001",
                   "Symmetric encryption", "AES", "RSA", "ECC", "Hashing"],
           labs=[Lab("Encrypted messaging system using maintained cryptographic libraries",
                     safety=SafetyClass.SECURE_CODING)],
           prerequisites=[4],
           safety=SafetyClass.DEFENSIVE_ANALYSIS,
           objectives=["Security principles, risk frameworks and applied cryptography"]),

    _offensive(9, "Web Application Security Testing",
               topics=["Broken access control", "Injection", "Authentication flaws", "SSRF",
                       "XSS", "CSRF"],
               labs=[Lab("Intercept requests in an authorized lab", tools=["Burp Suite"],
                         safety=SafetyClass.AUTHORIZED_LAB),
                     Lab("Modify responses in an authorized lab", tools=["Burp Suite"],
                         safety=SafetyClass.AUTHORIZED_LAB),
                     Lab("PortSwigger Web Security Academy", safety=SafetyClass.AUTHORIZED_LAB),
                     Lab("OWASP Juice Shop", safety=SafetyClass.AUTHORIZED_LAB)],
               tools=["Burp Suite", "PortSwigger Web Security Academy", "OWASP Juice Shop"],
               prerequisites=[6, 8]),

    _offensive(10, "Authorized Network Security Testing",
               topics=["Enumeration", "Vulnerability Assessment", "Domains", "Kerberos", "LDAP",
                       "Group Policy", "PowerShell"],
               labs=[Lab("Enumeration against isolated or explicitly authorized targets only",
                         tools=["Nmap", "Masscan"], safety=SafetyClass.AUTHORIZED_LAB),
                     Lab("Vulnerability assessment in an authorized lab",
                         tools=["Nessus", "OpenVAS"], safety=SafetyClass.AUTHORIZED_LAB)],
               tools=["Nmap", "Masscan", "Nessus", "OpenVAS"],
               prerequisites=[4, 9]),

    _offensive(11, "Windows Internals",
               topics=["Registry", "LSASS architecture and defensive protection", "Tokens",
                       "Services", "Event logs"],
               labs=[Lab("Isolated Windows lab environment", safety=SafetyClass.AUTHORIZED_LAB)],
               prerequisites=[10]),

    Module(12, "Cloud Security",
           topics=["AWS IAM", "S3", "EC2", "Microsoft Entra ID/Azure AD", "Azure Networking",
                   "GCP Identity", "GCP Storage", "Misconfigurations", "Secrets management"],
           prerequisites=[7, 8],
           safety=SafetyClass.DEFENSIVE_ANALYSIS,
           objectives=["Cloud IAM, storage, networking and misconfiguration/secrets hardening"]),

    _offensive(13, "Malware Analysis",
               topics=["Static analysis", "Dynamic analysis", "Reverse engineering", "Assembly",
                       "x86", "x64"],
               labs=[Lab("Isolated disposable sandbox only — no execution on the host, no "
                         "uncontrolled networking, no propagation or persistence",
                         tools=["Ghidra", "PEStudio", "Procmon", "Wireshark"],
                         safety=SafetyClass.POTENTIALLY_DESTRUCTIVE)],
               tools=["Ghidra", "PEStudio", "Procmon", "Wireshark"],
               prerequisites=[11],
               safety=SafetyClass.POTENTIALLY_DESTRUCTIVE),

    _offensive(14, "Reverse Engineering",
               topics=["Assembly language", "Calling conventions", "Memory layout", "Debugging"],
               labs=[Lab("Isolated RE lab", tools=["Ghidra", "IDA Free", "x64dbg"],
                         safety=SafetyClass.AUTHORIZED_LAB)],
               tools=["Ghidra", "IDA Free", "x64dbg"],
               prerequisites=[13]),

    Module(15, "Digital Forensics",
           topics=["Disk analysis", "Memory analysis", "Log analysis"],
           labs=[Lab("Forensic analysis of authorized images", tools=["Volatility", "Autopsy"],
                     safety=SafetyClass.DEFENSIVE_ANALYSIS)],
           tools=["Volatility", "Autopsy"],
           prerequisites=[11],
           safety=SafetyClass.DEFENSIVE_ANALYSIS,
           objectives=["Disk, memory and log forensics with evidence-handling discipline"]),

    Module(16, "Blue Team Operations",
           topics=["SIEM", "Threat hunting", "Incident response"],
           labs=[Lab("Detection engineering in a lab SIEM", tools=["Splunk", "Elastic", "Wazuh"],
                     safety=SafetyClass.DEFENSIVE_ANALYSIS)],
           tools=["Splunk", "Elastic", "Wazuh"],
           prerequisites=[8, 15],
           safety=SafetyClass.DEFENSIVE_ANALYSIS,
           objectives=["SIEM, threat hunting and incident response as a defender"]),

    _offensive(17, "Red Team Concepts",
               topics=["Authorized adversary emulation", "OPSEC concepts",
                       "Detection-evasion concepts at a defensive and high level",
                       "Attack chains", "Reporting", "MITRE ATT&CK", "Cyber Kill Chain"],
               tools=["MITRE ATT&CK", "Cyber Kill Chain"],
               prerequisites=[10, 16],
               objectives=["Education, detection engineering, reporting and isolated authorized "
                           "simulation only — no real-world intrusion automation"]),

    Module(18, "Application Security",
           topics=["SAST", "DAST", "Dependency scanning", "Secure code review"],
           prerequisites=[7],
           safety=SafetyClass.SECURE_CODING,
           objectives=["Integrate SAST/DAST/dependency scanning and secure code review"]),

    Module(19, "Senior Software Engineering",
           topics=["Distributed systems", "Event-driven systems", "Caching", "Message queues",
                   "Mentoring", "Architecture reviews", "Technical documentation",
                   "Project estimation"],
           projects=[Project("Design YouTube"), Project("Design Netflix"), Project("Design Uber")],
           prerequisites=[7],
           objectives=["System design at scale plus technical leadership skills"]),

    Module(20, "Capstone Projects",
           topics=["Password manager", "Task manager", "Portfolio website",
                   "Defensive vulnerability scanner for authorized targets",
                   "Secure chat application", "SIEM dashboard", "Defensive EDR platform",
                   "Threat-intelligence platform", "Secure cloud architecture"],
           projects=[
               Project("Password manager", safety=SafetyClass.SECURE_CODING),
               Project("Task manager"), Project("Portfolio website"),
               Project("Defensive vulnerability scanner for authorized targets",
                       safety=SafetyClass.AUTHORIZED_LAB),
               Project("Secure chat application", safety=SafetyClass.SECURE_CODING),
               Project("SIEM dashboard", safety=SafetyClass.DEFENSIVE_ANALYSIS),
               Project("Defensive EDR platform", safety=SafetyClass.DEFENSIVE_ANALYSIS),
               Project("Threat-intelligence platform", safety=SafetyClass.DEFENSIVE_ANALYSIS),
               Project("Secure cloud architecture", safety=SafetyClass.SECURE_CODING)],
           prerequisites=[18, 19],
           objectives=["Beginner→advanced capstones, defensive-first"]),
]


CERTIFICATION_TIMELINE = {
    "Year 1": ["CompTIA A+", "CompTIA Network+", "CompTIA Security+"],
    "Year 2": ["Linux+", "eJPT"],
    "Year 3": ["PNPT", "CySA+"],
    "Year 4": ["OSCP", "AWS Certified Security – Specialty or its current official equivalent"],
    "Year 5": ["CISSP", "Appropriate GIAC certifications"],
}


def training_capability_note() -> str:
    """Honest statement for the UI: this subsystem does NOT retrain a base model."""
    return ("This curriculum organises knowledge, retrieval, progress and safe-practice "
            "guidance. It does NOT train or fine-tune the underlying language model — no "
            "model weights change by adding curriculum content.")


# ── target-scope permission gate (default-deny) ──────────────────────────────

def validate_target_scope(target: str, authorized_scope: set[str] | None = None) -> tuple[bool, str]:
    """Decide whether a security action may touch ``target``.

    Loopback/localhost is always allowed (a self-contained lab).  Any other host
    is permitted ONLY if it appears in ``authorized_scope`` (explicit, informed
    authorisation).  Everything else — public hosts, private ranges without
    explicit authorisation, and anything ambiguous — is DENIED by default."""
    t = str(target or "").strip().lower()
    if not t:
        return False, "no target specified — denied (default-deny)"
    authorized = {a.strip().lower() for a in (authorized_scope or set())}
    if t in authorized:
        return True, "target is explicitly authorised"

    host = t
    m = re.match(r"^[a-z]+://(\[[^\]]+\]|[^/]+)", t)
    if m:
        host = m.group(1)
    host = host.strip("[]")
    # Strip a trailing :port only for hostnames/IPv4 (a single colon); IPv6
    # addresses contain multiple colons and must not be split.
    if host.count(":") == 1:
        host = host.split(":")[0]

    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".localhost"):
        return True, "loopback/localhost — self-contained lab"

    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback:
            return True, "loopback address"
        # Private ranges are lab-like but still require explicit authorisation.
        return False, ("private/lab address requires explicit authorisation — denied"
                       if ip.is_private else "external/public address — denied (default-deny)")
    except ValueError:
        pass  # not a bare IP — treat as a hostname

    return False, "unauthorised or external hostname — denied (default-deny)"


class SecurityActivityClass(str, Enum):
    EDUCATION = "education"
    DEFENSIVE_ANALYSIS = "defensive_analysis"
    SECURE_CODING = "secure_coding"
    AUTHORIZED_LAB = "authorized_lab"
    INTRUSIVE = "intrusive"


def gate_security_action(activity: SecurityActivityClass, target: str = "",
                         authorized_scope: set[str] | None = None) -> tuple[bool, str]:
    """Permission gate before any security-related system/network action.

    Non-actioning classes (education, defensive analysis, secure coding) are
    always permitted — they touch no external system.  AUTHORIZED_LAB actions
    are permitted only against an authorised/loopback target.  INTRUSIVE actions
    are denied outright: this subsystem never automates real-world intrusion."""
    if activity in (SecurityActivityClass.EDUCATION,
                    SecurityActivityClass.DEFENSIVE_ANALYSIS,
                    SecurityActivityClass.SECURE_CODING):
        return True, "knowledge/defensive activity — no external action"
    if activity is SecurityActivityClass.AUTHORIZED_LAB:
        ok, reason = validate_target_scope(target, authorized_scope)
        return ok, ("authorised lab action permitted: " + reason if ok
                    else "authorised-lab action blocked: " + reason)
    return False, ("intrusive/real-world actions are not permitted by this subsystem "
                   "(education, detection engineering and isolated authorised simulation only)")


# ── curriculum engine (search, retrieval, progress) ──────────────────────────

class CyberCurriculum:
    """Search, retrieval and progress over the versioned curriculum."""

    def __init__(self, modules: list[Module] | None = None,
                 progress_path: Path | None = None) -> None:
        self.modules = modules if modules is not None else CURRICULUM_MODULES
        self.version = CURRICULUM_VERSION
        self._progress_path = progress_path or PROGRESS_PATH

    def module(self, number: int) -> Module | None:
        return next((m for m in self.modules if m.number == number), None)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = [w for w in re.findall(r"[a-z0-9+]+", str(query or "").lower()) if w]
        if not terms:
            return []
        scored: list[tuple[int, Module]] = []
        for m in self.modules:
            text = m.searchable_text()
            score = sum(text.count(term) for term in terms)
            if score:
                scored.append((score, m))
        scored.sort(key=lambda x: (-x[0], x[1].number))
        return [{"module": m.number, "title": m.title, "score": s,
                 "safety": m.safety.value, "isolation_required": m.isolation_required}
                for s, m in scored[:limit]]

    def outline(self) -> list[dict[str, Any]]:
        return [{"module": m.number, "title": m.title, "safety": m.safety.value,
                 "authorization_required": m.authorization_required,
                 "isolation_required": m.isolation_required,
                 "topics": len(m.topics), "labs": len(m.labs), "projects": len(m.projects)}
                for m in self.modules]

    # ── progress ──────────────────────────────────────────────────────────────

    def _load_progress(self) -> dict[str, Any]:
        try:
            if self._progress_path.exists():
                return json.loads(self._progress_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_progress(self, data: dict[str, Any]) -> None:
        try:
            self._progress_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._progress_path, json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def mark_complete(self, user: str, module_number: int) -> dict[str, Any]:
        if self.module(module_number) is None:
            raise ValueError(f"no module {module_number}")
        data = self._load_progress()
        entry = data.setdefault(user, {"completed": [], "updated_at": 0})
        if module_number not in entry["completed"]:
            entry["completed"].append(module_number)
            entry["completed"].sort()
        entry["updated_at"] = int(time.time())
        self._save_progress(data)
        return self.progress(user)

    def progress(self, user: str) -> dict[str, Any]:
        data = self._load_progress()
        completed = sorted(set(data.get(user, {}).get("completed", [])))
        total = len(self.modules)
        return {
            "user": user,
            "completed": completed,
            "total_modules": total,
            "percent": round(100.0 * len(completed) / total, 1) if total else 0.0,
            "next_module": next((m.number for m in self.modules if m.number not in completed), None),
        }


def validate_curriculum(modules: list[Module] | None = None) -> list[str]:
    """Structural + safety validation used by tests and content contributors."""
    mods = modules if modules is not None else CURRICULUM_MODULES
    errs: list[str] = []
    numbers = [m.number for m in mods]
    if numbers != list(range(1, len(mods) + 1)):
        errs.append("module numbers must be contiguous starting at 1")
    for m in mods:
        if not m.title or not m.topics:
            errs.append(f"module {m.number} missing title/topics")
        # Every offensive/destructive module must be gated.
        if m.safety in _REQUIRES_ISOLATION or m.authorization_required:
            if not m.isolation_required:
                errs.append(f"module {m.number} is offensive but not isolation_required")
            if not m.legal_reminder:
                errs.append(f"module {m.number} is offensive but has no legal_reminder")
        for prereq in m.prerequisites:
            if prereq not in numbers:
                errs.append(f"module {m.number} has unknown prerequisite {prereq}")
    return errs
