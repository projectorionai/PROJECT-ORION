"""
Teaching cybersecurity the way a student actually learns it: by recognition.

  "there must be key identifiers that teach me PROPERLY so that it makes sense
   for me as a student ... so then if I look at a piece of code in the future
   from a keylogger or a rootkit then I'll identify that ... solely for
   educational purposes ... with drawbacks of course so it doesn't harm other
   systems."

cyber_curriculum.py already provides the SYLLABUS — what to study, in order,
with safety gating. This provides the LESSONS: for each technique, what it is,
how it works, and — the part the request is really about — the KEY IDENTIFIERS
that let you spot it in code you did not write. That is the defensible core of
security education: learning to recognise an attack is what makes you able to
defend against it, and it is a skill you use by reading, not by running.

Where a technique can be practised safely — web vulnerabilities on a
deliberately-vulnerable target, a port scan of your own machine — the lesson
says exactly where and how. Where it cannot be made safe — a rootkit, ransomware
— the lesson teaches the concept and the identifiers and stops there, on
purpose: there is no version of "here is working ransomware" that belongs in a
learning tool, and the recognition skill does not need one.

Everything points at LEGAL practice ranges (TryHackMe, Hack The Box, PortSwigger
Web Security Academy, OverTheWire, VulnHub, DVWA) — systems built to be attacked,
which is the only place any of this is practised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    WEB = "web"
    NETWORK = "network"
    MALWARE = "malware"
    ACCESS = "access-and-escalation"
    SOCIAL = "social-engineering"
    CRYPTO = "cryptography"


class DemoPolicy(str, Enum):
    """How far a lesson may go towards hands-on, by how dangerous it is."""

    SAFE_LAB = "safe_lab"          # a full hands-on lab, on a lawful target
    DEFANGED = "defanged"          # only a loud, consented, non-weaponisable demo
    CONCEPT_ONLY = "concept_only"  # recognition only — no runnable attack code


#: Lawful places to practise. Named once so every lesson points to real ranges
#: rather than "a machine you have permission to test", which a student cannot
#: act on.
PRACTICE_RANGES = {
    "web": ["PortSwigger Web Security Academy (free)", "DVWA", "OWASP Juice Shop",
            "TryHackMe (web paths)"],
    "network": ["TryHackMe", "Hack The Box", "your own machine / a VM you own"],
    "malware": ["an ISOLATED, offline analysis VM (FlareVM / REMnux) you own",
                "malware-analysis courses; never a networked machine"],
    "access": ["TryHackMe", "Hack The Box", "VulnHub images", "OverTheWire (Bandit)"],
    "general": ["TryHackMe", "Hack The Box", "OverTheWire", "PortSwigger Academy"],
}


@dataclass
class Lesson:
    topic: str
    title: str
    category: Category
    what_it_is: str
    how_it_works: list[str]
    identifiers: list[str]           # ← how to RECOGNISE it (the point)
    safeguards: list[str]            # the "drawbacks" that keep a demo harmless
    demo_policy: DemoPolicy
    practice: list[str] = field(default_factory=list)
    study_path: list[str] = field(default_factory=list)
    aliases: tuple[str, ...] = ()

    def format(self) -> str:
        """A readable, student-facing lesson."""
        lines = [f"# {self.title}  ({self.category.value})", "",
                 self.what_it_is, "", "## How it works"]
        lines += [f"  {i}. {step}" for i, step in enumerate(self.how_it_works, 1)]
        lines += ["", "## Key identifiers — how to recognise it in code or behaviour"]
        lines += [f"  • {sig}" for sig in self.identifiers]
        lines += ["", "## Safe study"]
        if self.demo_policy is DemoPolicy.CONCEPT_ONLY:
            lines.append("  This is a category that cannot be made safe to build, "
                         "so we learn it by RECOGNITION only — no working attack "
                         "code. The identifiers above are what you actually need.")
        elif self.demo_policy is DemoPolicy.DEFANGED:
            lines.append("  Learn the mechanism only through a loud, consented, "
                         "throwaway demo in an isolated VM you own — with these "
                         "safeguards, never anything stealthy or persistent:")
        else:
            lines.append("  Safe to practise hands-on, on a lawful target:")
        lines += [f"  - {s}" for s in self.safeguards]
        if self.practice:
            lines += ["", "## Where to practise (legal ranges only)"]
            lines += [f"  - {p}" for p in self.practice]
        if self.study_path:
            lines += ["", "## How to study it"]
            lines += [f"  - {s}" for s in self.study_path]
        return "\n".join(lines)


def _lesson(topic, title, category, what, how, ids, safeguards, policy,
            practice_key="general", study=None, aliases=()):
    return Lesson(
        topic=topic, title=title, category=category, what_it_is=what,
        how_it_works=how, identifiers=ids, safeguards=safeguards,
        demo_policy=policy, practice=PRACTICE_RANGES.get(practice_key, []),
        study_path=study or [], aliases=aliases)


LESSONS: dict[str, Lesson] = {
    # ── web (safe to practise on deliberately-vulnerable targets) ─────────────
    "sql_injection": _lesson(
        "sql_injection", "SQL Injection", Category.WEB,
        "Untrusted input reaches a database query as CODE rather than as DATA, "
        "so an attacker's text becomes part of the SQL the server runs.",
        ["User input is concatenated straight into a query string",
         "The database executes the combined string, so input can add clauses",
         "A classic probe (' OR '1'='1) turns a login check always-true",
         "UNION SELECT reads other tables; error messages leak schema"],
        ["String-built SQL: \"... WHERE name='\" + user + \"'\" or f-strings/%/.format in a query",
         "A query executed without parameters/placeholders (no ? or :name binding)",
         "Input echoed into an ORDER BY / table name (can't be parameterised)",
         "Sudden UNION SELECT, --, /*, or ' OR in logs or request bodies",
         "DB errors surfaced to the page ('SQL syntax near ...')"],
        ["Fix pattern to learn: PARAMETERISED queries / prepared statements — "
         "the input can never be code",
         "Practise only on DVWA/Juice Shop/PortSwigger, which exist to be attacked",
         "Never point it at a site you do not own or are not authorised to test"],
        DemoPolicy.SAFE_LAB, "web",
        study=["PortSwigger's SQLi labs, in order — they grade you",
               "Then rewrite the vulnerable query with placeholders and re-test"],
        aliases=("sqli", "sql")),
    "xss": _lesson(
        "xss", "Cross-Site Scripting (XSS)", Category.WEB,
        "A site reflects or stores attacker input and a victim's browser then "
        "runs it as JavaScript in the site's origin.",
        ["Input is written into the page without being escaped",
         "The browser cannot tell the injected <script> from the site's own",
         "The script runs with the victim's session — it can steal cookies, act as them"],
        ["User input written into HTML with innerHTML / document.write and no escaping",
         "Server templates that emit a variable unescaped ({{ x|safe }}, <%= raw %>)",
         "A reflected parameter appearing verbatim in the response body",
         "<script>, onerror=, javascript: or <img src=x onerror=...> in inputs/logs"],
        ["Fix pattern: context-aware OUTPUT ENCODING + a Content-Security-Policy",
         "Practise on Juice Shop / PortSwigger XSS labs",
         "Test only your own or a training app"],
        DemoPolicy.SAFE_LAB, "web",
        study=["PortSwigger XSS labs", "Learn the three types: reflected, stored, DOM"],
        aliases=("cross-site scripting", "cross site scripting")),
    # ── network (safe on your own hosts) ──────────────────────────────────────
    "port_scanning": _lesson(
        "port_scanning", "Port Scanning", Category.NETWORK,
        "Probing which network ports a host has open, to map what services it "
        "runs — the reconnaissance step before anything else.",
        ["Send connection probes to a range of ports",
         "An open port answers; a closed one refuses; a filtered one is silent",
         "Service/version detection fingerprints what is listening"],
        ["Bursts of connection attempts across many ports from one source",
         "nmap / masscan in command history or process lists",
         "SYN packets with no completed handshake (a SYN scan)",
         "Sequential or fanned destination ports in firewall/IDS logs"],
        ["Scan ONLY your own machine (127.0.0.1) or a VM you own",
         "ORION already ships security_recon.py behind a default-deny scope gate — "
         "it refuses targets you have not authorised",
         "Unsolicited scanning of others is unlawful in many places"],
        DemoPolicy.SAFE_LAB, "network",
        study=["Run `nmap localhost` on your own machine and read every field",
               "TryHackMe 'Nmap' room"],
        aliases=("nmap", "portscan", "scanning")),
    "reverse_shell": _lesson(
        "reverse_shell", "Reverse Shell", Category.ACCESS,
        "A shell on a target that connects OUT to the attacker (rather than the "
        "attacker connecting in), because outbound connections slip past most "
        "firewalls.",
        ["Code on the target opens a socket back to the attacker's listener",
         "It wires that socket to a shell's stdin/stdout/stderr",
         "The attacker types commands over the connection"],
        ["A socket connect() to an external IP followed by dup2 onto fd 0/1/2",
         "subprocess/os.system fed from a socket; /bin/sh -i over a pipe",
         "bash -i >& /dev/tcp/…/… ; nc -e ; python -c 'import socket,subprocess,os'",
         "A process with an outbound connection AND a child shell it did not need"],
        ["Understand it as the detection target it is — the identifiers above are "
         "exactly what blue teams alert on",
         "Practise ONLY between two VMs you own on an isolated network",
         "TryHackMe/HTB provide safe target boxes for this"],
        DemoPolicy.DEFANGED, "access",
        study=["TryHackMe 'What the Shell' room",
               "Learn the matching defence: egress filtering + EDR on child shells"],
        aliases=("reverse shell", "bind shell", "revshell")),
    # ── malware (recognition only — no working code) ──────────────────────────
    "keylogger": _lesson(
        "keylogger", "Keyloggers", Category.MALWARE,
        "Software that records keystrokes. Legitimately it is how accessibility "
        "and macro tools work; maliciously it captures passwords. The skill "
        "worth having is recognising one.",
        ["It registers for keyboard events at the OS or library level",
         "It writes captured keys somewhere (file, memory, or straight out over "
         "the network)",
         "A malicious one hides — no window, autostart, obscured file — and "
         "exfiltrates"],
        ["Windows: SetWindowsHookEx(WH_KEYBOARD_LL, …), GetAsyncKeyState in a loop",
         "Python: pynput.keyboard.Listener, keyboard.hook/on_press, ctypes onto user32",
         "Linux: reading /dev/input/event*, an X11 grab",
         "Written to a hidden file in %APPDATA%/ProgramData/temp, or POSTed out",
         "Persistence: a Run registry key, a scheduled task, a startup entry",
         "No visible window and no legitimate reason for the process to read the keyboard"],
        ["Learned by RECOGNITION — the identifiers above are the deliverable",
         "If you ever build a mechanism demo to learn from, do it in an OFFLINE VM, "
         "log only to the visible console, add a permanent on-screen 'RECORDING' "
         "banner, no persistence, no network — so it can never be a tool",
         "Reading someone else's keystrokes without consent is a crime"],
        DemoPolicy.CONCEPT_ONLY, "malware",
        study=["Read annotated malware write-ups (e.g. Malware Unicorn, MalwareTech)",
               "Practise SPOTTING the identifiers in sample code, not writing it"],
        aliases=("key logger", "keystroke logger")),
    "rootkit": _lesson(
        "rootkit", "Rootkits", Category.MALWARE,
        "Malware that hides its own presence — from process lists, file listings, "
        "network tools — usually by tampering with the OS itself. Genuinely "
        "dangerous; taught here to recognise, not to build.",
        ["It gains high privilege (kernel or deep userland hooks)",
         "It intercepts the calls that would reveal it and filters itself out",
         "Everything above then reports a false, clean picture"],
        ["Userland: hooked API/import tables, LD_PRELOAD, injected DLLs filtering results",
         "Kernel: SSDT/inline hooks, a loaded driver with no vendor, DKOM unlinking "
         "a process from the list",
         "Symptoms: tools disagree (a port open in one tool, absent in another); "
         "files present on a raw read but hidden to the shell",
         "Detection tools: rootkit scanners, cross-view diffs, kernel integrity checks"],
        ["RECOGNITION ONLY. There is no safe working rootkit, and building one is "
         "not part of learning to defend against it",
         "Study detection and integrity-checking instead — that is the useful skill"],
        DemoPolicy.CONCEPT_ONLY, "malware",
        study=["Read about how detectors work (cross-view detection, integrity monitors)",
               "Volatility framework for memory-forensics recognition"],
        aliases=("root kit", "bootkit")),
    "ransomware": _lesson(
        "ransomware", "Ransomware", Category.MALWARE,
        "Malware that encrypts files and demands payment. Taught purely so you "
        "recognise it early — the identifiers below are what lets you pull the "
        "plug before it finishes.",
        ["It enumerates files, often skipping system files to keep the OS alive",
         "It encrypts each with a key only the attacker holds",
         "It drops a ransom note and often deletes shadow copies/backups"],
        ["Rapid mass file rewrites with a new extension; a note file in every folder",
         "Calls to delete shadow copies (vssadmin delete shadows, wbadmin)",
         "Crypto APIs over a whole directory tree in a short window",
         "A spike of file-modified events across many folders at once (the strongest signal)"],
        ["RECOGNITION ONLY — no working code, ever",
         "The real lesson is defence: offline backups, and killing the process the "
         "instant the mass-rewrite pattern above appears"],
        DemoPolicy.CONCEPT_ONLY, "malware",
        study=["Study incident-response playbooks and backup strategy",
               "Learn the behavioural detections above — they are what EDR uses"],
        aliases=("crypto locker", "crypto-ransomware")),
    # ── access & social ───────────────────────────────────────────────────────
    "privilege_escalation": _lesson(
        "privilege_escalation", "Privilege Escalation", Category.ACCESS,
        "Turning limited access into higher access — the step between 'a foothold' "
        "and 'full control'. Central to both attack and hardening.",
        ["Enumerate the system for a misconfiguration or a vulnerable component",
         "Abuse it — a writable service binary, a SUID file, a kernel bug, a stored credential",
         "Re-run as the higher-privileged user or root/SYSTEM"],
        ["Enumeration scripts: linpeas, winPEAS, GTFOBins lookups",
         "SUID/SGID binaries in unusual places; writable service paths; sudo misconfig",
         "Scheduled tasks / cron running writable scripts as root",
         "Credentials in config files, history, or environment variables"],
        ["Practise on TryHackMe/HTB/VulnHub boxes built for it",
         "The defensive twin is what to learn: least privilege, patching, file ACLs",
         "Only ever on systems you own or are authorised to test"],
        DemoPolicy.SAFE_LAB, "access",
        study=["TryHackMe privesc rooms (Linux and Windows)",
               "Keep a checklist — privesc is enumeration discipline more than exploits"],
        aliases=("privesc", "escalation", "priv esc")),
    "phishing": _lesson(
        "phishing", "Phishing", Category.SOCIAL,
        "Tricking a person into revealing credentials or running something, by "
        "impersonating something they trust. The most common real-world entry point.",
        ["A message impersonates a trusted sender or brand",
         "It creates urgency and a link/attachment",
         "The link leads to a credential-harvest page or a malicious file"],
        ["Sender domain that is close-but-wrong (rn for m, extra hyphens, wrong TLD)",
         "A link whose visible text differs from its real href",
         "Urgency + a request for credentials, payment or an install",
         "Attachments that ask you to 'enable macros' or run something"],
        ["Learn by DISSECTING real phishing samples (PhishTank, your own spam)",
         "Build recognition, not campaigns — sending phishing to others is illegal",
         "The defence is the lesson: verify the domain, hover before clicking, report"],
        DemoPolicy.CONCEPT_ONLY, "general",
        study=["Google's phishing quiz", "Dissect headers of real spam you receive"],
        aliases=("phish", "social engineering email")),
}


def _index() -> dict[str, str]:
    """topic + every alias → canonical topic."""
    idx: dict[str, str] = {}
    for topic, lesson in LESSONS.items():
        idx[topic] = topic
        idx[topic.replace("_", " ")] = topic
        for alias in lesson.aliases:
            idx[alias.lower()] = topic
    return idx


_ALIASES = _index()


def resolve_topic(query: str) -> str | None:
    q = str(query or "").lower().strip()
    if not q:
        return None
    if q in _ALIASES:
        return _ALIASES[q]
    # loose contains-match, longest alias first so "sql injection" beats "sql"
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if alias in q:
            return _ALIASES[alias]
    return None


def teach(query: str) -> str:
    """A full lesson for *query*, or a catalogue when it isn't recognised."""
    topic = resolve_topic(query)
    if topic is None:
        names = ", ".join(sorted(l.title for l in LESSONS.values()))
        return ("I don't have a dedicated lesson for that yet. Lessons I teach "
                f"with recognition identifiers: {names}. Ask about any of them, "
                "or ask the curriculum tool for the full 20-module syllabus.")
    return LESSONS[topic].format()


def catalogue() -> list[dict[str, str]]:
    return [{"topic": t, "title": l.title, "category": l.category.value,
             "demo_policy": l.demo_policy.value} for t, l in LESSONS.items()]


def identify(snippet: str) -> list[tuple[str, list[str]]]:
    """Given a code snippet, name the technique(s) it resembles and WHY.

    This is the recognition skill made concrete: "if I look at a piece of code
    from a keylogger I'll identify that." It matches the real signatures each
    lesson lists against the text and reports the hits, so a student can paste
    unfamiliar code and be shown what it is and which identifiers gave it away.
    Heuristic and defensive — it flags, it does not judge intent.
    """
    text = str(snippet or "").lower()
    if not text.strip():
        return []
    # Concrete code/behaviour tokens per technique, drawn from the identifiers.
    signatures: dict[str, list[str]] = {
        "keylogger": ["setwindowshookex", "wh_keyboard", "getasynckeystate",
                      "pynput", "keyboard.on_press", "keyboard.hook",
                      "/dev/input", "on_press", "keylog"],
        "reverse_shell": ["/dev/tcp/", "socket.socket", "dup2", "nc -e",
                          "/bin/sh", "pty.spawn", "subprocess.call([\"/bin/",
                          "connect((", "sh -i"],
        "sql_injection": ["' or '1'='1", "union select", "or 1=1", "'; drop",
                          "execute(\"select", "\" + user", "f\"select", "%s' %"],
        "xss": ["<script>", "onerror=", "innerhtml", "document.write",
                "javascript:", "<img src=x"],
        "port_scanning": ["nmap", "masscan", "connect_ex", "for port in range",
                          "socket.connect_ex"],
        "ransomware": ["vssadmin delete", "encrypt_file", "cryptography.fernet",
                       "os.walk", ".encrypt(", "ransom", "shadowcopy"],
        "rootkit": ["ld_preload", "ssdt", "dkom", "unlink", "hook",
                    "syscall table"],
        "privilege_escalation": ["setuid(0)", "linpeas", "winpeas", "gtfobins",
                                 "suid", "sudo -l"],
    }
    hits: list[tuple[str, list[str]]] = []
    for topic, tokens in signatures.items():
        matched = [tok for tok in tokens if tok in text]
        if matched:
            hits.append((topic, matched))
    hits.sort(key=lambda h: -len(h[1]))
    return hits


def identify_report(snippet: str) -> str:
    """Human-readable identification, for the tool."""
    hits = identify(snippet)
    if not hits:
        return ("Nothing in that matches a technique I have identifiers for. That "
                "is not proof it is safe — only that it does not trip these "
                "specific signatures.")
    lines = ["Here's what that code resembles, and what gave it away:"]
    for topic, matched in hits:
        lesson = LESSONS.get(topic)
        title = lesson.title if lesson else topic
        lines.append(f"\n▶ {title} — matched: {', '.join(matched)}")
        if lesson:
            lines.append(f"  {lesson.identifiers[0]}")
    lines.append("\nAsk me to 'teach <topic>' for the full lesson on any of these.")
    return "\n".join(lines)


__all__ = [
    "LESSONS", "PRACTICE_RANGES", "Category", "DemoPolicy", "Lesson",
    "catalogue", "identify", "identify_report", "resolve_topic", "teach",
]
