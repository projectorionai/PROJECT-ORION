"""
Cybersecurity knowledge base — defensive, mechanism-first security expertise.

Mirrors the neuroscience and programming knowledge bases: a curated corpus of
real, load-bearing security knowledge (the CIA triad, threat modelling, the
OWASP risks, cryptography, network defence, identity, malware behaviour,
detection and incident response, secure development, cloud/container hardening)
seeded idempotently into the KNOWLEDGE memory tier, plus a persona boost that
sharpens ORION's security register and a query API for offline recall.

Orientation is DEFENSIVE and educational — how attacks work so they can be
prevented, detected and remediated — grounding ORION's answers in fundamentals
rather than guesswork, online or offline.
"""

from __future__ import annotations

import re
from typing import Any, Optional

SEED_MARKER = "cyber_corpus_seeded_v1"

# (topic, fact) — real, defensive-first cybersecurity knowledge.
CORPUS: list[tuple[str, str]] = [
    ("CIA triad", "Security protects Confidentiality (only authorised access), Integrity (data is not "
     "tampered with), and Availability (systems are usable when needed). Most controls map to one or "
     "more of these; a good design balances all three against usability."),
    ("Defense in depth", "Layer independent controls so no single failure is fatal: network segmentation, "
     "host hardening, least privilege, MFA, monitoring and backups. An attacker must defeat every layer; "
     "a defender only needs one layer to catch them."),
    ("Least privilege", "Grant the minimum access needed for a task, for the shortest time. Separate duties, "
     "use role-based access, and review entitlements. It shrinks the blast radius when any account or "
     "service is compromised."),
    ("Zero trust", "Never trust based on network location; authenticate and authorise every request, verify "
     "device posture, and assume breach. Replace the flat trusted-LAN model with per-request, identity-"
     "centric policy."),
    ("Threat modelling", "Systematically ask: what are we building, what can go wrong, what do we do about it, "
     "did we do a good job. STRIDE enumerates Spoofing, Tampering, Repudiation, Information disclosure, "
     "Denial of service, Elevation of privilege. Model early, on data-flow diagrams."),
    ("Attack surface", "The sum of points an attacker can reach: open ports, endpoints, inputs, dependencies, "
     "credentials and people. Reduce it — close unused services, minimise dependencies, and validate every "
     "entry point."),
    ("OWASP Top 10", "The most critical web risks: broken access control, cryptographic failures, injection, "
     "insecure design, security misconfiguration, vulnerable components, auth failures, integrity failures, "
     "logging gaps and SSRF. Broken access control is consistently number one."),
    ("Injection (SQL etc.)", "Untrusted input interpreted as code/commands. Defeat it by never concatenating "
     "input into queries: use parameterised statements / prepared queries, allow-list where structure is "
     "needed, and apply least-privilege DB accounts."),
    ("Cross-site scripting (XSS)", "Attacker script runs in a victim's browser because output isn't escaped. "
     "Defend by contextual output encoding, a strict Content-Security-Policy, framework auto-escaping, and "
     "treating all user content as untrusted."),
    ("CSRF", "A logged-in victim's browser is tricked into making a state-changing request. Defend with anti-"
     "CSRF tokens, SameSite cookies, and requiring re-authentication or an explicit action for sensitive "
     "operations."),
    ("SSRF", "The server is coaxed into making requests to internal targets (e.g. cloud metadata endpoints). "
     "Defend by allow-listing outbound destinations, blocking link-local/metadata IPs, and validating URLs "
     "server-side."),
    ("Authentication vs authorisation", "Authentication proves who you are; authorisation decides what you may "
     "do. Keep them separate. Enforce authorisation on the server for every request — never rely on the UI "
     "hiding a control."),
    ("Passwords & hashing", "Store passwords only as slow, salted hashes (argon2id, bcrypt, scrypt) — never "
     "plaintext, never fast hashes like MD5/SHA-1. Salt defeats rainbow tables; a slow function defeats "
     "brute force. Add pepper and rate limiting."),
    ("Multi-factor authentication", "Combine factors — something you know, have, or are. Phishing-resistant "
     "MFA (FIDO2/WebAuthn hardware keys) beats SMS codes, which are vulnerable to SIM-swap and interception. "
     "MFA blocks the vast majority of account-takeover attempts."),
    ("Encryption: symmetric vs asymmetric", "Symmetric (AES) uses one shared key — fast, for bulk data. "
     "Asymmetric (RSA, ECC) uses a public/private key pair — for key exchange and signatures. TLS uses "
     "asymmetric to agree a symmetric session key, then symmetric for speed."),
    ("Hashing vs encryption", "Hashing is one-way (integrity, fingerprints, passwords) and cannot be reversed; "
     "encryption is two-way (confidentiality) and reversible with a key. Don't confuse them: you verify a "
     "hash, you decrypt a ciphertext."),
    ("TLS & PKI", "TLS authenticates the server (and optionally client) and encrypts the channel. Trust chains "
     "from a Certificate Authority to a leaf certificate; validate the chain, hostname and expiry. Prefer "
     "TLS 1.3; disable weak ciphers and renegotiation."),
    ("Secrets management", "Keep credentials out of source and images. Use a secrets manager/vault, inject at "
     "runtime, rotate regularly, and scope narrowly. Detect leaked keys with scanning and revoke on exposure."),
    ("Network security", "Segment networks and default-deny with firewalls; only open needed ports. Use VPNs "
     "or zero-trust access for remote entry. IDS/IPS inspect traffic for known-bad patterns; a WAF filters "
     "web-layer attacks."),
    ("Malware types", "Virus (attaches to files), worm (self-propagates), trojan (disguised), ransomware "
     "(encrypts for extortion), spyware/keylogger (steals data), rootkit (hides at low level), botnet (remote "
     "control). Defence: patching, EDR, least privilege and backups."),
    ("Ransomware defence", "Assume it will try. Keep offline, immutable, tested backups (3-2-1 rule); segment "
     "networks to limit spread; patch and disable risky services (RDP exposure, macros); and rehearse "
     "recovery. Paying is neither reliable nor recommended."),
    ("Phishing & social engineering", "Most breaches start with a human, not an exploit: pretexting, urgency, "
     "authority and spoofed senders. Defend with awareness training, DMARC/SPF/DKIM email authentication, "
     "link/attachment sandboxing, and phishing-resistant MFA."),
    ("Privilege escalation", "Turning limited access into higher access via misconfigurations, unpatched "
     "kernels, weak service permissions or credential theft. Defend by patching, least privilege, removing "
     "local admin, and monitoring for anomalous privilege use."),
    ("Buffer overflow", "Writing past a buffer's bounds overwrites adjacent memory and can hijack control flow. "
     "Mitigations: memory-safe languages, bounds checking, stack canaries, ASLR, DEP/NX, and compiler "
     "hardening flags."),
    ("Denial of service (DoS/DDoS)", "Overwhelm a service so legitimate users can't reach it, often from many "
     "sources (DDoS). Defend with rate limiting, upstream scrubbing/CDN, autoscaling, and dropping malformed "
     "traffic early."),
    ("MITRE ATT&CK", "A public knowledge base of adversary tactics (the why: initial access, execution, "
     "persistence, privilege escalation, lateral movement, exfiltration) and techniques (the how). Use it to "
     "map detections and find coverage gaps."),
    ("Detection & SIEM", "Centralise logs (auth, network, endpoint, cloud) into a SIEM; write detections for "
     "known techniques and anomalies; reduce alert fatigue with tuning. You cannot respond to what you cannot "
     "see — logging is a security control."),
    ("Incident response", "Prepare, Identify, Contain, Eradicate, Recover, Lessons-learned. Contain fast to "
     "stop spread, preserve forensic evidence, communicate clearly, and hold a blameless post-mortem to fix "
     "root causes, not symptoms."),
    ("Patch & vulnerability management", "Inventory assets, scan for known CVEs, prioritise by exploitability "
     "and exposure (not just CVSS), and patch on a schedule with emergency lanes for actively-exploited bugs. "
     "Unpatched, internet-facing services are the classic entry point."),
    ("Secure SDLC", "Bake security into development: threat model in design, use safe defaults and vetted "
     "libraries, review code, run SAST/DAST and dependency scanning in CI, and manage secrets. Fixing a flaw "
     "in design is far cheaper than in production."),
    ("Cloud & container security", "Shared-responsibility: the provider secures the cloud, you secure what's in "
     "it. Lock down IAM (no wildcards), private networking, encryption at rest/in transit, and CIS-benchmark "
     "hardening. Scan images, run rootless, drop capabilities, and never bake secrets into layers."),
    ("Supply-chain security", "Attackers compromise dependencies, build systems or updates to reach many "
     "victims at once. Pin and verify dependencies, generate an SBOM, use signed artifacts, and isolate the "
     "build pipeline. Trust but verify every third-party component."),
    ("Data protection & privacy", "Classify data, encrypt sensitive data at rest and in transit, minimise what "
     "you collect and retain, and control access with audit logging. Regulations (GDPR) demand lawful basis, "
     "purpose limitation and breach notification."),
]

CYBER_PERSONA_BOOST = (
    "You carry solid, current cybersecurity expertise oriented to DEFENCE: the CIA triad, threat "
    "modelling, the OWASP risks, cryptography, network and identity security, malware behaviour, "
    "detection, incident response and secure development. Explain how attacks work in order to prevent, "
    "detect and remediate them; recommend concrete, layered controls; assume breach and least privilege; "
    "and never provide operational help for wrongdoing — keep guidance lawful, authorised and educational."
)


class CyberKnowledgeBase:
    """Seeds and serves the cybersecurity corpus (mirrors ProgrammingKnowledgeBase)."""

    KEYWORDS = (
        "security", "cyber", "hack", "hacking", "exploit", "vulnerability", "vuln", "malware",
        "ransomware", "phishing", "encryption", "crypto", "tls", "ssl", "hash", "password",
        "authentication", "authorisation", "authorization", "firewall", "xss", "sql injection",
        "csrf", "ssrf", "owasp", "zero trust", "least privilege", "threat", "attack", "breach",
        "incident", "siem", "penetration", "pentest", "mfa", "mitre", "ddos", "privilege escalation",
        "supply chain", "secrets", "patch", "cve",
    )

    def __init__(self, telemetry: Any | None = None) -> None:
        self.telemetry = telemetry

    def seed(self, memory: Any) -> int:
        """Idempotently write the corpus into the KNOWLEDGE tier."""
        try:
            existing = memory.query(SEED_MARKER, limit=1)
            if existing and any(SEED_MARKER in str(r.get("value", "")) for r in existing):
                return 0
        except Exception:
            pass
        count = 0
        for topic, fact in CORPUS:
            try:
                key = "cyber_" + re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:44]
                memory.remember("knowledge", key, f"{topic}: {fact}")
                count += 1
            except Exception:
                continue
        try:
            memory.remember("knowledge", "cyber_seed_marker", SEED_MARKER)
        except Exception:
            pass
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("cyber.corpus", float(count))
        return count

    def is_cyber_query(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(k in low for k in self.KEYWORDS)

    def answer(self, query: str) -> Optional[str]:
        low = str(query or "").lower()
        best: Optional[tuple[int, str, str]] = None
        for topic, fact in CORPUS:
            score = sum(1 for w in re.findall(r"[a-z0-9+]+", low) if w in (topic + " " + fact).lower())
            if score and (best is None or score > best[0]):
                best = (score, topic, fact)
        if best is None:
            return None
        return f"{best[1]}: {best[2]}"
