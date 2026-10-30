"""Reviewed, small learning steps for ORION's paced teaching sessions.

All content is supplied here, never generated from a learner's freeform input.
This module selects immutable text; it does not run code, open files, contact a
target or change hardware. Longer lesson services remain the reference material.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import cyber_exercises, cyber_teaching, electronics_teaching


@dataclass(frozen=True)
class LearningStep:
    title: str
    goal: str
    content: str
    question: str
    choices: tuple[str, ...] = ()
    answer: str | None = None
    feedback: str | None = None
    simpler: str = ""
    recap: str = ""


@dataclass(frozen=True)
class LearningPlan:
    topic: str
    title: str
    steps: tuple[LearningStep, ...]


def _step(title, goal, content, question, choices, answer, feedback, simpler, recap):
    return LearningStep(title, goal, content, question, choices, answer, feedback,
                        simpler, recap)


def _reflect(title, goal, content, question, simpler, recap):
    return LearningStep(title, goal, content, question, simpler=simpler, recap=recap)


_SQL = LearningPlan("sql_injection", "Secure SQL in small steps", (
    _step(
        "Keep input as data", "Recognise a risky query.",
        "SQL instructions tell a database what to do. Joining a supplied username "
        "directly into those instructions can let the input change the query. "
        "Look for SQL made with string concatenation or f-strings near database calls.",
        "Which pattern needs review?",
        ("SQL joined with user input", "A query with bound values", "A static query without input"), "A",
        "Review how supplied values reach the query; use bound parameters.",
        "Think of SQL as instructions and a username as a label. A label should "
        "never be allowed to rewrite the instructions.",
        "Supplied values must remain data, separate from SQL instructions."),
    _step(
        "Bind one value", "Read a parameterised query.",
        "This synthetic database exists only in memory. The ? binds the whole "
        "supplied string as one value. It cannot add SQL instructions. ORION "
        "displays this example; it does not run it.\n\n"
        '```python\nimport sqlite3\ndb = sqlite3.connect(":memory:")\n'
        'db.execute("CREATE TABLE learners (name TEXT)")\n'
        'db.execute("INSERT INTO learners VALUES (?)", ("Ada",))\n'
        'supplied = "Ada\' OR \'1\'=\'1"\n'
        'rows = db.execute("SELECT name FROM learners WHERE name = ?", (supplied,)).fetchall()\n'
        'print(rows)\ndb.close()\n```',
        "What does print(rows) show?", ("All learners", "An empty list: []", "A query syntax error"), "B",
        "No name equals the complete supplied text, so the lookup returns [].",
        "The question mark holds the entire supplied name as text. No learner "
        "has that complete name, so the result is an empty list.",
        "A bound parameter compares the complete input as a literal value."),
    _step(
        "Know the boundary", "Distinguish values from identifiers.",
        "A placeholder binds a value such as a username. It does not choose a "
        "table name or SQL keyword. If users may choose a sort column, map "
        "reviewed choices to fixed column names rather than inserting arbitrary input.",
        "Which can a value placeholder bind?", ("A table name", "SELECT", "A username"), "C",
        "Placeholders bind values; identifiers need a separate reviewed allow-list.",
        "A value slot holds a username or number. The database's instruction "
        "words and table names need a separate, controlled choice.",
        "Parameter binding protects values; it does not bind SQL identifiers."),
    _reflect(
        "Choose a safe practice space", "Keep practice within permission.",
        "Practise with supplied examples or an authorised training lab. Before "
        "testing a live system, agree its targets, methods and stop conditions "
        "in writing. A secure query example is not permission to test a website.",
        "Where could you practise with explicit permission?",
        "Use a training environment whose rules permit the exercise. Having "
        "a website address or an account does not establish testing permission.",
        "Practice needs an authorised environment and an agreed scope."),
))


_XSS = LearningPlan("xss", "HTML text safety in small steps", (
    _step(
        "Spot an HTML sink", "Recognise unsafe text placement.",
        "A browser treats HTML as instructions for a page. Supplied text can "
        "become markup if a site places it into HTML without safe handling. "
        "Review untrusted values passed to innerHTML or document.write.",
        "Which deserves a security review?",
        ("Escaped text in a text node", "Untrusted text sent to innerHTML", "A fixed HTML template"), "B",
        "Trace that value and check the output context and its safe handling.",
        "Text from another person should be shown as text. An HTML sink can "
        "interpret that text as page instructions unless handled safely.",
        "Trace untrusted text into HTML sinks when reviewing for XSS."),
    _step(
        "Escape HTML text", "Read one safe text conversion.",
        "This example prepares synthetic text for an HTML text context. escape "
        "changes markup characters into text representations. The result is "
        "printed text; ORION does not run it or open a browser.\n\n"
        '```python\nfrom html import escape\nmessage = "Hello <Ada> & Bo"\n'
        'safe_text = escape(message, quote=True)\nprint(safe_text)\n```',
        "Which output keeps the message as text?",
        ("Hello &lt;Ada&gt; &amp; Bo", "Hello <Ada> & Bo", "Hello Ada & Bo"), "A",
        "The encoded characters preserve the words without treating them as HTML markup.",
        "The less-than sign has meaning in HTML. Escaping gives it a text "
        "representation so the browser can display the sign instead.",
        "HTML escaping turns markup characters into displayable text representations."),
    _step(
        "Check the context", "Avoid applying one fix everywhere.",
        "The example covers HTML text. JavaScript, CSS and URL values use "
        "different rules. Prefer a framework's autoescaping and safe APIs "
        "appropriate to each context. HTML escaping alone is not a universal XSS fix.",
        "Does HTML escaping secure every JavaScript context?",
        ("Always", "Yes, if quotes are escaped", "No; context matters"), "C",
        "Choose safe handling for the actual output context.",
        "Different parts of a page speak different languages. A rule for "
        "ordinary HTML text may be wrong inside JavaScript or a URL.",
        "Safe output handling depends on where the value will be used."),
    _reflect(
        "Explain the review", "Practise reading safely.",
        "Use supplied text examples or an authorised training app. Describe "
        "where input enters, where it appears and which safe API handles it. "
        "You can practise this reasoning without sending a payload to any live site.",
        "Which part of a page would you check first?",
        "Start by tracing one piece of supplied text to its destination. "
        "Naming that destination is enough for this small review exercise.",
        "A useful review traces input to its output context and safe handling."),
))


_ACCESS = LearningPlan("access_control", "Object permissions in small steps", (
    _step(
        "Check the specific record", "Separate identity from permission.",
        "Authentication establishes who is signed in. Authorisation decides "
        "what they may access. Knowing a report's number does not grant "
        "permission to read it; the server must check the requested record.",
        "What makes reading a report permissible?",
        ("Knowing its number", "Permission for that record", "Being signed in"), "B",
        "The trusted current identity needs permission for the specific record.",
        "Signing in is like showing who you are. Reading a report still "
        "requires checking whether that report is available to you.",
        "Check permission for each requested object, even after sign-in."),
    _step(
        "Read the ownership check", "Find where access is denied.",
        "These practice records exist only in memory. The function denies "
        "a missing report or a different owner before returning its text. "
        "ORION displays the code; it does not run it.\n\n"
        '```python\nreports = {101: {"owner": "Ada", "text": "practice note"}}\n'
        'def read_report(user, report_id):\n    report = reports.get(report_id)\n'
        '    if report is None or report["owner"] != user:\n        return "DENIED"\n'
        '    return report["text"]\nprint(read_report("Bo", 101))\n```',
        "What is printed for Bo?", ("practice note", "An ownership lookup error", "DENIED"), "C",
        "The report belongs to Ada, so Bo's request is denied.",
        "The report belongs to Ada, but Bo asks for it. The ownership "
        "check happens first, so Bo receives DENIED instead of the report text.",
        "The permission check must happen before returning the record's contents."),
    _step(
        "Trust the right identity", "Locate the identity source.",
        "A real server gets the current user from trusted authentication. "
        "A user ID supplied in a request is just input; it must not establish "
        "who the requester is. Our supplied names only model identities for practice.",
        "Where should the server get the current identity?",
        ("Trusted authentication", "A caller's owner field", "An ID hidden in the page"), "A",
        "Use trusted authentication, then apply the object's permission policy.",
        "Someone typing Ada into a request has not proved they are Ada. "
        "The server needs its trusted sign-in result before checking permissions.",
        "Caller-supplied identity fields cannot replace trusted authentication."),
    _step(
        "Deny missing records", "Handle an unknown object safely.",
        "reports.get returns None when an ID is missing. The first part of "
        "the if condition then denies the request. This keeps the function "
        "from treating an unknown record as accessible.",
        "What should read_report return for missing ID 999?",
        ("Any nearby report", "DENIED", "The caller's own report"), "B",
        "No record means no permitted content to return.",
        "If the requested report does not exist, there is nothing to authorise "
        "or return. Stop at the missing-record check rather than guessing.",
        "An unknown object must not fall through to an allowed response."),
))


_LOGS = LearningPlan("log_analysis", "Defensive log review in small steps", (
    _step(
        "Define a window", "Understand the practice data.",
        "The next example uses synthetic login events from one five-minute "
        "practice window. A count only makes sense alongside its time window; "
        "three failures in a minute and in a year tell different stories.",
        "Which detail makes a failure count interpretable?",
        ("The time window", "Only the all-time total", "The account's display name"), "A",
        "An explicit time window gives the count context.",
        "Count events within a stated period. Without knowing the period, "
        "you cannot tell whether failures arrived together or far apart.",
        "A detection count needs an explicit time window."),
    _step(
        "Count failures", "Read a small event summary.",
        "Counter groups failed events by account. These supplied events "
        "contain no real users or passwords. The example reads no files "
        "and contacts no service; ORION displays it without running it.\n\n"
        '```python\nfrom collections import Counter\n'
        'events = [("Ada", "failure"), ("Ada", "failure"), ("Bo", "success")]\n'
        'failures = Counter(name for name, result in events if result == "failure")\n'
        'print(failures["Ada"])\n```',
        "How many failures belong to Ada?", ("0", "2", "3"), "B",
        "Two Ada events are failures; Bo's success is not counted.",
        "Keep only events marked failure, then group them by account name. "
        "Ada has two matching events in this supplied practice list.",
        "Filter events by outcome before counting them for each account."),
    _step(
        "Review without jumping to conclusions", "Interpret a signal carefully.",
        "A failure threshold is a reason to review, not proof of an attack. "
        "Mistyped passwords or a broken client can cause failures too. "
        "Check timing and surrounding events before deciding what happened.",
        "What does crossing a teaching threshold mean?",
        ("An attack is proven", "Delete the account", "Review the context"), "C",
        "A threshold flags events for review; it does not establish intent.",
        "An alert is a prompt to investigate. Several failed attempts may "
        "have an ordinary explanation, so look for supporting evidence first.",
        "A detection signal requires investigation before an incident verdict."),
    _step(
        "Keep secrets out of logs", "Choose useful safe evidence.",
        "Record only necessary context and protect access to logs. Passwords "
        "and secret tokens do not belong in event records. An outcome and "
        "timestamp often support login review without exposing a credential.",
        "Which is appropriate login evidence?",
        ("Outcome and timestamp", "The password", "A secret session token"), "A",
        "Use necessary event context without collecting credentials.",
        "A log should explain the event without exposing the key to someone's "
        "account. Keep passwords and secret tokens out of recorded evidence.",
        "Collect necessary context while excluding passwords and other secrets."),
))


# Curated recognition lessons retain the longer lessons' policy: no runnable
# malware, persistence, command shell payloads or exploitation instructions.
_RECOGNITION = (
    LearningPlan("port_scanning", "Recognising port scans", (
        _step("Map listening services", "Understand a port scan.",
              "A port scan asks which network ports answer on a host. An "
              "open port indicates a listening service; it does not by itself "
              "prove that service is vulnerable.",
              "What does an open port establish?",
              ("A service is listening", "An exploit will work", "The service's patch status"), "A",
              "An answering port is a service clue, not a vulnerability verdict.",
              "Think of a port as a service's contact point. An answer tells "
              "you something is listening, not whether it is insecure.",
              "Open ports identify possible services, not proven vulnerabilities."),
        _step("Read the pattern", "Recognise a scanning signal.",
              "Many connection attempts across different destination ports "
              "from one source can indicate scanning. Firewall logs provide "
              "a pattern to review; ordinary administration can also produce it.",
              "Which pattern suggests a scan?",
              ("A single routine HTTPS request", "Many different destination ports", "A scheduled local file check"), "B",
              "Review the connections' source, timing and authorised purpose.",
              "Look for one source trying many service contact points close "
              "together. Treat that pattern as a review signal, with context.",
              "Correlate connection attempts across ports before interpreting them."),
        _reflect("Set the scope first", "Choose authorised practice.",
                 "Use supplied logs or a training VM with explicit scan "
                 "permission. ORION's network tooling still requires an "
                 "approved target scope. This text lesson does not scan anything.",
                 "What permission would you check before scanning?",
                 "Confirm that the target and method are allowed by the "
                 "exercise rules. A lesson about scanning does not authorise a scan.",
                 "Scanning practice requires explicit target and method authorisation."),
    )),
    LearningPlan("reverse_shell", "Recognising unexpected command channels", (
        _step("Understand the connection direction", "Recognise the concept.",
              "A reverse shell is a command shell that connects outward "
              "from the affected machine. The defender's concern is an "
              "unexpected remote channel controlling commands on that machine.",
              "Which direction does a reverse shell connect?",
              ("Inward from the remote controller", "Outward from the affected machine", "Only between local processes"), "B",
              "The affected machine initiates the outbound command channel.",
              "The machine being controlled calls outward. That direction can "
              "make the connection surprising, so defenders review unexpected remote command channels.",
              "A reverse shell initiates an outbound remote command channel."),
        _step("Correlate process and connection", "Read a defensive signal.",
              "An unexpected outbound connection followed by a child command "
              "shell deserves review. Either event alone may be legitimate. "
              "Their relationship, the parent process and its purpose matter.",
              "Which evidence is most useful together?",
              ("Connection timing alone", "Filename alone", "Connection and child-shell history"), "C",
              "Correlate the network event with process ancestry and purpose.",
              "Ask which program opened the connection and whether it needed "
              "to start a command shell. The relationship is the useful clue.",
              "Review unexpected outbound connections alongside their process ancestry."),
        _reflect("Choose a defence", "Explain a protective control.",
                 "Egress filtering limits outgoing destinations. Endpoint "
                 "monitoring can flag unusual child shells. Use a supplied "
                 "process diagram for practice; this plan supplies no command-channel code.",
                 "How could outgoing-connection rules reduce exposure?",
                 "Limit outgoing connections to services a system actually "
                 "needs for its approved purpose. Pair those limits with review of unusual command-shell processes.",
                 "Outgoing-connection controls and process monitoring support defence."),
    )),
    LearningPlan("keylogger", "Recognising keyboard capture", (
        _step("Separate behaviour from intent", "Understand keyboard capture.",
              "A keylogger records keystrokes. Accessibility and shortcut "
              "tools may also receive keyboard events, so keyboard access "
              "alone does not establish malicious intent.",
              "Does keyboard-event access alone prove malware?",
              ("No; purpose and context matter", "Yes, always", "Only if a keyboard API is named"), "A",
              "Legitimate tools also handle keyboard events; examine the purpose.",
              "Some helpful tools need keyboard events. First ask what the "
              "program is meant to do and whether that access is expected.",
              "Keyboard-event access is behaviour to investigate, not proof of intent."),
        _step("Look for a collection chain", "Recognise correlated indicators.",
              "Review unexpected keyboard hooks alongside hidden storage "
              "or outbound transmission of captured input. The combination "
              "is more concerning than a familiar keyboard API name alone.",
              "Which combination deserves closer review?",
              ("An expected shortcut handler", "Unexpected capture plus hidden transmission", "A documented keyboard driver"), "B",
              "Correlate collection, storage and transmission with legitimate purpose.",
              "Follow the information: where are keys collected and where "
              "do they go? Unexpected hidden destinations are a stronger warning sign.",
              "Correlate unexpected keyboard capture with its storage or transmission."),
        _reflect("Practise recognition", "Choose a safe study activity.",
                 "Read supplied descriptions and explain their warning "
                 "signs. Do not run an unfamiliar sample on a daily-use "
                 "machine. This plan teaches recognition and supplies no keyboard-capture code.",
                 "Which clue would you explain to a defender?",
                 "Use a written sample to practise describing the collection "
                 "chain. You can build recognition skill without recording anyone's private keystrokes.",
                 "Learn keyboard-capture threats through recognition and defensive review."),
    )),
    LearningPlan("rootkit", "Recognising hidden system activity", (
        _step("Understand the hidden view", "Recognise a rootkit's purpose.",
              "A rootkit hides activity by tampering with what system tools "
              "can see. A normal-looking process list may therefore be an "
              "incomplete view of the machine.",
              "What is a rootkit's defining concern?",
              ("Only encrypting files", "Only changing file permissions", "Hiding activity through tampering"), "C",
              "The concern is that the system's reported view has been altered.",
              "Imagine a list that silently leaves entries out. A clean-looking "
              "list cannot reassure you when the mechanism producing it is untrusted.",
              "A rootkit can make normal system views omit real activity."),
        _step("Compare independent views", "Recognise a detection clue.",
              "Tools disagreeing about a process, file or connection can "
              "justify an integrity review. Such differences are clues, "
              "not automatic proof; permissions and timing can also cause disagreement.",
              "What should conflicting views prompt?",
              ("An integrity review", "Immediate certainty", "Ignoring both tools"), "A",
              "Investigate the difference using trusted sources and relevant context.",
              "If two views disagree, check why before deciding. A trusted "
              "independent view can help when the usual system view may be altered.",
              "Conflicting system views are a reason to investigate integrity."),
        _reflect("Protect the investigation", "Describe safe recognition practice.",
                 "Study supplied cross-view examples or memory-forensics "
                 "reports. Suspected system tampering needs a trusted "
                 "investigation process. This plan provides no rootkit construction or hiding mechanism.",
                 "What would make an independent view more trustworthy?",
                 "A view is more useful when it does not depend on the "
                 "same possibly altered mechanism. Follow a trusted incident-response procedure.",
                 "Recognition relies on trusted evidence and an integrity-review process."),
    )),
    LearningPlan("ransomware", "Recognising ransomware warning signs", (
        _step("Understand the impact", "Recognise the threat.",
              "Ransomware can deny access to files and demand payment. "
              "Defensive study focuses on noticing unusual changes early "
              "and preserving a path to recovery.",
              "What is the main defensive concern?",
              ("Only leaking account passwords", "Losing access to important files", "Only monitoring network traffic"), "B",
              "Protect availability and prepare recovery from trusted backups.",
              "The immediate harm is losing usable files. Recognition and "
              "recovery planning help defenders respond before the impact spreads across connected systems.",
              "Ransomware defence protects file availability and recovery."),
        _step("Notice changes across folders", "Recognise a behavioural signal.",
              "Rapid unexpected file rewrites across many folders, changed "
              "extensions and payment notes are warning signs. A normal "
              "bulk operation can also change many files; context remains necessary.",
              "Which combination is concerning?",
              ("One expected document update", "A scheduled backup alone", "Mass rewrites plus payment notes"), "C",
              "That combination warrants urgent incident review under the response plan.",
              "Watch for many unusual changes happening together. New payment "
              "notes alongside widespread rewrites make the pattern especially concerning and worth immediate review.",
              "Correlated mass changes and payment notes warrant urgent review."),
        _reflect("Plan recoverable backups", "Name a recovery control.",
                 "Keep protected backups separated from ordinary writable "
                 "access and test restoration. Follow an incident-response "
                 "plan for suspected infection. This recognition lesson contains no file-encryption code.",
                 "How would you check that a backup is usable?",
                 "A backup helps only if it survives the incident and "
                 "can be restored. Practise restoration using a designated safe recovery exercise.",
                 "Protected backups need tested restoration to support recovery."),
    )),
    LearningPlan("privilege_escalation", "Recognising excessive permissions", (
        _step("Understand a privilege change", "Recognise escalation.",
              "Privilege escalation turns limited access into greater "
              "access. A defender asks which permission boundary was crossed "
              "and whether that change was authorised.",
              "What changes during privilege escalation?",
              ("The level of access", "Only the login username", "Only the number of open ports"), "A",
              "The security concern is gaining permissions beyond the original access.",
              "A limited account should stay limited unless an authorised "
              "process grants more access. Review unexpected changes in what it can do.",
              "Privilege escalation crosses a boundary into greater access."),
        _step("Spot a writable privileged task", "Recognise a misconfiguration.",
              "A task running with high privileges should not execute "
              "a script writable by an ordinary user. That mismatch is "
              "a permissions risk to review, without attempting to exploit it.",
              "Which setup needs review?",
              ("A task using owner-controlled scripts", "A privileged task using an ordinary-user-writable script", "A service with reviewed least privilege"), "B",
              "Review the script's permissions and the task's required privileges.",
              "If a powerful task trusts a file that anyone can change, "
              "its trust boundary is weak. Check ownership and write permissions.",
              "Privileged tasks must not trust files writable by unauthorised users."),
        _reflect("Apply least privilege", "Explain a defensive improvement.",
                 "Give accounts and services only the access they need. "
                 "Review file permissions and patching within the owner's "
                 "approved process. Use supplied permission diagrams for this practice.",
                 "Which permission could you reduce in a practice diagram?",
                 "Start with one service and list its necessary access. "
                 "Any broader permission should have a clear, authorised reason to exist.",
                 "Least privilege limits the reach of mistakes and compromise."),
    )),
    LearningPlan("phishing", "Recognising deceptive messages", (
        _step("Notice pressure", "Recognise a message warning sign.",
              "Phishing impersonates a trusted source to persuade someone "
              "to reveal information or take an unsafe action. Urgency "
              "combined with a credential request is a warning sign to verify.",
              "Which request needs independent verification?",
              ("A routine notice without credential requests", "An expected receipt from a verified sender", "An urgent request for your password"), "C",
              "Verify through a known contact route before taking the requested action.",
              "A message can sound confident and still be deceptive. "
              "Pause when it pressures you to disclose a password or act immediately.",
              "Pressure to disclose credentials is a reason to verify independently."),
        _step("Check the actual destination", "Recognise a misleading link.",
              "A link's visible words may differ from its actual "
              "destination. Review the full domain, not just a familiar "
              "brand name in the text. A lookalike domain is a warning sign.",
              "What identifies where a link goes?",
              ("Its actual destination", "Its reassuring label", "The sender's claimed identity"), "A",
              "The destination matters; a trustworthy-looking label proves nothing.",
              "The words on a button can say anything. The actual "
              "destination tells you which site would receive the visit or information.",
              "Check the actual destination rather than trusting the link label."),
        _reflect("Choose a known route", "Practise verification safely.",
                 "Use a saved official address or a known contact "
                 "method to verify a suspicious request. Report it through "
                 "the organisation's process. Practise on supplied descriptions without opening suspicious links.",
                 "Which known contact route would you use to verify?",
                 "Reach the organisation using information you already trust. "
                 "Do not rely on a phone number or reply address supplied by the suspicious message.",
                 "Verify suspicious requests through an independently trusted contact route."),
    )),
)


_ELECTRONICS = LearningPlan("electronics_fundamentals", "Electronics through small simulations", (
    _step("Understand voltage", "Recognise voltage's role.",
          "Voltage is energy per unit charge between two points. It is "
          "measured in volts, written V. Start with this quantity before "
          "calculating current in a resistor; this plan uses arithmetic only.",
          "Which unit measures voltage?", ("Volts", "Ohms", "Amperes"), "A",
          "Volts measure voltage; ohms measure resistance and amperes measure current.",
          "Voltage compares two points and describes energy per charge. "
          "For this step, remember its name and unit; no physical circuit is needed.",
          "Voltage is measured in volts between two points."),
    _step("Calculate resistor current", "Read Ohm's law in Python.",
          "For an ideal resistor, current equals voltage divided by "
          "resistance. These are fictional simulation values, not hardware "
          "ratings or build instructions. ORION displays the arithmetic without running it.\n\n"
          '```python\nresistor_voltage_v = 1.3\nresistance_ohm = 330.0\n'
          'current_a = resistor_voltage_v / resistance_ohm\n'
          'print(f"{current_a * 1000:.2f} mA")\n```',
          "What current does this model print?", ("330 mA", "3.94 mA", "1.3 A"), "B",
          "1.3 / 330 is about 0.00394 A, or 3.94 mA.",
          "In this fictional model, divide 1.3 volts by 330 ohms: about "
          "0.00394 amperes. Multiply by one thousand to express that as 3.94 milliamperes.",
          "An ideal resistor's current is its voltage divided by its resistance."),
    _step("Calculate resistor power", "Read a power calculation.",
          "An ideal resistor's power is current squared times resistance. "
          "Power measures energy transfer per second. The model ignores "
          "temperature and real component limits, so it cannot establish physical safety.\n\n"
          '```python\ncurrent_a = 1.3 / 330.0\npower_w = current_a ** 2 * 330.0\n'
          'print(f"{power_w * 1000:.2f} mW")\n```',
          "What does the model print?", ("330 W", "3.94 A", "5.12 mW"), "C",
          "Squaring the current and multiplying by 330 gives about 0.00512 W.",
          "In this fictional model, current is 1.3 / 330 amperes. Square "
          "it and multiply by 330 ohms: about 0.00512 watts, or 5.12 milliwatts.",
          "Resistor power in this model equals current squared times resistance."),
))


_HARDWARE = LearningPlan("hardware_protection", "Protecting electronics in small steps", (
    _step("Recognise static discharge", "Understand a handling risk.",
          "Electrostatic discharge is a sudden transfer of accumulated "
          "charge. Sensitive electronics need the manufacturer's handling "
          "procedure, suitable packaging and an appropriate work area. "
          "Do not deliberately create discharge to test a device.",
          "What guides handling sensitive electronics?",
          ("The manufacturer's procedure", "A general software tutorial", "A visual check alone"), "A",
          "Follow the device-specific handling procedure and approved work setup.",
          "Static charge can move suddenly and harm a sensitive part. "
          "Use the maker's handling rules instead of trying an improvised discharge test.",
          "ESD prevention starts with approved handling and work-area controls."),
    _step("Respect operating limits", "Distinguish normal operation from stress.",
          "Electrical input outside a device's ratings can damage it. "
          "An absolute maximum rating is a stress limit, not a normal "
          "operating target. A qualified engineer should review device-specific power protection.",
          "Is an absolute maximum a normal operating target?",
          ("Yes, if used briefly", "No", "Yes, if the device is new"), "B",
          "Use specified operating conditions; absolute maximum values are stress limits.",
          "A limit marking where damage may occur is not a recommended "
          "setting. Normal operation must stay within the maker's specified operating conditions.",
          "Normal operating conditions differ from absolute maximum stress limits."),
    _step("Keep unknown peripherals disconnected", "Choose a safe review action.",
          "An unknown USB device may present data and electrical risks. "
          "Antivirus cannot establish electrical safety. Keep suspicious "
          "equipment disconnected and refer it through the approved specialist-assessment process.",
          "What should you do with an unknown found USB device?",
          ("Scan it on the main PC", "Use it if antivirus is clean", "Keep it disconnected and report it"), "C",
          "Use approved peripherals and specialist assessment of unknown equipment.",
          "A USB device's appearance does not establish its safety. "
          "Keep unknown equipment disconnected so investigating it does not expose a valuable machine.",
          "Unknown peripherals need approved assessment before any connection."),
))


_FIRMWARE = LearningPlan("firmware_security", "Firmware controls in small steps", (
    _step("Authenticate an update", "Understand signed firmware.",
          "A device should verify an update's signature using its "
          "protected trust anchor and authorisation policy. A checksum "
          "may detect corruption, but does not establish who published the update.",
          "Does a checksum alone authenticate the publisher?",
          ("Yes", "No", "Only when its hash is long"), "B",
          "Publisher authentication needs a trusted signature-verification process.",
          "A checksum checks whether data changed. A verified signature "
          "ties the image to a trusted signing key under the device's update policy.",
          "Authenticated updates need trusted signature verification and an authorisation policy."),
    _step("Control debug access", "Recognise a sensitive interface.",
          "Debug interfaces can expose device memory and control. "
          "Review supported authentication and lifecycle controls with "
          "the owner before changes. Some settings erase data or are irreversible; "
          "this lesson changes no live device.",
          "What should precede a live debug-policy change?",
          ("Disabling protections immediately", "Trying a standard unlock command", "Owner-approved vendor review"), "C",
          "Use the owner's approved process and the vendor's device-specific procedure.",
          "Debug access is powerful access to a device. Check the "
          "vendor's procedure with its owner before changing settings that may have permanent effects.",
          "Debug controls require an owner-approved, device-specific change process."),
    _step("Reject forbidden rollback", "Apply a minimum-version rule.",
          "A valid signature does not make an old vulnerable update "
          "acceptable. A rollback policy can require a protected minimum "
          "security version. Practise this decision with mock metadata, keeping recovery authorised.",
          "Accept a signed image below the permitted minimum version?",
          ("No; reject under the policy", "Yes; signatures override policy", "Yes, if the filename says latest"), "A",
          "Both signature verification and the minimum-security-version policy must pass.",
          "A trusted publisher may have signed an older unsafe release. "
          "The device must still apply its minimum-version rule before accepting that image.",
          "Update acceptance must satisfy both authenticity and rollback policy."),
))


_PYTHON = LearningPlan("python_basics", "Your first small Python steps", (
    _step("Name a value", "Read a variable assignment.",
          "A variable gives a value a name. The equals sign here "
          "assigns the value; print displays it. This supplied example "
          "uses only an in-memory number, and ORION does not run it.\n\n"
          '```python\nminutes = 2\nprint(minutes)\n```',
          "What does print(minutes) display?", ("2", "minutes", "An error"), "A",
          "minutes names the number 2, so print displays 2.",
          "Read minutes = 2 as give the number two the name minutes. "
          "When print uses that name, it displays the stored number.",
          "Assignment gives a value a name that later code can use."),
    _step("Choose with if", "Read one condition.",
          "An if block runs when its condition is true. Indented "
          "lines belong to the block. Here, ready is true, so "
          "the indented print line runs in this supplied example.\n\n"
          '```python\nready = True\nif ready:\n    print("One small step")\n```',
          "What would happen if ready were False?",
          ("The message repeats", "The message is not printed", "Python changes ready"), "B",
          "A false condition skips this if block.",
          "The if line is a yes-or-no gate. True enters the "
          "indented block; False skips it. Indentation shows which instructions belong together.",
          "An if condition controls whether its indented block runs."),
    _step("Repeat with for", "Read a small loop.",
          "A for loop visits each item in a collection. The "
          "name word receives one item at a time. The indented "
          "print repeats for those items; it does not repeat forever.\n\n"
          '```python\nfor word in ("read", "pause"):\n    print(word)\n```',
          "Which order is printed?", ("pause then read", "read forever", "read then pause"), "C",
          "The loop visits the tuple's two items in order.",
          "The tray holds read first and pause second. The loop takes "
          "one item at a time, prints it, then stops after those two items.",
          "A for loop performs its block once for each supplied item."),
    _step("Reuse with a function", "Read a function call.",
          "A function names a reusable piece of code. return gives "
          "its result back to the caller. This function adds one "
          "to its input, then print displays the returned result.\n\n"
          '```python\ndef next_step(number):\n    return number + 1\nprint(next_step(2))\n```',
          "What result is printed?", ("3", "2", "next_step"), "A",
          "The call supplies 2; the function returns 2 + 1, which is 3.",
          "A function is a named recipe. Here the input is 2 and "
          "the recipe adds 1, so it returns 3 for print to display.",
          "A function call supplies input and receives the returned result."),
))


_PLANS = (_PYTHON, _SQL, _XSS, _ACCESS, _LOGS, *_RECOGNITION,
          _ELECTRONICS, _HARDWARE, _FIRMWARE)
_BY_TOPIC = {plan.topic: plan for plan in _PLANS}


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("_", " ")
                  .replace("-", " ")).strip()


_ALIASES = {_normalise(plan.topic): plan.topic for plan in _PLANS}
for _topic, _lesson in cyber_teaching.LESSONS.items():
    for _alias in _lesson.aliases:
        _ALIASES[_normalise(_alias)] = _topic
for _exercise in cyber_exercises.EXERCISES:
    for _alias in _exercise.aliases:
        _ALIASES[_normalise(_alias)] = _exercise.topic
for _topic, _lesson in electronics_teaching.LESSONS.items():
    for _alias in _lesson.aliases:
        _ALIASES[_normalise(_alias)] = _topic
for _alias in ("python", "python basics", "beginner python", "python fundamentals",
               "programming basics", "programming foundations", "learn to code",
               "coding basics", "python variables"):
    _ALIASES[_alias] = "python_basics"


def plans() -> tuple[LearningPlan, ...]:
    """Return immutable reviewed plans, without accessing learner state."""
    return _PLANS


def resolve_plan(topic: str) -> LearningPlan | None:
    """Select a reviewed topic or alias; do not guess from arbitrary prose.

    The caller still applies construction and authorisation gates. Declining
    direct destructive construction here also prevents accidental topic routing
    from replacing that boundary with an ordinary lesson.
    """
    if not isinstance(topic, str) or not topic.strip():
        return None
    if electronics_teaching.construction_boundary(topic):
        return None
    query = _normalise(topic)
    for prefix in ("start a lesson on ", "start lesson on ", "teach me about ",
                   "teach me ", "teach ", "show code for ", "code lesson on ",
                   "code lesson for ", "lesson on ", "lesson for ",
                   "exercise on ", "exercise for ", "learn about ", "learn "):
        if query.startswith(prefix):
            query = query[len(prefix):]
            break
    resolved = _ALIASES.get(query)
    return _BY_TOPIC.get(resolved) if resolved else None


__all__ = ["LearningStep", "LearningPlan", "plans", "resolve_plan"]
