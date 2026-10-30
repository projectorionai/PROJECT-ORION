"""Offline electronics and firmware lessons with a safe, numerical exercise.

This service only returns curated text. It performs no network, hardware,
file, or code-execution actions. Intent matching selects lessons and a
construction boundary; it is not an authorisation gate for other tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


LED_SIMULATION_CODE = '''def simulate_led(supply_v, led_drop_v, resistance_ohm):
    if resistance_ohm <= 0:
        raise ValueError("Resistance must be positive")
    resistor_voltage_v = max(supply_v - led_drop_v, 0.0)
    current_a = resistor_voltage_v / resistance_ohm
    resistor_power_w = current_a ** 2 * resistance_ohm
    return current_a, resistor_power_w

current_a, power_w = simulate_led(3.3, 2.0, 330.0)
print(f"LED current: {current_a * 1000:.2f} mA")
print(f"Resistor power: {power_w * 1000:.2f} mW")'''


@dataclass(frozen=True)
class Lesson:
    topic: str
    title: str
    category: str
    demo_policy: str
    body: str
    aliases: tuple[str, ...]

    def format(self) -> str:
        return f"# {self.title}\n\n{self.body}"


LESSONS: dict[str, Lesson] = {
    "electronics_fundamentals": Lesson(
        topic="electronics_fundamentals",
        title="Electronics fundamentals: voltage, current, resistance and power",
        category="electronics",
        demo_policy="simulation_only",
        body=(
            "Voltage (V) is energy per unit charge between two points. Current (A) "
            "is the rate of charge flow. Resistance (ohms) relates voltage to current "
            "in an ohmic resistor: V = I * R. Power (W) is energy transferred per "
            "second: P = V * I; in a resistor, P = I ** 2 * R.\n\n"
            "For an LED and resistor in a simplified series circuit, the resistor "
            "takes the supply voltage minus the LED's forward drop. The LED itself "
            "does not obey a constant resistance model. Current limiting matters. "
            "[Arduino's LED explanation](https://docs.arduino.cc/built-in-examples/basics/Blink/) "
            "explains the voltage subtraction and Ohm's law.\n\n"
            "## Python exercise: a simulated 3.3 V LED\n\n"
            "Run the following with Python 3. It uses only arithmetic and prints "
            "results; it needs no libraries or connected hardware. ORION displays "
            "this example and does not execute it.\n\n"
            f"```python\n{LED_SIMULATION_CODE}\n```\n\n"
            "The `def` line names a function and its three inputs. The `if` and "
            "`raise` lines reject zero or negative resistance before division. "
            "`max` prevents a negative forward current in this elementary model. "
            "Dividing the resistor's voltage by resistance gives amperes. Squaring "
            "current and multiplying by resistance gives resistor power in watts. "
            "`return` supplies both results. The call provides simulated values; "
            "the final two lines convert A to mA and W to mW and print two decimals.\n\n"
            "Expected output:\n\n"
            "```text\nLED current: 3.94 mA\nResistor power: 5.12 mW\n```\n\n"
            "Exercise: predict the result when `330.0` becomes `660.0`, then run "
            "the changed example. Expected: 1.97 mA and 2.56 mW. Explain why "
            "doubling resistance halves current and resistor power here.\n\n"
            "These numbers are simulation assumptions, not component ratings or "
            "a hardware build specification. The model ignores temperature, "
            "tolerances, source limits and the LED's nonlinear response. It is "
            "useful for learning units and Python, not certifying physical safety."
        ),
        aliases=(
            "electronics fundamentals", "electronics basics", "basic electronics",
            "electronics", "voltage", "current", "resistance", "resistor",
            "ohm's law", "ohms law", "led", "led simulation", "low voltage",
            "electrical power",
        ),
    ),
    "hardware_protection": Lesson(
        topic="hardware_protection",
        title="Hardware protection: ESD, input power and untrusted peripherals",
        category="hardware-security",
        demo_policy="concept_only",
        body=(
            "Electrostatic discharge (ESD) is a sudden transfer of accumulated "
            "charge. Sensitive electronics need handling controls, protective "
            "packaging and an appropriate work area. Follow the device maker's "
            "handling procedure. [Texas Instruments' ESD handling guidance]"
            "(https://www.ti.com/lit/an/szza047/szza047.pdf) describes this risk.\n\n"
            "Abnormal input power is a separate threat: electrical stress beyond "
            "a device's ratings can cause damage. At a defensive review, identify "
            "exposed connectors, document permitted supplies and require a qualified "
            "engineer to assess protection and recovery. Do not test by delivering "
            "damaging pulses. Device-specific operating limits matter; an absolute "
            "maximum rating is not a normal operating target. [Microchip's stress "
            "rating explanation](https://onlinedocs.microchip.com/oxy/GUID-22527069-B4D6-49B9-BACC-3AF1C52EB48C-en-US-21/GUID-7208A641-CF0B-44CA-893E-010595DD2494.html) "
            "makes that distinction.\n\n"
            "USB risks also include malicious files and peripherals posing as "
            "unexpected device classes. Use approved peripherals, physical access "
            "controls and device-control policy. A clean malware scan does not "
            "establish electrical safety. Do not plug an unknown device into a "
            "valuable machine to investigate it. Preserve and report suspicious "
            "equipment for specialist assessment. [NIST SP 1800-26]"
            "(https://www.nccoe.nist.gov/publication/1800-26/VolB/index.html) "
            "discusses USB threats and restrictions on allowed devices.\n\n"
            "## Review exercise\n\n"
            "A found USB device is described as a charger. List one control for "
            "data threats and one for electrical threats before it is connected. "
            "Expected: enforce approved-device policy; keep the unknown device "
            "disconnected and refer electrical assessment to a qualified engineer. "
            "Explain why antivirus alone cannot address both threats.\n\n"
            "This lesson teaches protection and recognition. It supplies no "
            "construction instructions, component choices, pulse specifications or "
            "wiring for destructive hardware or shock devices."
        ),
        aliases=(
            "hardware protection", "hardware security", "esd", "electrostatic discharge",
            "abnormal input power", "input power protection", "surge protection",
            "usb security", "usb device risks", "usb killer", "usb killers",
            "usbkill", "usbkiller", "taser", "tasers", "stun gun", "stun guns",
            "destructive hardware", "destructive pulses", "hardware damage",
            "electronics taser", "electronic taser",
        ),
    ),
    "firmware_security": Lesson(
        topic="firmware_security",
        title="Firmware security: signed updates, debug access and rollback",
        category="firmware-security",
        demo_policy="concept_only",
        body=(
            "Firmware is code that helps initialise and operate a device. A "
            "firmware security review asks how changes are authorised, how "
            "tampering is detected and how a trusted state can be recovered.\n\n"
            "A signed update binds an image to a signing key. The device must "
            "verify the signature using a protected trust anchor before accepting "
            "the update, and apply its authorisation policy. A checksum alone "
            "detects some corruption but does not authenticate the publisher. "
            "Plan recovery and signing-key compromise handling. [NIST SP 800-193]"
            "(https://csrc.nist.gov/pubs/sp/800/193/final) explains authenticated "
            "updates and the protect, detect, recover approach.\n\n"
            "Debug interfaces may expose code, memory and device control. "
            "Production policy should restrict debug access through supported "
            "device lifecycle controls or authenticated access. Review the "
            "vendor's procedure with the device owner before changing any setting; "
            "some transitions erase data or are irreversible. [STMicroelectronics' "
            "debug authentication documentation]"
            "(https://wiki.st.com/stm32mcu/wiki/Security:Debug_Authentication) "
            "illustrates vendor-specific controls.\n\n"
            "Rollback policy decides which earlier versions may be installed. "
            "A valid signature alone does not make an old, vulnerable version "
            "acceptable. Track a protected minimum security version and plan "
            "authorised recovery. [MCUboot's rollback protection design]"
            "(https://docs.mcuboot.com/design.html#hardware-based-downgrade-prevention) "
            "describes software and hardware approaches.\n\n"
            "## Review exercise\n\n"
            "An update has a valid signature, but its security version is below "
            "the permitted minimum. Should it be accepted? Expected: reject it "
            "under the rollback policy. Review signatures, update authorisation, "
            "debug access, rollback rules and recovery as separate controls. "
            "Practise with a paper checklist or mock metadata, without changing "
            "a live device or disabling its protections."
        ),
        aliases=(
            "firmware security", "firmware", "signed updates", "signed firmware",
            "secure boot", "debug access", "debug authentication", "jtag", "swd",
            "rollback", "anti rollback", "antirollback", "trust anchor",
        ),
    ),
}


_DANGEROUS_DEVICE = re.compile(
    r"\b(?:tasers?|stun\s*guns?|shock\s+(?:devices?|weapons?)|usb\s*(?:killers?|kill)|"
    r"destructive\s+(?:hardware|devices?|pulses?)|"
    r"(?:electronic|electronics|hardware)\s+zappers?|emp\s+guns?)\b"
)
_ACTIONABLE = re.compile(
    r"\b(?:build|building|make|making|construct|construction|assemble|assembly|"
    r"wire|wiring|design|program|programming|code|schematics?|circuits?|"
    r"components?|parts|specifications?|specs|specifics|voltages?|"
    r"pulse\s+parameters?|instructions?|step\s+by\s+step)\b"
)
_DEFENSIVE = re.compile(
    r"\b(?:protect(?:ion|ive|ing)?|defen[cs]e|defen[cs]ive|defend(?:ing)?|"
    r"mitigat(?:e|ion|ing)|prevent(?:ion|ing)?|detect(?:ion|ing)?|"
    r"recognis(?:e|ing)|recogniz(?:e|ing))\b"
)
_DAMAGE_INTENT = re.compile(
    r"\b(?:destroy|fry|damage|disable|kill)\b.{0,45}"
    r"\b(?:electronics|hardware|computers?|usb\s+ports?|devices?)\b"
)
_BOUNDARY = (
    "I can't provide construction instructions, wiring, components, electrical "
    "specifications or code for tasers, stun guns, USB killers or devices intended "
    "to damage electronics. I can teach electronics fundamentals with a "
    "Python-only LED simulation, hardware protection, and firmware security."
)


def _normalise(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").lower().replace("_", " ")
                  .replace("-", " ")).strip()


def construction_boundary(query: str) -> str | None:
    """Decline actionable shock/destructive construction, permit defence study.

    Defensive phrasing is not a blanket exception: ``for defence, build a USB
    killer`` still requests construction. ``Build protection against a USB
    killer`` requests the protection instead and may receive the defence lesson.
    Every return path elsewhere remains curated, harmless text.
    """
    text = _normalise(query)
    threats = list(_DANGEROUS_DEVICE.finditer(text))
    if not threats and not _DAMAGE_INTENT.search(text):
        return None
    actions = list(_ACTIONABLE.finditer(text))
    if not actions:
        return None
    for threat in threats:
        # Use the closest preceding defence/action term to determine its object.
        # A later direct construction verb overrides an earlier defence preface.
        prefix = text[max(0, threat.start() - 100):threat.start()]
        preceding = list(_ACTIONABLE.finditer(prefix))
        defending = list(_DEFENSIVE.finditer(prefix))
        if defending and (not preceding or defending[-1].start() > preceding[-1].start()):
            # A follow-up "build one/it" still asks to construct the named threat.
            suffix = text[threat.end():]
            if re.search(r"\b(?:build|make|wire|construct|assemble|program)\s+(?:one|it)\b", suffix):
                return _BOUNDARY
            # A defence preface must not permit subsequent requests for the
            # named device's circuit, wiring or specifications.
            refers_to_device = re.search(
                r"\b(?:its|one)\b|\b(?:for|of)\s+(?:it|(?:that|this|the)\s+(?:device|hardware))\b",
                suffix,
            )
            if refers_to_device and _ACTIONABLE.search(suffix):
                return _BOUNDARY
            continue
        return _BOUNDARY
    if not threats and _DAMAGE_INTENT.search(text):
        damaging = _DAMAGE_INTENT.search(text)
        prefix = text[:damaging.start()]
        defending = list(_DEFENSIVE.finditer(prefix))
        preceding = list(_ACTIONABLE.finditer(prefix))
        if defending and (not preceding or defending[-1].start() > preceding[-1].start()):
            return None
        return _BOUNDARY
    return None


def resolve_topic(query: str) -> str | None:
    """Resolve known electronics topics, including natural defensive requests."""
    text = _normalise(query)
    for topic in LESSONS:
        if text == topic.replace("_", " "):
            return topic
    # Prefer firmware/protection before general "electronics" or "current".
    for topic in ("firmware_security", "hardware_protection", "electronics_fundamentals"):
        aliases = sorted(LESSONS[topic].aliases, key=len, reverse=True)
        for alias in aliases:
            normal_alias = _normalise(alias)
            # These words also occur in ordinary security questions ("current
            # threats", database "rollback"). A bare concept is a lesson;
            # otherwise require electronics context before redirecting it.
            if normal_alias in {"current", "resistance", "rollback"} and text != normal_alias:
                if not re.search(r"\b(?:electronics?|electrical|circuit|firmware|voltage|ohms?|led)\b", text):
                    continue
            if re.search(r"(?<!\w)" + re.escape(normal_alias) + r"(?!\w)", text):
                return topic
    if _DANGEROUS_DEVICE.search(text) or _DAMAGE_INTENT.search(text):
        return "hardware_protection"
    return None


def catalogue() -> list[dict[str, str]]:
    """Return fresh serialisable lesson metadata without accessing devices."""
    return [
        {"topic": lesson.topic, "title": lesson.title, "category": lesson.category,
         "demo_policy": lesson.demo_policy}
        for lesson in LESSONS.values()
    ]


def teach(query: str) -> str:
    """Return a safe lesson or construction boundary and a defensive alternative."""
    boundary = construction_boundary(query)
    if boundary:
        return boundary + "\n\n" + LESSONS["hardware_protection"].format()
    topic = resolve_topic(query)
    if topic:
        return LESSONS[topic].format()
    choices = ", ".join(lesson.title for lesson in LESSONS.values())
    return "Electronics lessons available: " + choices + ". Ask me to teach one."


__all__ = [
    "LED_SIMULATION_CODE", "LESSONS", "Lesson", "catalogue",
    "construction_boundary", "resolve_topic", "teach",
]
