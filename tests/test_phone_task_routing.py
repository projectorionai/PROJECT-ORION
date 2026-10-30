"""Phone commands retain their device, exact content and native receipt path."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from orion_core.remote_tasks import is_action_request, phone_plan, plan_request, requested_target


def _args(message):
    steps = phone_plan(message)
    assert len(steps) == 1
    assert steps[0]["tool"] == "phone_device"
    return steps[0]["args"]


@pytest.mark.parametrize("message, target", [
    ('Open Notepad and write "Hello"', "phone"),
    ('Open Notepad on my PC and write "Hello"', "desktop"),
    ('Please email Jordan subject "PC access" body "Use my desktop tomorrow"', "phone"),
    ('Email Jordan subject PC access body Use my desktop tomorrow', "phone"),
    ('Email Jordan subject "Meeting" body "Hello" on my PC', "desktop"),
    ('Share Use my PC tomorrow', "phone"),
    ('Open "Remote Desktop"', "phone"),
    ('Open Remote Desktop', "phone"),
    ('Call "PC"', "phone"),
    ('Open Settings on my phone, not on my PC', "phone"),
    ('Open Notepad on the PC, not on my phone', "desktop"),
    ('Check PC telemetry', "desktop"),
    ('PC: open Notepad', "desktop"),
])
def test_phone_is_the_default_and_payloads_do_not_change_target(message, target):
    assert requested_target(message) == target


def test_ambiguous_multiple_devices_are_not_silently_chosen():
    with pytest.raises(ValueError, match="choose one device"):
        requested_target("Open the mail app on my phone and on my PC")


@pytest.mark.parametrize("message", ["Email Jordan", "Text Jordan", "SMS +447700900123", "Draft an email", "PC: open Notepad"])
def test_outbound_and_explicit_device_verbs_enter_the_task_path(message):
    assert is_action_request(message)


def test_e_mail_send_and_draft_are_different_exact_actions():
    body = 'Hi Jordan,\nCan we meet at 15:30?\nThanks, Sam.'
    request = f'Please email Jordan subject "Meeting" body "{body}"'
    assert _args(request) == {"action": "send_email", "to": "Jordan", "subject": "Meeting", "body": body}
    assert _args('Compose an email to jane@example.test with subject "Meeting" and body "Hello"') == {
        "action": "compose_email", "to": "jane@example.test", "subject": "Meeting", "body": "Hello",
    }
    assert _args('Draft an email to Jane subject “Checking in” saying “Hello, Jane” on my phone')["action"] == "compose_email"


@pytest.mark.parametrize("message, question", [
    ("Email someone", "exact contact name"),
    ("Email Jordan", "subject and message text"),
    ('Email Jordan subject "Meeting"', "message text"),
    ('Email Jordan saying "Hello"', "exact subject"),
    ("Text Jordan", "exact text"),
    ("Call", "exact contact name or phone number"),
    ("Navigate to", "exact address or place"),
    ("Share", "exact text"),
    ("Open an app", "exact name"),
    ("Set an alarm for tomorrow morning", "exact alarm time"),
    ("Set a timer for a while", "exact number"),
])
def test_missing_details_are_requested_before_any_plan_can_run(message, question):
    with pytest.raises(ValueError, match=question):
        phone_plan(message)


def test_sms_retains_every_word_and_does_not_remove_a_payload_destination():
    assert _args('Text Jordan saying "I am on my phone"') == {
        "action": "send_sms", "number": "Jordan", "body": "I am on my phone",
    }
    assert _args("Text Jordan saying I am on my phone")["body"] == "I am on my phone"
    assert _args('Send a text to +447700900123: "I am on my way"') == {
        "action": "send_sms", "number": "+447700900123", "body": "I am on my way",
    }


def test_exact_app_names_are_resolved_on_device_instead_of_guessed_packages():
    assert _args("Open Gmail") == {"action": "open_app", "name": "Gmail"}
    assert _args("Open My University app") == {"action": "open_app", "name": "My University"}
    assert _args("Open com.example.my_app") == {"action": "open_app", "package": "com.example.my_app"}
    assert _args("Open Wi-Fi settings") == {"action": "open_settings", "screen": "wifi"}


def test_navigation_maps_links_and_share_keep_the_supplied_destination():
    assert _args('Navigate to "Birmingham New Street"') == {"action": "navigate", "query": "Birmingham New Street"}
    assert _args('Open a map of "Kings Cross Station"') == {"action": "map", "query": "Kings Cross Station"}
    assert _args("Open https://example.test/a?q=1") == {"action": "open_url", "url": "https://example.test/a?q=1"}
    assert _args('Share "A note from my phone"') == {"action": "share", "text": "A note from my phone"}


def test_alarm_timer_contacts_and_navigation_buttons_use_exact_native_functions():
    assert _args("Set an alarm for 12:15 am") == {"action": "set_alarm", "hour": 0, "minutes": 15}
    assert _args("Set an alarm for 7:30 pm") == {"action": "set_alarm", "hour": 19, "minutes": 30}
    assert _args("Set a timer for 3 minutes") == {"action": "set_timer", "seconds": 180}
    assert _args("Find contact Jordan") == {"action": "find_contact", "name": "Jordan"}
    assert _args("List installed apps") == {"action": "list_apps"}
    assert _args("Go back") == {"action": "back"}
    assert _args("Go to the home screen") == {"action": "home"}
    with pytest.raises(ValueError, match="valid alarm time"):
        phone_plan("Set an alarm for 27:00")


def _gate():
    from orion_core.dispatch_schema import PHONE_DEVICE_DECLARATION
    return SimpleNamespace(TOOL_DECLARATIONS=[
        PHONE_DEVICE_DECLARATION,
        {"name": "open_app", "parameters": {"type": "OBJECT"}},
        {"name": "desktop_control", "parameters": {"type": "OBJECT"}},
        {"name": "outlook_mail", "parameters": {"type": "OBJECT"}},
        {"name": "phone_action", "parameters": {"type": "OBJECT"}},
        {"name": "web_search", "parameters": {"type": "OBJECT"}},
        {"name": "mcp", "parameters": {"type": "OBJECT"}},
    ])


class _Planner:
    def __init__(self, steps):
        self.steps = steps
        self.calls = []

    async def generate_text(self, message, *, system_extra, complete_reply=True):
        self.calls.append((message, system_extra, complete_reply))
        return None, json.dumps({"steps": self.steps})


def test_exact_native_and_explicit_notepad_instructions_need_no_model():
    async def run():
        phone = await plan_request(None, _gate(), 'Email Jordan subject "Meeting" body "Hello"', "")
        assert phone[0]["tool"] == "phone_device"
        pc = await plan_request(None, _gate(), 'Open Notepad on my PC and write "Hello"', "")
        assert [step["tool"] for step in pc] == ["open_app", "desktop_control"]
        desktop = await plan_request(None, _gate(), 'Open Notepad and write "Hello"', "", target="desktop")
        assert [step["tool"] for step in desktop] == ["open_app", "desktop_control"]
    asyncio.run(run())


def test_complex_phone_plans_do_not_expose_desktop_mail_or_unreviewed_intents():
    planner = _Planner([{"tool": "phone_device", "args": {"action": "capture_dom"}}])
    steps = asyncio.run(plan_request(planner, _gate(), "Read the Gmail inbox on my phone", ""))
    assert steps[0]["tool"] == "phone_device"
    _, system, complete_reply = planner.calls[0]
    tools = json.loads(system.split("TOOLS:\n", 1)[1])
    assert [tool["tool"] for tool in tools] == ["phone_device"]
    assert complete_reply is False


@pytest.mark.parametrize("step, message, question", [
    ({"tool": "outlook_mail", "args": {"action": "priority"}}, "Read my e-mail inbox", "outside the requested device"),
    ({"tool": "desktop_control", "args": {"action": "click"}}, "Tap the Send button", "outside the requested device"),
    ({"tool": "phone_action", "args": {"kind": "email"}}, "Read my e-mail inbox", "outside the requested device"),
    ({"tool": "phone_device", "args": {"action": "open_app", "package": "com.guessed.mail"}}, "Open the university portal and then read the inbox", "exact package"),
    ({"tool": "phone_device", "args": {"action": "send_email", "to": "invented@example.test", "subject": "Hi", "body": "Hello"}}, "Email Jordan and then open the inbox", "exact to"),
    ({"tool": "phone_device", "args": {"action": "click", "snapshotId": "invented", "nodeId": "0/1"}}, "Tap Send", "read your phone screen first"),
    ({"tool": "phone_device", "args": {"action": "make_call", "number": "Jordan"}}, "Find contact Jordan", "not part of your instruction"),
    ({"tool": "mcp", "args": {"action": "call", "server": "filesystem", "tool": "write_file"}}, "List my MCP services", "inspect MCP services"),
])
def test_model_cannot_switch_devices_invent_targets_or_add_actions(step, message, question):
    planner = _Planner([step])
    # A complex form reaches the model instead of the deterministic fast path.
    message += " and then check the result"
    with pytest.raises(ValueError, match=question):
        asyncio.run(plan_request(planner, _gate(), message, ""))


def test_multi_action_requests_are_not_swallowed_into_a_single_native_payload():
    assert phone_plan('Email Jordan subject "Meeting" body "Hello" and then open Gmail') is None
