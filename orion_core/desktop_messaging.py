"""Guarded, observed messaging through already signed-in Windows applications.

No private service APIs, account tokens, recipient guesses or blind Enter sends.
UI changes that cannot be read back stop the operation with an actionable result.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any

from .data import ToolResult
from .execution_policy import foreground_authorized, is_background, require_foreground


class _ForegroundRequired(RuntimeError):
    def __init__(self, result: ToolResult) -> None:
        super().__init__(result.text)
        self.result = result


class DesktopMessenger:
    _SELF = {"me", "myself", "self", "self chat", "message myself", "message yourself"}
    PLATFORMS = {
        "whatsapp": {"whatsapp.exe", "whatsapp.root.exe"},
        "discord": {"discord.exe", "discordcanary.exe", "discordptb.exe"},
        "telegram": {"telegram.exe"},
        "slack": {"slack.exe"},
        "teams": {"ms-teams.exe", "msteams.exe", "teams.exe"},
    }
    WAIT_S = 3.0
    POLL_S = .1
    MAX_NODES = 3000
    MAX_DEPTH = 64
    SCAN_S = 3.0
    _lock = threading.Lock()

    def __init__(self, control: Any, bus: Any) -> None:
        self.control, self.bus = control, bus
        self._tree, self._parents = {}, {}

    @staticmethod
    def _raw_name(node: Any) -> str:
        return str(node.element_info.name or "")

    @staticmethod
    def _name(node: Any) -> str:
        return DesktopMessenger._raw_name(node).strip()

    @staticmethod
    def _kind(node: Any) -> str:
        return str(node.element_info.control_type or "")

    @staticmethod
    def _fold(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _key(node: Any) -> tuple:
        return tuple(node.element_info.runtime_id)

    @staticmethod
    def _value(node: Any) -> str | None:
        for read in (lambda: node.get_value(), lambda: node.iface_text.DocumentRange.GetText(-1)):
            try:
                value = read()
                if isinstance(value, str):
                    value = value.replace("\r\n", "\n").rstrip("\r")
                    # Chromium's empty contenteditable paragraph is exposed as
                    # a lone line break. Preserve every nonempty draft verbatim.
                    return "" if value and not value.strip("\r\n") else value
            except Exception:
                pass
        return None

    def _invalidate(self) -> None:
        self._tree, self._parents = {}, {}

    def _parent(self, node: Any) -> Any:
        return self._parents.get(self._key(node)) or node.parent()

    def _children(self, node: Any) -> list:
        cached = self._tree.get(self._key(node))
        return cached if cached is not None else node.children()

    def _nodes(self, parent: Any) -> list:
        root = self._key(parent)
        if root not in self._tree:
            # Bound the COM traversal itself, rather than slicing a fully
            # enumerated chat. An incomplete search cannot prove uniqueness.
            tree, parents, seen = {}, {}, {root}
            pending = deque([(parent, 0)])
            deadline = time.monotonic() + self.SCAN_S
            while pending:
                node, depth = pending.popleft()
                if time.monotonic() >= deadline:
                    raise RuntimeError("The app's accessible chat view took too long to inspect. Close extra panes or narrow the chat list and try again.")
                children = node.children()
                if children and depth >= self.MAX_DEPTH:
                    raise RuntimeError("The app's accessible chat view is too deeply nested to verify safely.")
                visible = []
                for child in children:
                    key = self._key(child)
                    if key in seen:
                        continue
                    seen.add(key)
                    if len(seen) > self.MAX_NODES:
                        raise RuntimeError("The app's accessible chat view is too large to verify safely. Narrow the contact search and try again.")
                    if child.is_visible():
                        visible.append(child)
                        parents[key] = node
                        pending.append((child, depth + 1))
                tree[self._key(node)] = visible
            self._tree, self._parents = tree, parents
        result, pending = [], deque(self._tree[root])
        while pending:
            node = pending.popleft()
            result.append(node)
            pending.extend(self._tree[self._key(node)])
        return result

    def _wait(self, read):
        deadline = time.monotonic() + self.WAIT_S
        while True:
            self._invalidate()
            value = read()
            if value or time.monotonic() >= deadline:
                return value
            time.sleep(self.POLL_S)

    def _guard(self) -> None:
        blocked = self.control._guard("messaging", pointer=False)
        if blocked is not None:
            raise RuntimeError(blocked.text)

    @staticmethod
    def _require_foreground(effect: str) -> None:
        blocked = require_foreground(effect)
        if blocked is not None:
            raise _ForegroundRequired(blocked)

    @staticmethod
    def _foreground_is(window: Any) -> bool:
        try:
            import ctypes
            return int(ctypes.windll.user32.GetForegroundWindow()) == int(window.handle)
        except Exception:
            return False

    def _windows(self, desktop: Any, platform: str) -> list:
        import psutil
        matches = []
        for window in desktop.windows():
            try:
                if window.is_visible() and psutil.Process(window.process_id()).name().casefold() in self.PLATFORMS[platform]:
                    matches.append(window)
            except (psutil.Error, OSError, RuntimeError):
                continue
        return matches

    def _activate(self, node: Any, window: Any) -> None:
        self._require_foreground("native messaging recipient selection")
        self._guard()
        self._invalidate()
        for action in (lambda: node.invoke(), lambda: node.select(), lambda: node.iface_selection_item.Select()):
            try:
                action()
                return
            except Exception:
                pass
        self._require_foreground("focus messaging app and click recipient")
        window.set_focus()
        if not self._foreground_is(window):
            raise RuntimeError("The messaging app could not take focus.")
        rect = node.rectangle()
        result = self.control.click((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2,
                                    target_check=lambda: node.is_visible() and node.is_enabled()
                                    and self._foreground_is(window) and node.rectangle() == rect)
        if not result.ok:
            raise RuntimeError(result.text)

    def _write(self, node: Any, text: str, window: Any, *, replace: bool = False) -> bool:
        self._require_foreground("native messaging editor mutation")
        self._invalidate()
        before = self._value(node)
        if before is None or (before and not replace):
            return False  # Never overwrite somebody's existing draft.
        for action in (lambda: node.set_edit_text(text), lambda: node.iface_value.SetValue(text)):
            self._guard()
            try:
                action()
            except Exception:
                pass
            current = self._value(node)
            if current == text:
                return True
            if current != before:
                return False  # A partial write is terminal, not permission to append again.

        # Chromium contenteditable controls often expose TextPattern but not
        # ValuePattern. Paste into that exact focused control, then read back.
        self._require_foreground("focus messaging editor and write clipboard")
        import pyperclip
        self._guard()
        window.set_focus()
        node.set_focus()
        # WhatsApp's UIA SetFocus can focus the contenteditable while clearing
        # the native foreground window. Restore that same verified top-level
        # window, then require both independent focus checks before any paste.
        if not self._foreground_is(window):
            window.set_focus()
        if not self._foreground_is(window) or not node.has_keyboard_focus():
            return False
        previous = pyperclip.paste()
        pyperclip.copy(text)
        try:
            if pyperclip.paste() != text:
                return False
            if before and not self.control.send_hotkeys("ctrl+a").ok:
                return False
            self._guard()
            if not self._foreground_is(window) or not node.has_keyboard_focus():
                return False
            if not self.control.send_hotkeys("ctrl+v").ok:
                return False
            return bool(self._wait(lambda: self._value(node) == text))
        finally:
            # Restore only after the asynchronous paste has demonstrably read
            # our value, and only while no one else has changed the clipboard.
            if self._value(node) == text and pyperclip.paste() == text:
                pyperclip.copy(previous)

    def _search(self, window: Any, platform: str) -> Any:
        hints = (("find or start a conversation", "where would you like to go", "quick switcher")
                 if platform == "discord" else
                 ("search", "find or start a conversation", "where would you like to go", "quick switcher"))
        def fields():
            return [node for node in self._nodes(window)
                    if self._kind(node) in {"Edit", "ComboBox"}
                    and any(hint in self._fold(self._name(node)) for hint in hints)]
        found = fields()
        if not found and platform in {"discord", "slack"}:
            launcher_names = {"find or start a conversation"} if platform == "discord" else {"find or start a conversation", "search"}
            launchers = [node for node in self._nodes(window) if self._kind(node) == "Button"
                         and self._fold(self._name(node)) in launcher_names]
            if len(launchers) == 1:
                self._activate(launchers[0], window)
                found = self._wait(fields)
        if not found and platform == "discord":
            # Discord's ordinary Search field searches message history. It is
            # not a recipient picker. Open the app's specific quick switcher.
            self._require_foreground("open Discord recipient picker with keyboard input")
            self._guard()
            window.set_focus()
            if not self._foreground_is(window):
                raise RuntimeError("Discord could not take focus to open its recipient picker.")
            opened = self.control.send_hotkeys("ctrl+k")
            if not opened.ok:
                raise RuntimeError(opened.text)
            found = self._wait(fields)
        if len(found) != 1:
            raise RuntimeError("The app must expose one readable contact search box. Open its chat list and try again.")
        return found[0]

    def _recipients(self, window: Any, contact: str) -> list:
        wanted = self._fold(contact)
        matches = {}
        for node in self._nodes(window):
            kind = self._kind(node)
            if kind not in {"ListItem", "TreeItem", "Hyperlink", "DataItem"}:
                continue
            label = self._name(node)
            target = node
            if kind == "DataItem":
                # Current WhatsApp exposes a DataGrid row, then a nested
                # selectable DataItem with the same aggregated accessible name.
                # Preview cells are also DataItems: only consider real grid rows.
                if self._kind(self._parent(node)) != "DataGrid":
                    continue
                fields = re.split(r"[\r\n,:]", label, maxsplit=1)
                head = fields[0].strip()
                if wanted in self._SELF:
                    matched = bool(re.search(r"\(\s*you\s*\)\s*$", head, re.I)) or self._fold(head) == "message yourself"
                else:
                    # Delimiter-bounded candidate matching is followed by an
                    # independent exact conversation-header check before typing.
                    matched = self._fold(label) == wanted or self._fold(head) == wanted
                if not matched:
                    continue
                for child in self._nodes(node):
                    if self._kind(child) != "DataItem" or self._name(child) != label:
                        continue
                    try:
                        child.iface_selection_item
                    except Exception:
                        continue
                    target = child
                    break
            elif self._fold(label) != wanted:
                # Chat rows commonly expose preview text after their first
                # Text child. Match that name, never a substring in a preview.
                labels = [self._name(child) for child in self._children(node)
                          if self._kind(child) == "Text" and self._name(child)]
                if not labels or self._fold(labels[0]) != wanted:
                    continue
            matches[self._key(node)] = target
        return list(matches.values())

    def _select_self(self, window: Any) -> bool:
        """Resolve only the application's explicit self entry, never a name guess."""
        choices = []
        for node in self._nodes(window):
            if self._kind(node) not in {"DataItem", "ListItem", "TreeItem", "Button", "Hyperlink"}:
                continue
            first = re.split(r"[\r\n,:]", self._name(node), maxsplit=1)[0].strip()
            if not re.search(r"\(\s*you\s*\)\s*$", first, re.I) and self._fold(first) != "message yourself":
                continue
            try:
                node.iface_selection_item
            except Exception:
                continue
            choices.append(node)
        if not choices:
            launchers = [node for node in self._nodes(window) if self._kind(node) == "Button"
                         and self._fold(self._name(node)) in {"new chat", "new message", "new conversation"}]
            if len(launchers) != 1:
                return False
            self._activate(launchers[0], window)
            # Re-enter only once: a second attempt must not open another picker.
            self._wait(lambda: any(re.search(r"\(\s*you\s*\)\s*$", re.split(r"[\r\n,:]", self._name(node), maxsplit=1)[0], re.I)
                                  for node in self._nodes(window)))
            for node in self._nodes(window):
                if self._kind(node) not in {"DataItem", "ListItem", "TreeItem"}:
                    continue
                first = re.split(r"[\r\n,:]", self._name(node), maxsplit=1)[0].strip()
                if not re.search(r"\(\s*you\s*\)\s*$", first, re.I):
                    continue
                try:
                    node.iface_selection_item
                except Exception:
                    continue
                choices.append(node)
        if len(choices) != 1:
            return False
        self._activate(choices[0], window)
        return True

    def _conversation(self, window: Any, contact: str) -> tuple[Any, Any] | None:
        wanted = self._fold(contact)
        self_chat = wanted in self._SELF
        matches = []
        for composer in self._nodes(window):
            if self._kind(composer) not in {"Edit", "Document"}:
                continue
            label = self._fold(self._name(composer))
            if not any(hint in label for hint in ("message", "type here", "write a reply")) or "search" in label:
                continue
            direct = not self_chat and label in {f"message {wanted}", f"message @{wanted}", f"message #{wanted}"}
            parent = self._parent(composer)
            candidate = None
            for _ in range(5):
                if self._key(parent) == self._key(window):
                    break
                children = self._nodes(parent)
                # A conversation pane contains the composer and an exact header,
                # and excludes the contact list/search. The whole app is never
                # accepted as recipient evidence.
                allowed_headers = ({"message yourself"} if self_chat else
                                   {wanted, f"chat with {wanted}", f"contact info, {wanted}"})
                header = any(self._kind(node) in ({"Header"} if self_chat else {"Header", "Text", "Button", "Group"})
                             and self._fold(self._name(node)) in allowed_headers
                             for node in children)
                def active_self_header(node):
                    if (self._kind(node) != "Button" or "message yourself" not in self._fold(self._name(node))
                            or not re.search(r"\(\s*you\s*\)", self._name(node), re.I)):
                        return False
                    ancestor = self._parent(node)
                    for _ in range(self.MAX_DEPTH):
                        if self._key(ancestor) == self._key(parent):
                            return True
                        if self._kind(ancestor) in {"DataGrid", "DataItem", "List", "ListItem", "Tree", "TreeItem"}:
                            return False
                        ancestor = self._parent(ancestor)
                    return False
                explicit_self_header = self_chat and sum(active_self_header(node) for node in children) == 1
                has_list = any(self._kind(node) in {"ListItem", "TreeItem"}
                               and self._fold(self._name(node)) == wanted for node in children)
                has_search = any(self._kind(node) in {"Edit", "ComboBox"}
                                 and "search" in self._fold(self._name(node)) for node in children)
                if (direct or header or explicit_self_header) and not has_list and (not has_search or explicit_self_header):
                    # Modern WhatsApp shares the first common ancestor with its
                    # sidebar. The explicit '(You) Message yourself' header is
                    # independent account evidence; retain the nearest ancestor.
                    if not explicit_self_header or candidate is None:
                        candidate = (parent, composer)
                parent = self._parent(parent)
            if candidate:
                matches.append(candidate)
        return matches[0] if len(matches) == 1 else None

    def _outgoing(self, pane: Any, message: str) -> set:
        observed = set()
        nodes = self._nodes(pane)
        self_headers = []
        for node in nodes:
            raw = self._name(node)
            if (self._kind(node) != "Button" or "message yourself" not in self._fold(raw)
                    or not re.search(r"\(\s*you\s*\)", raw, re.I)):
                continue
            ancestor = self._parent(node)
            for _ in range(self.MAX_DEPTH):
                if self._key(ancestor) == self._key(pane):
                    self_headers.append(re.split(r"\(\s*you\s*\)", raw, maxsplit=1, flags=re.I)[0].strip().lstrip("@"))
                    break
                if self._kind(ancestor) in {"DataGrid", "DataItem", "List", "ListItem", "Tree", "TreeItem"}:
                    break
                ancestor = self._parent(ancestor)
        for node in nodes:
            raw = self._raw_name(node)
            if self._kind(node) not in {"Text", "DataItem", "ListItem", "Group"}:
                continue
            # WhatsApp exposes a self-message in the account's timestamped
            # chat-row preview. Require its active self header independently,
            # the same account identity and exact body. The caller still
            # requires a fresh observation compared with its pre-Send state.
            if (len(self_headers) == 1 and self._kind(node) == "DataItem"
                    and self._kind(self._parent(node)) == "DataGrid"):
                preview = re.fullmatch(r"(?P<identity>.+?)\s*\(\s*you\s*\)\s+\d{1,2}:\d{2}(?:\s*[ap]m)?\s+"
                                       + re.escape(message), raw, re.I)
                if preview and self._fold(preview.group("identity").strip().lstrip("@")) == self._fold(self_headers[0]):
                    observed.add(self._key(node))
                    continue
            if re.fullmatch(r"you\s*[:,]\s*" + re.escape(message)
                            + r"(?:(?:\s|,)+(?:\d{1,2}:\d{2}(?:\s*[ap]m)?|sent|delivered|read))*", raw, re.I):
                observed.add(self._key(node))
                continue
            if raw != message:
                continue
            parent = self._parent(node)
            for _ in range(6):
                label = self._fold(self._name(parent))
                markers = [self._fold(self._name(child)) for child in self._children(parent)
                           if self._kind(child) in {"Text", "Image", "Button"}]
                if (re.match(r"^(you\s*[:,]|sent:|outgoing\b)", label)
                        or any(marker in {"you", "sent", "delivered", "read"} for marker in markers)):
                    observed.add(self._key(node))
                    break
                if self._key(parent) == self._key(pane):
                    break
                parent = self._parent(parent)
        return observed

    def send(self, platform: str, contact: str, message: str) -> ToolResult:
        if not self._lock.acquire(blocking=False):
            return ToolResult("Another desktop message is in progress. Wait for its result before sending again.", ok=False)
        invoked = False
        try:
            self._invalidate()
            self._guard()
            self._require_foreground("native messaging delivery")
            if not foreground_authorized():
                return ToolResult("Native app messaging requires explicit host foreground authorization. Connect the owned messaging browser for background delivery.", ok=False,
                                  receipt={"kind": "messaging", "status": "requires_foreground", "sent": False, "attempted": False})
            with self.control._uia_desktop() as desktop:
                windows = self._windows(desktop, platform)
                if not windows:
                    self._require_foreground("launch visible messaging app")
                    opened = self.control.open_application(platform)
                    if not opened.ok:
                        return ToolResult(f"Could not open {platform.title()}: {opened.text}", ok=False)
                    windows = self._wait(lambda: self._windows(desktop, platform))
                if len(windows) != 1:
                    return ToolResult(f"Open one signed-in {platform.title()} app window and try again; found {len(windows)} matching app windows.", ok=False)
                window = windows[0]
                conversation = self._conversation(window, contact) if self._fold(contact) in self._SELF else None
                if not conversation and self._fold(contact) in self._SELF:
                    if not self._select_self(window):
                        raise RuntimeError("The app did not expose one explicit self entry. No contact name was guessed.")
                    conversation = self._wait(lambda: self._conversation(window, contact))
                    if not conversation:
                        raise RuntimeError("The selected self conversation could not be independently verified.")
                if not conversation:
                    search = self._search(window, platform)
                    if not self._write(search, contact, window, replace=True):
                        raise RuntimeError("The contact search could not be verified.")
                    recipients = self._wait(lambda: self._recipients(window, contact))
                    if len(recipients) != 1:
                        raise RuntimeError(f"Expected one exact contact named '{contact}'; found {len(recipients)}. Use their full unique displayed name.")
                    self._activate(recipients[0], window)
                    conversation = self._wait(lambda: self._conversation(window, contact))
                if not conversation:
                    raise RuntimeError("The selected conversation's recipient could not be verified.")
                pane, composer = conversation
                if not self._write(composer, message, window):
                    raise RuntimeError("The message could not be verified in an empty composer; any existing draft was preserved.")
                before = self._outgoing(pane, message)
                def send_buttons():
                    return [node for node in self._nodes(pane) if self._kind(node) == "Button"
                            and self._fold(self._name(node)) in {"send", "send message", "send now", "send (enter)", "send message (enter)"}
                            and node.is_enabled()]
                send = send_buttons() if platform == "discord" else self._wait(send_buttons)
                targeted_enter = (not send and platform == "discord"
                                  and self._fold(self._name(composer)) == f"message @{self._fold(contact)}")
                if len(send) != 1 and not targeted_enter:
                    raise RuntimeError("The message is drafted, but one explicit Send button could not be identified.")
                if targeted_enter:
                    self._require_foreground("submit Discord message with keyboard input")
                    self._guard()
                    window.set_focus()
                    composer.set_focus()
                    if not self._foreground_is(window):
                        window.set_focus()
                    if not self._foreground_is(window) or not composer.has_keyboard_focus():
                        raise RuntimeError("Discord's verified recipient composer could not take keyboard focus.")
                self._guard()
                self._invalidate()
                current = self._conversation(window, contact)
                if (not current or self._key(current[1]) != self._key(composer)
                        or self._value(composer) != message):
                    raise RuntimeError("The recipient or composer changed before sending.")
                if targeted_enter and (not self._foreground_is(window) or not composer.has_keyboard_focus()):
                    raise RuntimeError("Discord's recipient composer lost keyboard focus before sending.")
                invoke_send = send[0].iface_invoke.Invoke if is_background() and not targeted_enter else None
                self.bus.log.emit(f"MESSAGING: sending through {platform.title()} to the verified contact {contact}.")
                # From this point any exception is an uncertain submission. Never
                # try another button, delivery route or automatic resend.
                invoked = True
                if targeted_enter:
                    # Discord normally has no Send button. UIA has identified
                    # this exact recipient's editor and confirmed foreground
                    # keyboard focus; never send Enter to an inferred target.
                    composer.type_keys("{ENTER}", set_foreground=False)
                else:
                    if is_background():
                        invoke_send()
                    else:
                        send[0].invoke()
                def submitted():
                    active = self._conversation(window, contact)
                    return (active and self._key(active[1]) == self._key(composer)
                            and self._value(composer) == "" and bool(self._outgoing(active[0], message) - before))
                if not self._wait(submitted):
                    raise RuntimeError("The outgoing message was not observed after Send.")
                return ToolResult(f"Sent to {contact} in {platform.title()}; the app shows the outgoing message. Delivery/read status is not confirmed.",
                                  receipt={"kind": "messaging", "platform": platform, "status": "submitted",
                                           "recipient": contact, "sent": True, "delivery_verified": False})
        except _ForegroundRequired as exc:
            if invoked:
                return ToolResult("A desktop message may have been submitted. Check its result before sending again.", ok=False,
                                  receipt={"kind": "messaging", "status": "uncertain", "sent": None})
            return exc.result
        except Exception as exc:
            state = "uncertain" if invoked else "not_sent"
            suffix = " Do not resend automatically; check the conversation first." if invoked else " No message was sent."
            return ToolResult(f"{platform.title()} messaging: {exc}.{suffix}", ok=False,
                              receipt={"kind": "messaging", "platform": platform, "status": state, "sent": None if invoked else False})
        finally:
            self._lock.release()
