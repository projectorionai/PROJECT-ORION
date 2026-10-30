"""Bounded display and visual controls adopted from the JARVIS tool pattern.

HTML is data, never executable code. Remote images are fetched through a
resolver that rejects private addresses, then embedded as validated bytes.
"""
from __future__ import annotations

import asyncio
import base64
import html
from html.parser import HTMLParser
import ipaddress
import math
import re
import socket
from urllib.parse import urlsplit


TOOLS = ("display", "ui_theme", "ui_reactor", "ui_orbit", "ui_chrome",
         "ui_effect", "ui_screen", "ui_reset")
MAX_HTML = 32_768
MAX_IMAGE = 131_072
TAGS = frozenset({"div", "span", "p", "h1", "h2", "h3", "h4", "strong", "b", "i",
                  "em", "small", "ul", "ol", "li", "table", "thead", "tbody", "tr",
                  "th", "td", "pre", "code", "br", "hr", "a", "img"})
BLOCKED = frozenset({"script", "style", "iframe", "object", "embed", "svg", "math",
                     "form", "input", "button", "textarea", "select", "link", "meta"})


def public_url(value: str) -> str:
    value = str(value).strip()
    parts = urlsplit(value)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or
            parts.password or parts.port not in {None, 80, 443} or len(value) > 2048 or
            any(ord(char) < 32 for char in value)):
        raise ValueError("Use a public HTTP or HTTPS media address without credentials.")
    host = parts.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Private media addresses are unavailable.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Private media addresses are unavailable.")
    return value


def image_data(data: bytes) -> str:
    if not data or len(data) > MAX_IMAGE:
        raise ValueError("A display image must be smaller than 128 KB.")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        kind = "png"
    elif data.startswith(b"\xff\xd8\xff"):
        kind = "jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        kind = "gif"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = "webp"
    else:
        raise ValueError("Only PNG, JPEG, GIF and WebP display images are supported.")
    from io import BytesIO
    from PIL import Image
    try:
        with Image.open(BytesIO(data)) as picture:
            if picture.width * picture.height > 1_048_576:
                raise ValueError("Use a display image smaller than one megapixel.")
            picture.verify()
    except ValueError:
        raise
    except Exception:
        raise ValueError("The display image could not be decoded.") from None
    return f"data:image/{kind};base64," + base64.b64encode(data).decode("ascii")


async def fetch_image(url: str) -> str:
    import aiohttp
    from aiohttp.abc import AbstractResolver

    class PublicResolver(AbstractResolver):
        async def resolve(self, host, port=0, family=socket.AF_INET):
            records = await asyncio.get_running_loop().getaddrinfo(
                host, port, family=family, type=socket.SOCK_STREAM)
            if not records or any(not ipaddress.ip_address(row[4][0]).is_global for row in records):
                raise ValueError("Private media addresses are unavailable.")
            return [{"hostname": host, "host": row[4][0], "port": port,
                     "family": row[0], "proto": row[2], "flags": socket.AI_NUMERICHOST}
                    for row in records]

        async def close(self):
            pass

    target = public_url(url)
    timeout = aiohttp.ClientTimeout(total=12, sock_connect=4, sock_read=6)
    async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(resolver=PublicResolver()),
            timeout=timeout, trust_env=False, cookie_jar=aiohttp.DummyCookieJar()) as session:
        for _ in range(4):
            async with session.get(target, allow_redirects=False) as response:
                if response.status in {301, 302, 303, 307, 308}:
                    from urllib.parse import urljoin
                    target = public_url(urljoin(target, response.headers.get("Location", "")))
                    continue
                if response.status != 200:
                    raise ValueError("The image host did not return an image.")
                data = bytearray()
                async for chunk in response.content.iter_chunked(16384):
                    data.extend(chunk)
                    if len(data) > MAX_IMAGE:
                        raise ValueError("A display image must be smaller than 128 KB.")
                return image_data(bytes(data))
    raise ValueError("The image address redirected too many times.")


class _Markup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self.images: list[str] = []
        self.blocked = 0
        self.visible = False

    def handle_starttag(self, tag, attrs):
        if tag in BLOCKED:
            if tag not in {"input", "link", "meta", "embed"}:
                self.blocked += 1
            return
        if self.blocked or tag not in TAGS:
            return
        values = dict(attrs)
        safe = ""
        if tag == "a":
            try:
                url = public_url(values.get("href", ""))
            except (ValueError, TypeError):
                url = ""
            if url:
                safe = f' href="{html.escape(url, quote=True)}" rel="noopener noreferrer"'
        elif tag == "img":
            source = values.get("src", "")
            try:
                if source.startswith("data:image/"):
                    prefix, encoded = source.split(",", 1)
                    if not prefix.endswith(";base64") or len(encoded) > MAX_IMAGE * 2:
                        raise ValueError("Invalid image data")
                    source = image_data(base64.b64decode(encoded, validate=True))
                else:
                    source = public_url(source)
                    if len(self.images) >= 4:
                        raise ValueError("A display can contain at most four images.")
                    self.images.append(source)
                    source = f"__ORION_IMAGE_{len(self.images) - 1}__"
            except (ValueError, TypeError):
                return
            self.visible = True
            safe = f' src="{html.escape(source, quote=True)}" alt="{html.escape(values.get("alt", "Image")[:150], quote=True)}"'
        elif tag in {"td", "th"}:
            for attr in ("colspan", "rowspan"):
                value = values.get(attr, "")
                if str(value).isdigit() and 1 <= int(value) <= 20:
                    safe += f' {attr}="{int(value)}"'
        self.output.append(f"<{tag}{safe}>")

    def handle_endtag(self, tag):
        if tag in BLOCKED:
            self.blocked = max(0, self.blocked - 1)
        elif not self.blocked and tag in TAGS and tag not in {"img", "br", "hr"}:
            self.output.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.blocked:
            self.visible |= bool(data.strip())
            self.output.append(html.escape(data))


def sanitise_markup(markup: str) -> tuple[str, list[str]]:
    if not isinstance(markup, str) or not 0 < len(markup) <= MAX_HTML:
        raise ValueError("Provide non-empty display markup smaller than 32 KB.")
    parser = _Markup()
    parser.feed(markup)
    parser.close()
    if not parser.visible:
        raise ValueError("The display needs visible content.")
    return "".join(parser.output), parser.images


def _colour(value) -> str:
    if value is None or str(value).lower() in {"auto", "reset"}:
        return "#ff1a3c"
    text = str(value)
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        raise ValueError("Use a six-digit hexadecimal colour, such as #25bce8.")
    return text.lower()


def _number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Use a number between {low} and {high}.")
    return value


async def prepare(tool: str, args: dict) -> dict:
    if tool not in TOOLS or not isinstance(args, dict):
        raise ValueError("Unknown presentation operation.")
    result = {"operation": tool}
    if tool == "display":
        markup, images = sanitise_markup(args.get("html", ""))
        embedded = await asyncio.gather(*(fetch_image(url) for url in images))
        for index, data in enumerate(embedded):
            markup = markup.replace(f"__ORION_IMAGE_{index}__", data)
        result.update(title=str(args.get("title", "ORION display"))[:150], html=markup)
    elif tool == "ui_theme":
        result["accent"] = _colour(args.get("accent"))
    elif tool == "ui_reactor":
        shape = args.get("shape", "sphere")
        if shape not in {"sphere", "torus", "cube", "octahedron", "icosahedron"}:
            raise ValueError("Choose sphere, torus, cube, octahedron or icosahedron.")
        result.update(shape=shape, scale=_number(args.get("scale", 1), 0.2, 3),
                      intensity=_number(args.get("intensity", 1), 0, 3),
                      colour=_colour(args.get("color")), rings=args.get("rings", True))
        if not isinstance(result["rings"], bool):
            raise ValueError("rings must be true or false.")
    elif tool == "ui_orbit":
        items = args.get("images", [])
        if not isinstance(items, list) or len(items) > 4:
            raise ValueError("Provide at most four image addresses, or [] to clear them.")
        result["images"] = await asyncio.gather(*(fetch_image(public_url(url)) for url in items))
    elif tool == "ui_chrome":
        for key in ("telemetry", "transcript"):
            if key in args:
                if not isinstance(args[key], bool):
                    raise ValueError(f"{key} must be true or false.")
                result[key] = args[key]
        if len(result) == 1:
            raise ValueError("Set telemetry or transcript visibility.")
    elif tool == "ui_effect":
        kind = args.get("kind")
        if kind not in {"pulse", "scan", "shake", "flash", "glitch"}:
            raise ValueError("Choose pulse, scan, shake, flash or glitch.")
        result["kind"] = kind
    elif tool == "ui_screen":
        target = args.get("target", "panels")
        if target not in {"panels", "transcript", "all"}:
            raise ValueError("Choose panels, transcript or all.")
        result["target"] = target
    return result


def declarations():
    string = lambda description: {"type": "STRING", "description": description}
    number = lambda description: {"type": "NUMBER", "description": description}
    specs = {
        "display": ("Show a graphical information card on the PC and paired phone. Compose safe HTML with headings, tables, lists and up to four small public images. Scripts and embedded pages are unavailable.",
                    {"title": string("Card title"), "html": string("Visible HTML, at most 32 KB")}, ["html"]),
        "ui_theme": ("Retint ORION's desktop shell. Restore its normal accent with ui_reset.", {"accent": string("Six-digit hex colour or auto")}, ["accent"]),
        "ui_reactor": ("Reshape ORION's 3-D orb. Requires the 3-D face renderer.",
                       {"shape": string("sphere, torus, cube, octahedron or icosahedron"), "color": string("Six-digit hex colour"),
                        "scale": number("0.2–3"), "intensity": number("0–3"), "rings": {"type": "BOOLEAN"}}, ["shape"]),
        "ui_orbit": ("Place up to four small public images around ORION's 3-D orb. An empty list clears them.",
                     {"images": {"type": "ARRAY", "items": string("Public image address, smaller than 128 KB")}}, ["images"]),
        "ui_chrome": ("Show or hide desktop telemetry and transcript panels. Hidden panels automatically return after 30 seconds; command controls stay accessible.",
                      {"telemetry": {"type": "BOOLEAN"}, "transcript": {"type": "BOOLEAN"}}, []),
        "ui_effect": ("Show one brief desktop visual acknowledgement. Respect reduced motion; use sparingly.", {"kind": string("pulse, scan, shake, flash or glitch")}, ["kind"]),
        "ui_screen": ("Clear displayed cards or the visible transcript. Stored memory is retained.", {"target": string("panels, transcript or all")}, ["target"]),
        "ui_reset": ("Restore ORION's accent, orb and panel visibility; keep displayed content.", {}, []),
    }
    return [{"name": name, "description": description,
             "parameters": {"type": "OBJECT", "properties": properties, "required": required}}
            for name, (description, properties, required) in specs.items()]
