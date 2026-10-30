"""
Exact arithmetic and unit conversion — so ORION never does sums in his head.

The voice model can multiply, but "what's 12 stone in kilos", "30 degrees C
in F", "how many ml in a pint" and "17.5% of 1,240" are exactly where a
language model's mental arithmetic slips — and imperial units are worse:
a UK pint is 568 ml, a US one 473. This answers them exactly.

* Arithmetic is parsed into an AST and evaluated over a whitelist of operators
  and functions — never ``eval`` — with limits on exponent size so "9**9**9"
  cannot hang the process.
* Spoken maths is rewritten first: "15% of 240", "7 times 8", "9 squared",
  "square root of 81", "1,240".
* Conversions: length, mass, volume, area, speed, time, data, temperature,
  energy and pressure; British by default (pint, gallon, ton), US when said
  ("us pint"); compound forms both ways ("5 ft 10 in cm", "80 kg in stone" ->
  "12 st 8.4 lb").
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from .data import ToolResult

# ── arithmetic ───────────────────────────────────────────────────────────────

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log10, "ln": math.log,
          "sin": lambda x: math.sin(math.radians(x)), "cos": lambda x: math.cos(math.radians(x)),
          "tan": lambda x: math.tan(math.radians(x)), "floor": math.floor, "ceil": math.ceil,
          "factorial": math.factorial}
_NAMES = {"pi": math.pi, "e": math.e}


class CalcError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000 or (abs(left) > 1 and abs(right) * math.log10(abs(left)) > 300):
                raise CalcError("that power is too large to work out")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalcError("division by zero")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and len(node.args) == 1 and not node.keywords):
        value = _eval(node.args[0])
        if node.func.id == "factorial" and (value < 0 or value > 170 or value != int(value)):
            raise CalcError("factorial needs a whole number from 0 to 170")
        return _FUNCS[node.func.id](int(value) if node.func.id == "factorial" else value)
    raise CalcError("that is not a calculation I can do exactly")


_SPOKEN = (
    (r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\s+of\s+", r"(\1/100)*"),
    (r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", r"(\1/100)"),
    (r"\bsquare root of\s+", "sqrt "), (r"\bcube root of\s+([\d.]+)", r"(\1**(1/3))"),
    (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
    (r"\bto the power of\b", "**"), (r"\bmultiplied by\b|\btimes\b|×|\bx\b", "*"),
    (r"\bdivided by\b|\bover\b|÷", "/"), (r"\bplus\b", "+"), (r"\bminus\b", "-"),
    (r"\bmod(?:ulo)?\b", "%"), (r"\^", "**"),
)


def normalise_expression(text: str) -> str:
    expr = str(text or "").lower().strip()
    expr = re.sub(r"^(?:what(?:'s| is)|calculate|work out|compute|how much is)\s+", "", expr)
    expr = expr.rstrip("?.! =")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)            # 1,240 -> 1240
    for pattern, repl in _SPOKEN:
        expr = re.sub(pattern, repl, expr)
    expr = re.sub(r"\b(sqrt|ln|log|sin|cos|tan|abs|floor|ceil|factorial)\s+([\d.]+)", r"\1(\2)", expr)
    return expr


def evaluate(text: str) -> float:
    expr = normalise_expression(text)
    if not expr or len(expr) > 200:
        raise CalcError("give me a calculation")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalcError("I could not read that as a calculation") from exc
    return _eval(tree)


# ── algebra ──────────────────────────────────────────────────────────────────
#
# 27 Oct 2026: "2x to the power of 3 - 2" (and "x^3 - 2 = 2x^3 - 2") came back
# as "I couldn't process that as a calculation". Expressions and equations
# with a letter in them are now worked exactly with sympy - built from the
# parsed tree over the same kind of whitelist as the arithmetic above, never
# from sympy's own parser, which evaluates text.

_ALGEBRA_HINT = re.compile(r"=|\d\s*[a-z]\b|\b[a-z]\s*(?:\^|\*\*|²|³|squared|cubed|to the power)|"
                           r"\b(?:solve|simplify|expand|factori[sz]e|factor|differentiate|"
                           r"derivative|integrate)\b", re.IGNORECASE)
_ALGEBRA_VERB = re.compile(r"^\s*(solve|simplify|expand|factori[sz]e|factor|differentiate|"
                           r"derivative of|integrate)\b\s*(?:for\s+([a-z])\b)?\s*:?", re.IGNORECASE)
_ALGEBRA_SPOKEN = tuple((pattern.replace(r"|\bx\b", ""), repl) for pattern, repl in _SPOKEN) + (
    (r"²", "**2"), (r"³", "**3"))
_ALGEBRA_FUNCS = ("sqrt", "sin", "cos", "tan", "ln", "log", "abs", "exp")
#: Polynomials up to this degree are solved; beyond it sympy can take minutes.
_MAX_DEGREE = 6


def _algebra_expr(text: str) -> str:
    expr = text.lower().strip().rstrip("?.!").strip()
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    for pattern, repl in _ALGEBRA_SPOKEN:
        expr = re.sub(pattern, repl, expr)
    expr = re.sub(r"(\d|\))\s*(?=[a-z(])", r"\1*", expr)                # 2x, 3(x+1), (x)(y)
    funcs = "|".join(_ALGEBRA_FUNCS)
    expr = re.sub(rf"(?<![a-z])(?!(?:{funcs})\b)([a-z])\s*\(", r"\1*(", expr)   # x(x+1)
    return expr


def _sympy_of(node: ast.AST, sp: Any) -> Any:
    if isinstance(node, ast.Expression):
        return _sympy_of(node.body, sp)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return sp.Rational(str(node.value)) if isinstance(node.value, float) else sp.Integer(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sp.pi
        if node.id == "e":
            return sp.E
        if len(node.id) == 1:
            return sp.Symbol(node.id)
        raise CalcError(f"'{node.id}' is not something I can work with")
    if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
        left, right = _sympy_of(node.left, sp), _sympy_of(node.right, sp)
        if isinstance(node.op, ast.Pow) and (not right.is_number or abs(float(right)) > 100):
            raise CalcError("that power is too large to work with")
        return {ast.Add: left.__add__, ast.Sub: left.__sub__, ast.Mult: left.__mul__,
                ast.Div: left.__truediv__, ast.Pow: left.__pow__}[type(node.op)](right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        value = _sympy_of(node.operand, sp)
        return -value if isinstance(node.op, ast.USub) else value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _ALGEBRA_FUNCS and len(node.args) == 1 and not node.keywords):
        func = {"ln": sp.log, "log": lambda v: sp.log(v, 10), "abs": sp.Abs}.get(
            node.func.id) or getattr(sp, node.func.id)
        return func(_sympy_of(node.args[0], sp))
    raise CalcError("that is not something I can work out exactly")


def _shown(value: Any) -> str:
    text = str(value).replace("**", "^").replace(")*(", ")(")
    return re.sub(r"(\d)\*(?=[a-z(])", r"\1", text)


def _solution(value: Any) -> str:
    exact = _shown(value)
    try:
        if value.is_real and not value.is_Rational:
            return f"{exact} (about {_fmt(float(value))})"
    except (TypeError, ValueError):
        pass
    return exact


def _roots(found: list[Any], var: Any) -> str:
    """The real solutions, spoken; complex ones are counted, not read out."""
    real = [v for v in found if v.is_real]
    complex_count = len(found) - len(real)
    said = f" or {var} = ".join(_solution(v) for v in real)
    said = f"{var} = {said}" if real else f"no real value of {var}"
    if complex_count:
        said += f" (plus {complex_count} complex solution{'s' if complex_count != 1 else ''})"
    return said


def algebra(text: str) -> str | None:
    """Solve, simplify, expand, factorise or differentiate an expression with
    letters in it; None when *text* is not algebra."""
    raw = str(text or "").strip()
    if not raw or len(raw) > 200 or not _ALGEBRA_HINT.search(raw):
        return None
    verb_match = _ALGEBRA_VERB.match(raw)
    verb = (verb_match.group(1).lower() if verb_match else "")
    body = raw[verb_match.end():] if verb_match else raw
    body = re.sub(r"^\s*(?:what(?:'s| is)|calculate|work out|compute)\s+", "", body,
                  flags=re.IGNORECASE).strip().lstrip("= ").strip()
    sides = _algebra_expr(body).split("=")
    if len(sides) > 2 or not all(side.strip() for side in sides):
        return None
    try:
        import sympy as sp
    except ImportError:
        return None
    try:
        exprs = [_sympy_of(ast.parse(side.strip(), mode="eval"), sp) for side in sides]
    except (SyntaxError, CalcError, TypeError, ValueError):
        return None
    symbols = sorted(set().union(*(e.free_symbols for e in exprs)), key=lambda s: s.name)
    if not symbols:
        return None                     # plain arithmetic: evaluate() answers it
    wanted = verb_match.group(2) if verb_match and verb_match.group(2) else ""
    var = next((s for s in symbols if s.name == wanted), None) or \
        next((s for s in symbols if s.name == "x"), symbols[0])
    try:
        if len(exprs) == 2:
            equation = sp.expand(exprs[0] - exprs[1])
            if equation == 0:
                return f"{_shown(exprs[0])} = {_shown(exprs[1])} holds for every value of {var}."
            if not equation.is_polynomial(var) or sp.degree(equation, var) > _MAX_DEGREE:
                return (f"I can solve equations in {var} up to the power of {_MAX_DEGREE} "
                        "exactly; that one is beyond it.")
            found = sp.solve(equation, var)
            if not found:
                return f"{_shown(exprs[0])} = {_shown(exprs[1])} has no solution for {var}."
            return _roots(found, var) + "."
        expr = exprs[0]
        # The owner's expression as written down ("2x^3 - 2"), not as spoken.
        asked = re.sub(r"\s*\^\s*", "^", _shown(_algebra_expr(body)))
        asked = re.sub(r"\s+", " ", asked).strip()
        if verb in {"differentiate", "derivative of"}:
            return f"d/d{var} of {_shown(expr)} = {_shown(sp.diff(expr, var))}"
        if verb == "integrate":
            return f"The integral of {_shown(expr)} d{var} = {_shown(sp.integrate(expr, var))} + C"
        if verb == "expand":
            return f"{_shown(expr)} expands to {_shown(sp.expand(expr))}."
        if verb.startswith("factor"):
            return f"{_shown(expr)} factorises to {_shown(sp.factor(expr))}."
        simple, factored = sp.simplify(expr), sp.factor(expr)
        typed = asked
        said = (f"{typed} simplifies to {_shown(simple)}"
                if _shown(simple).replace(" ", "") != typed.replace(" ", "")
                else f"{typed} is already as simple as it goes")
        if _shown(factored) not in {_shown(simple), _shown(expr)}:
            said += f"; factorised, {_shown(factored)}"
        solvable = expr.is_polynomial(var) and 0 < sp.degree(expr, var) <= _MAX_DEGREE
        if solvable and (verb == "solve" or len(symbols) == 1):
            roots = sp.solve(expr, var)
            if roots:
                said += f". It is zero when {_roots(roots, var)}"
        return said + "."
    except (TypeError, ValueError, NotImplementedError, AttributeError):
        return None


# ── units ────────────────────────────────────────────────────────────────────

#: unit word -> (dimension, size in the dimension's base unit).
_UNITS: dict[str, tuple[str, float]] = {}


def _add(dimension: str, factor: float, *names: str) -> None:
    for name in names:
        _UNITS[name] = (dimension, factor)


_add("length", 1e-3, "mm", "millimetre", "millimetres", "millimeter", "millimeters")
_add("length", 1e-2, "cm", "centimetre", "centimetres", "centimeter", "centimeters")
_add("length", 1.0, "m", "metre", "metres", "meter", "meters")
_add("length", 1e3, "km", "kilometre", "kilometres", "kilometer", "kilometers", "k")
_add("length", 0.0254, "in", "inch", "inches", '"')
_add("length", 0.3048, "ft", "foot", "feet", "'")
_add("length", 0.9144, "yd", "yard", "yards")
_add("length", 1609.344, "mi", "mile", "miles")
_add("length", 1852.0, "nautical mile", "nautical miles", "nmi")
_add("mass", 1e-6, "mg", "milligram", "milligrams")
_add("mass", 1e-3, "g", "gram", "grams", "gramme", "grammes")
_add("mass", 1.0, "kg", "kilo", "kilos", "kilogram", "kilograms", "kgs")
_add("mass", 1000.0, "tonne", "tonnes", "metric ton", "metric tons")
_add("mass", 0.028349523125, "oz", "ounce", "ounces")
_add("mass", 0.45359237, "lb", "lbs", "pound", "pounds")
_add("mass", 6.35029318, "st", "stone", "stones")
_add("mass", 1016.0469088, "ton", "tons", "long ton", "long tons")
_add("mass", 907.18474, "short ton", "short tons", "us ton", "us tons")
_add("volume", 1e-3, "ml", "millilitre", "millilitres", "milliliter", "milliliters")
_add("volume", 1e-2, "cl", "centilitre", "centilitres")
_add("volume", 1.0, "l", "litre", "litres", "liter", "liters")
_add("volume", 0.56826125, "pint", "pints", "pt", "uk pint", "uk pints")
_add("volume", 0.473176473, "us pint", "us pints")
_add("volume", 4.54609, "gallon", "gallons", "gal", "uk gallon", "uk gallons")
_add("volume", 3.785411784, "us gallon", "us gallons")
_add("volume", 0.0284130625, "fl oz", "fluid ounce", "fluid ounces")
_add("volume", 0.0295735295625, "us fl oz", "us fluid ounce", "us fluid ounces")
_add("volume", 0.25, "cup", "cups")
_add("volume", 0.015, "tbsp", "tablespoon", "tablespoons")
_add("volume", 0.005, "tsp", "teaspoon", "teaspoons")
_add("area", 1.0, "m2", "sq m", "square metre", "square metres", "square meter", "square meters")
_add("area", 1e6, "km2", "sq km", "square kilometre", "square kilometres")
_add("area", 0.09290304, "sq ft", "ft2", "square foot", "square feet")
_add("area", 4046.8564224, "acre", "acres")
_add("area", 1e4, "hectare", "hectares", "ha")
_add("area", 2589988.110336, "sq mi", "square mile", "square miles")
_add("speed", 0.44704, "mph", "miles per hour")
_add("speed", 1 / 3.6, "kph", "km/h", "kmh", "kilometres per hour", "kilometers per hour")
_add("speed", 1.0, "m/s", "metres per second", "meters per second")
_add("speed", 0.514444, "knot", "knots", "kn")
_add("time", 1e-3, "ms", "millisecond", "milliseconds")
_add("time", 1.0, "s", "sec", "secs", "second", "seconds")
_add("time", 60.0, "min", "mins", "minute", "minutes")
_add("time", 3600.0, "h", "hr", "hrs", "hour", "hours")
_add("time", 86400.0, "day", "days")
_add("time", 604800.0, "week", "weeks")
_add("time", 31557600.0, "year", "years")
_add("data", 0.125, "bit", "bits")
_add("data", 1.0, "byte", "bytes")
_add("data", 1e3, "kb", "kilobyte", "kilobytes")
_add("data", 1e6, "mb", "megabyte", "megabytes")
_add("data", 1e9, "gb", "gigabyte", "gigabytes")
_add("data", 1e12, "tb", "terabyte", "terabytes")
_add("data", 1024.0, "kib")
_add("data", 1024.0 ** 2, "mib")
_add("data", 1024.0 ** 3, "gib")
_add("data", 1024.0 ** 4, "tib")
_add("energy", 1.0, "j", "joule", "joules")
_add("energy", 1e3, "kj", "kilojoule", "kilojoules")
_add("energy", 4184.0, "kcal", "calorie", "calories", "cal")
_add("energy", 3.6e6, "kwh", "kilowatt hour", "kilowatt hours")
_add("pressure", 1e5, "bar")
_add("pressure", 6894.757293168, "psi")
_add("pressure", 1e3, "kpa")
_add("pressure", 101325.0, "atm", "atmosphere", "atmospheres")

#: The smaller unit a bare second number belongs to ("5 ft 10", "12 st 4").
_PARTNER = {"ft": "in", "foot": "in", "feet": "in", "st": "lb", "stone": "lb", "stones": "lb",
            "lb": "oz", "lbs": "oz", "pound": "oz", "pounds": "oz", "m": "cm", "metre": "cm",
            "metres": "cm", "meter": "cm", "meters": "cm", "h": "min", "hour": "min",
            "hours": "min"}

_TEMPERATURE = {"c": "c", "celsius": "c", "centigrade": "c", "°c": "c", "degrees c": "c",
                "degrees celsius": "c", "f": "f", "fahrenheit": "f", "°f": "f",
                "degrees f": "f", "degrees fahrenheit": "f", "k": "k", "kelvin": "k"}

_UNIT_ALT = "|".join(sorted((re.escape(u) for u in list(_UNITS) + list(_TEMPERATURE)),
                            key=len, reverse=True))
_CONVERT = re.compile(
    rf"(?P<amount>-?[\d.]+)\s*(?P<src>{_UNIT_ALT})"
    rf"(?:\s*(?P<amount2>[\d.]+)\s*(?P<src2>{_UNIT_ALT})?)?"
    rf"\s+(?:in|into|to|as|in terms of)\s+(?P<dst>{_UNIT_ALT})\s*$")
_HOW_MANY = re.compile(
    rf"how many\s+(?P<dst>{_UNIT_ALT})\s+(?:are\s+)?(?:in|to)\s+(?:an?\s+|one\s+)?"
    rf"(?:(?P<amount>[\d.]+)\s*)?(?P<src>{_UNIT_ALT})\s*$")


def _temperature(value: float, src: str, dst: str) -> float:
    kelvin = {"c": value + 273.15, "f": (value - 32) * 5 / 9 + 273.15, "k": value}[src]
    return {"c": kelvin - 273.15, "f": (kelvin - 273.15) * 9 / 5 + 32, "k": kelvin}[dst]


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e15 or abs(value) < 1e-6:
        return f"{value:.6g}"
    digits = max(0, 6 - int(math.floor(math.log10(abs(value)))) - 1)
    text = f"{value:,.{min(digits, 6)}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _compound(value: float, unit: str) -> str:
    """'12.6 stone' -> ' (12 st 8.4 lb)'; '5.84 ft' -> ' (5 ft 10.1 in)'."""
    for big, small, per in (({"stone", "stones", "st"}, "lb", 14), ({"ft", "foot", "feet"}, "in", 12)):
        if unit in big and 0 < value < 30 and abs(value - round(value)) > 0.004:
            whole = int(value)
            rest = (value - whole) * per
            label = "st" if per == 14 else "ft"
            return f" ({whole} {label} {_fmt(round(rest, 1))} {small})"
    return ""


def convert(text: str) -> str | None:
    """The converted quantity as a sentence, or None if it is not a conversion."""
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(text or "").lower().strip())
    t = re.sub(r"^(?:convert|what(?:'s| is)|how much is|change)\s+", "", t).rstrip("?.! ")
    m = _CONVERT.search(t)
    how_many = None
    if m is None:
        how_many = _HOW_MANY.search(t)
        if how_many is None:
            return None
    groups = (m or how_many).groupdict()
    src, dst = groups["src"], groups["dst"]
    try:
        amount = float(groups.get("amount") or 1)
    except ValueError:
        return None
    if src in _TEMPERATURE or dst in _TEMPERATURE:
        if not (src in _TEMPERATURE and dst in _TEMPERATURE):
            return None
        value = _temperature(amount, _TEMPERATURE[src], _TEMPERATURE[dst])
        names = {"c": "°C", "f": "°F", "k": "K"}
        return (f"{_fmt(amount)}{names[_TEMPERATURE[src]]} is "
                f"{_fmt(round(value, 2))}{names[_TEMPERATURE[dst]]}.")
    s_dim, s_factor = _UNITS[src]
    d_dim, d_factor = _UNITS[dst]
    base = amount * s_factor
    said = f"{_fmt(amount)} {src}"
    if m is not None and groups.get("amount2"):
        # "5 ft 10" / "12 st 4": the second number is in the partner unit.
        partner = groups.get("src2") or _PARTNER.get(src)
        if partner is None or partner not in _UNITS:
            return None
        groups["src2"] = partner
        s2_dim, s2_factor = _UNITS[partner]
        if s2_dim != s_dim:
            return None
        base += float(groups["amount2"]) * s2_factor
        said += f" {_fmt(float(groups['amount2']))} {groups['src2']}"
    if s_dim != d_dim:
        return f"I can't convert {s_dim} into {d_dim}."
    value = base / d_factor
    return f"{said} is {_fmt(value)} {dst}{_compound(value, dst)}."


# ── time zones ───────────────────────────────────────────────────────────────
#
# "What time is it in Tokyo?" — the offset changes twice a year in most of
# these places on different dates, which is exactly where an estimate is wrong.
# Needs the tzdata package on Windows (there is no system database).

_ZONES = {
    "london": "Europe/London", "uk": "Europe/London", "britain": "Europe/London",
    "england": "Europe/London", "birmingham": "Europe/London", "manchester": "Europe/London",
    "edinburgh": "Europe/London", "dublin": "Europe/Dublin", "ireland": "Europe/Dublin",
    "paris": "Europe/Paris", "france": "Europe/Paris", "berlin": "Europe/Berlin",
    "germany": "Europe/Berlin", "madrid": "Europe/Madrid", "spain": "Europe/Madrid",
    "rome": "Europe/Rome", "italy": "Europe/Rome", "amsterdam": "Europe/Amsterdam",
    "netherlands": "Europe/Amsterdam", "brussels": "Europe/Brussels", "lisbon": "Europe/Lisbon",
    "portugal": "Europe/Lisbon", "zurich": "Europe/Zurich", "switzerland": "Europe/Zurich",
    "vienna": "Europe/Vienna", "stockholm": "Europe/Stockholm", "oslo": "Europe/Oslo",
    "copenhagen": "Europe/Copenhagen", "helsinki": "Europe/Helsinki", "warsaw": "Europe/Warsaw",
    "poland": "Europe/Warsaw", "prague": "Europe/Prague", "athens": "Europe/Athens",
    "greece": "Europe/Athens", "istanbul": "Europe/Istanbul", "turkey": "Europe/Istanbul",
    "moscow": "Europe/Moscow", "kyiv": "Europe/Kyiv", "kiev": "Europe/Kyiv",
    "new york": "America/New_York", "nyc": "America/New_York", "boston": "America/New_York",
    "washington": "America/New_York", "miami": "America/New_York", "toronto": "America/Toronto",
    "chicago": "America/Chicago", "texas": "America/Chicago", "dallas": "America/Chicago",
    "houston": "America/Chicago", "denver": "America/Denver", "phoenix": "America/Phoenix",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles",
    "california": "America/Los_Angeles", "vancouver": "America/Vancouver",
    "mexico city": "America/Mexico_City", "mexico": "America/Mexico_City",
    "sao paulo": "America/Sao_Paulo", "brazil": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires", "argentina": "America/Argentina/Buenos_Aires",
    "lagos": "Africa/Lagos", "nigeria": "Africa/Lagos", "cairo": "Africa/Cairo",
    "egypt": "Africa/Cairo", "johannesburg": "Africa/Johannesburg",
    "south africa": "Africa/Johannesburg", "nairobi": "Africa/Nairobi", "kenya": "Africa/Nairobi",
    "dubai": "Asia/Dubai", "uae": "Asia/Dubai", "abu dhabi": "Asia/Dubai",
    "riyadh": "Asia/Riyadh", "saudi arabia": "Asia/Riyadh", "tehran": "Asia/Tehran",
    "karachi": "Asia/Karachi", "pakistan": "Asia/Karachi", "lahore": "Asia/Karachi",
    "india": "Asia/Kolkata", "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata", "bangalore": "Asia/Kolkata", "kolkata": "Asia/Kolkata",
    "dhaka": "Asia/Dhaka", "bangladesh": "Asia/Dhaka", "bangkok": "Asia/Bangkok",
    "thailand": "Asia/Bangkok", "jakarta": "Asia/Jakarta", "singapore": "Asia/Singapore",
    "kuala lumpur": "Asia/Kuala_Lumpur", "malaysia": "Asia/Kuala_Lumpur",
    "hong kong": "Asia/Hong_Kong", "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai",
    "china": "Asia/Shanghai", "taipei": "Asia/Taipei", "manila": "Asia/Manila",
    "philippines": "Asia/Manila", "seoul": "Asia/Seoul", "korea": "Asia/Seoul",
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo", "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne", "brisbane": "Australia/Brisbane",
    "perth": "Australia/Perth", "adelaide": "Australia/Adelaide",
    "auckland": "Pacific/Auckland", "new zealand": "Pacific/Auckland",
    "honolulu": "Pacific/Honolulu", "hawaii": "Pacific/Honolulu", "utc": "UTC", "gmt": "UTC",
}
_ZONE_ALT = "|".join(sorted((re.escape(z) for z in _ZONES), key=len, reverse=True))
#: Asking for the time there, and nothing else ("what time does the shop open
#: in Birmingham" is not this question).
_TIME_IN = re.compile(
    rf"^(?:what(?:'s| is) the (?:local |current )?time|what time is it|"
    rf"(?:the |current |local )?time)(?: now| right now)?\s+(?:in|at|over in)\s+"
    rf"(?P<place>{_ZONE_ALT})(?: right now| now)?$")
_PLACE_NAMES = {"uk": "the UK", "la": "LA", "nyc": "NYC", "utc": "UTC", "gmt": "GMT",
                "uae": "the UAE"}


def _place(name: str) -> str:
    return _PLACE_NAMES.get(name, name.title())
_TIME_CONVERT = re.compile(
    rf"(?P<hour>\d{{1,2}})(?::(?P<minute>\d{{2}}))?\s*(?P<ampm>am|pm)?\s*"
    rf"(?:(?P<src>{_ZONE_ALT})\s+time|in\s+(?P<src2>{_ZONE_ALT}))?\s+"
    rf"(?:is\s+(?:what\s+)?)?(?:in|to)\s+(?P<dst>{_ZONE_ALT})(?:\s+time)?\b")


def world_time(text: str) -> str | None:
    """'what time is it in Tokyo' / '3pm UK time in New York', or None."""
    t = str(text or "").lower().strip().rstrip("?.!")
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return None
    from datetime import datetime, timedelta
    try:
        local = datetime.now().astimezone()
        m = _TIME_CONVERT.search(t)
        if m and (m.group("src") or m.group("src2") or m.group("ampm")):
            hour, minute = int(m.group("hour")), int(m.group("minute") or 0)
            ampm = m.group("ampm")
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return None
            src_name = m.group("src") or m.group("src2")
            src_zone = ZoneInfo(_ZONES[src_name]) if src_name else local.tzinfo
            there = ZoneInfo(_ZONES[m.group("dst")])
            start = datetime.now(src_zone).replace(hour=hour, minute=minute, second=0,
                                                   microsecond=0)
            converted = start.astimezone(there)
            day = ""
            if converted.date() > start.date():
                day = " the next day"
            elif converted.date() < start.date():
                day = " the day before"
            src_label = (_place(src_name).removeprefix('the ') + ' time') if src_name else 'your time'
            return (f"{start.strftime('%H:%M')} {src_label} is {converted.strftime('%H:%M')} "
                    f"in {_place(m.group('dst'))}{day}.")
        m = _TIME_IN.search(t)
        if m:
            place = m.group("place")
            there = datetime.now(ZoneInfo(_ZONES[place]))
            offset = (there.utcoffset() - (local.utcoffset() or timedelta())).total_seconds() / 3600
            relation = ("the same as here" if abs(offset) < 0.01 else
                        f"{abs(offset):g} hour{'s' if abs(offset) != 1 else ''} "
                        f"{'ahead of' if offset > 0 else 'behind'} you")
            return (f"It's {there.strftime('%H:%M')} on {there.strftime('%A')} in "
                    f"{_place(place)} ({there.strftime('%Z')}), {relation}.")
    except Exception:
        return None
    return None


_COIN = re.compile(r"^(?:please\s+)?(?:flip|toss|spin)\s+a\s+coin|^heads or tails\b", re.I)
_DICE = re.compile(r"^(?:please\s+)?roll\s+(?:an?|one|(?P<count>\d|two|three|four|five|six))?\s*"
                   r"(?:d(?P<sides1>\d{1,3})s?|(?P<sides2>\d{1,3})[- ]sided\s+(?:dice|die)|dice|die)\b", re.I)
_PICK = re.compile(r"(?:pick|choose|give me|generate)\s+(?:me\s+)?a\s+random\s+number|"
                   r"(?:pick|choose)\s+a\s+number|random\s+number", re.I)
_RANGE = re.compile(r"(?:between|from)\s+(-?\d{1,9})\s+(?:and|to)\s+(-?\d{1,9})", re.I)
_SMALL = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def chance(text: str) -> str | None:
    """A coin, dice or a random number - from the operating system's secure
    random source, because a language model asked to "flip a coin" is not
    random at all. None when *text* asks for none of them."""
    import secrets
    t = str(text or "").strip().rstrip("?.!")
    if _COIN.search(t):
        return f"{secrets.choice(('Heads', 'Tails'))}."
    m = _DICE.search(t)
    if m:
        count_word = (m.group("count") or "").lower()
        count = int(count_word) if count_word.isdigit() else _SMALL.get(count_word, 1)
        sides = int(m.group("sides1") or m.group("sides2") or 6)
        if not (2 <= sides <= 1000 and 1 <= count <= 10):
            return None
        rolls = [secrets.randbelow(sides) + 1 for _ in range(count)]
        die = "dice" if sides == 6 else f"d{sides}"
        if count == 1:
            return f"You rolled {'an' if rolls[0] in (8, 11, 18) else 'a'} {rolls[0]}" + \
                   ("." if sides == 6 else f" on the {die}.")
        return f"You rolled {', '.join(map(str, rolls[:-1]))} and {rolls[-1]} — {sum(rolls)} in total."
    if _PICK.search(t):
        r = _RANGE.search(t)
        low, high = (int(r.group(1)), int(r.group(2))) if r else (1, 100)
        if low > high:
            low, high = high, low
        if high - low > 10**9:
            return None
        return f"{low + secrets.randbelow(high - low + 1)}" + ("" if r else " (between 1 and 100)") + "."
    return None


def calculate(text: str) -> ToolResult:
    """Answer a calculation or conversion exactly."""
    text = str(text or "").strip()
    if not text:
        return ToolResult("Give me a calculation or a conversion.", ok=False)
    luck = chance(text)
    if luck:
        return ToolResult(luck)
    zoned = world_time(text)
    if zoned:
        return ToolResult(zoned)
    from .dates import answer as date_answer
    dated = date_answer(text)
    if dated:
        return ToolResult(dated)
    try:
        converted = convert(text)
    except Exception:
        converted = None
    if converted:
        return ToolResult(converted, ok=not converted.startswith("I can't"))
    try:
        value = evaluate(text)
    except (CalcError, OverflowError, ValueError) as exc:
        worked = algebra(text)
        if worked:
            return ToolResult(worked)
        return ToolResult(f"I couldn't work that out exactly: {exc}.", ok=False)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    said = re.sub(r"^(?:what(?:'s| is)|calculate|work out|compute|how much is)\s+", "",
                  text.strip(), flags=re.IGNORECASE).rstrip("?.! =")
    shown = _fmt(value) if isinstance(value, float) else f"{value:,}"
    return ToolResult(f"{said} = {shown}")


__all__ = ["CalcError", "algebra", "calculate", "chance", "convert", "evaluate", "normalise_expression",
           "world_time"]
