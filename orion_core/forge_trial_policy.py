"""Source admission policy for reserved ``trial_`` analysis tools.

This deliberately small Python subset removes I/O and reflective capabilities
from generated trials. It is a fail-closed admission filter, not a Python or OS
sandbox: verifier isolation, resource budgets and human review still matter.
Ordinary Forge tools keep their existing policy.
"""

from __future__ import annotations

import ast

from .security import SecurityViolation

MAX_SOURCE_CHARACTERS = 60000
MAX_AST_NODES = 12000

# Permit exports, rather than merely permitting a module: json.codecs,
# statistics.sys and functools.types would otherwise reintroduce capabilities.
SAFE_EXPORTS: dict[str, frozenset[str]] = {
    "__future__": frozenset({"annotations"}),
    "math": frozenset(
        [
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atan2",
            "atanh",
            "ceil",
            "comb",
            "copysign",
            "cos",
            "cosh",
            "degrees",
            "dist",
            "e",
            "erf",
            "erfc",
            "exp",
            "exp2",
            "expm1",
            "fabs",
            "factorial",
            "floor",
            "fmod",
            "frexp",
            "fsum",
            "gamma",
            "gcd",
            "hypot",
            "inf",
            "isclose",
            "isfinite",
            "isinf",
            "isnan",
            "isqrt",
            "lcm",
            "ldexp",
            "lgamma",
            "log",
            "log10",
            "log1p",
            "log2",
            "modf",
            "nan",
            "nextafter",
            "perm",
            "pi",
            "pow",
            "prod",
            "radians",
            "remainder",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "tau",
            "trunc",
            "ulp",
        ]
    ),
    "statistics": frozenset(
        [
            "mean",
            "fmean",
            "geometric_mean",
            "harmonic_mean",
            "median",
            "median_low",
            "median_high",
            "median_grouped",
            "mode",
            "multimode",
            "quantiles",
            "pvariance",
            "variance",
            "pstdev",
            "stdev",
            "covariance",
            "correlation",
            "linear_regression",
            "StatisticsError",
        ]
    ),
    "collections": frozenset(
        ["Counter", "deque", "defaultdict", "OrderedDict", "ChainMap"]
    ),
    "itertools": frozenset(
        [
            "accumulate",
            "chain",
            "combinations",
            "combinations_with_replacement",
            "compress",
            "count",
            "cycle",
            "dropwhile",
            "filterfalse",
            "groupby",
            "islice",
            "pairwise",
            "permutations",
            "product",
            "repeat",
            "starmap",
            "takewhile",
            "tee",
            "zip_longest",
        ]
    ),
    "functools": frozenset(["reduce", "lru_cache", "cache", "partial", "cmp_to_key"]),
    "re": frozenset(
        [
            "compile",
            "escape",
            "findall",
            "finditer",
            "fullmatch",
            "match",
            "search",
            "split",
            "sub",
            "subn",
            "ASCII",
            "IGNORECASE",
            "MULTILINE",
            "DOTALL",
            "UNICODE",
            "VERBOSE",
            "NOFLAG",
        ]
    ),
    "json": frozenset({"loads", "dumps", "JSONDecodeError"}),
    "string": frozenset(
        [
            "ascii_letters",
            "ascii_lowercase",
            "ascii_uppercase",
            "digits",
            "hexdigits",
            "octdigits",
            "punctuation",
            "printable",
            "whitespace",
            "capwords",
        ]
    ),
    "unicodedata": frozenset(
        [
            "lookup",
            "name",
            "decimal",
            "digit",
            "numeric",
            "category",
            "bidirectional",
            "combining",
            "east_asian_width",
            "mirrored",
            "decomposition",
            "normalize",
            "is_normalized",
            "unidata_version",
        ]
    ),
    "decimal": frozenset({"Decimal", "InvalidOperation"}),
    "fractions": frozenset({"Fraction"}),
    "datetime": frozenset({"date", "datetime", "time", "timedelta", "timezone"}),
    "typing": frozenset(
        [
            "Any",
            "Iterable",
            "Sequence",
            "Mapping",
            "MutableMapping",
            "Optional",
            "Union",
            "Literal",
            "List",
            "Dict",
            "Tuple",
            "Set",
            "Callable",
            "Iterator",
        ]
    ),
}

SAFE_BUILTINS = frozenset(
    [
        "abs",
        "all",
        "any",
        "bool",
        "callable",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "Exception",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "ArithmeticError",
        "StopIteration",
        "AssertionError",
    ]
)
FORBIDDEN_NAMES = frozenset(
    [
        "eval",
        "exec",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "__import__",
        "builtins",
        "importlib",
        "sys",
        "os",
        "object",
        "type",
        "super",
        "memoryview",
        "help",
        "exit",
        "quit",
        "classmethod",
        "staticmethod",
        "property",
    ]
)

# Value operations only; no stream/file methods, dynamic formatting, module
# namespaces, frames, function globals or other reflective attributes.
VALUE_ATTRIBUTES = frozenset(
    [
        "append",
        "extend",
        "insert",
        "pop",
        "clear",
        "copy",
        "count",
        "index",
        "reverse",
        "sort",
        "get",
        "items",
        "keys",
        "values",
        "setdefault",
        "update",
        "popitem",
        "fromkeys",
        "add",
        "discard",
        "remove",
        "difference",
        "difference_update",
        "intersection",
        "intersection_update",
        "union",
        "symmetric_difference",
        "symmetric_difference_update",
        "issubset",
        "issuperset",
        "isdisjoint",
        "most_common",
        "elements",
        "total",
        "popleft",
        "appendleft",
        "extendleft",
        "rotate",
        "maxlen",
        "join",
        "split",
        "rsplit",
        "splitlines",
        "strip",
        "lstrip",
        "rstrip",
        "lower",
        "upper",
        "casefold",
        "title",
        "capitalize",
        "swapcase",
        "replace",
        "startswith",
        "endswith",
        "find",
        "rfind",
        "rindex",
        "partition",
        "rpartition",
        "removeprefix",
        "removesuffix",
        "isalnum",
        "isalpha",
        "isascii",
        "isdecimal",
        "isdigit",
        "isidentifier",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
        "center",
        "ljust",
        "rjust",
        "zfill",
        "expandtabs",
        "encode",
        "decode",
        "translate",
        "maketrans",
        "hex",
        "fromhex",
        "bit_length",
        "bit_count",
        "numerator",
        "denominator",
        "real",
        "imag",
        "conjugate",
        "as_integer_ratio",
        "is_integer",
        "to_bytes",
        "from_bytes",
        "quantize",
        "normalize",
        "as_tuple",
        "is_finite",
        "is_infinite",
        "is_nan",
        "sqrt",
        "ln",
        "log10",
        "exp",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "tzinfo",
        "fold",
        "days",
        "seconds",
        "microseconds",
        "today",
        "now",
        "utcnow",
        "fromisoformat",
        "isoformat",
        "strptime",
        "strftime",
        "weekday",
        "isoweekday",
        "isocalendar",
        "date",
        "time",
        "timetz",
        "timestamp",
        "total_seconds",
        "astimezone",
        "utcoffset",
        "dst",
        "tzname",
        "combine",
        "group",
        "groups",
        "groupdict",
        "start",
        "end",
        "span",
        "pattern",
        "flags",
        "fullmatch",
        "match",
        "search",
        "findall",
        "finditer",
        "sub",
        "subn",
        "union",
        "parents",
        "maps",
        "new_child",
    ]
)


def generation_guidance(tool_name: str) -> str:
    """Explain the enforced subset before a provider spends tokens on code."""
    exports = "; ".join(
        f"{module}: {', '.join(sorted(names))}"
        for module, names in SAFE_EXPORTS.items()
    )
    return (
        "ENFORCED ANALYSIS TRIAL POLICY: use pure functions and JSON/value data only. "
        "No requirements, file/network/process/device access, dynamic imports, reflection, "
        "eval/exec/open, classes or dynamically selected callables. Use f-strings instead "
        "of str.format/format_map. Access imported modules only through their approved "
        "exports; do not pass module objects around. Include explicit meaningful tests "
        f"importing run/get_tool_schema from {tool_name}_tool; pytest.approx and pytest.raises "
        "are the only permitted pytest exports. Tests have the same restrictions. "
        f"Approved standard-library exports: {exports}."
    )


def guard_trial_source(
    source: str, tool_name: str, *, test: bool = False, context: str = "forge.trial"
) -> str:
    """Reject capabilities outside the analysis subset before any execution.

    Both the module and its tests are checked. The exact generated module may
    be imported by its tests; other local/generated modules cannot be imported.
    Size, syntax and parser failures reject a trial rather than skipping checks.
    """

    def refuse(reason: str) -> None:
        raise SecurityViolation(f"blocked unsafe {context}: analysis trial {reason}")

    if not isinstance(source, str) or not source.strip():
        refuse("requires non-empty source and an explicit acceptance test")
    if len(source) > MAX_SOURCE_CHARACTERS:
        refuse("exceeds the source size budget")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, RecursionError) as exc:
        refuse(f"cannot be parsed ({type(exc).__name__})")
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        refuse("exceeds the AST size budget")
    if test and not any(isinstance(node, ast.Assert) for node in nodes):
        refuse("requires explicit acceptance assertions")
    parents = {
        child: parent for parent in nodes for child in ast.iter_child_nodes(parent)
    }
    exports = dict(SAFE_EXPORTS)
    if test:
        exports.update(
            {
                tool_name: frozenset({"run", "get_tool_schema"}),
                tool_name + "_tool": frozenset({"run", "get_tool_schema"}),
                "pytest": frozenset({"approx", "raises"}),
            }
        )
    modules: dict[str, str] = {}
    imported: set[str] = set()
    functions = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in exports or alias.name == "__future__":
                    refuse(f"does not permit import '{alias.name}'")
                local = alias.asname or alias.name
                if local in FORBIDDEN_NAMES or local.startswith("__"):
                    refuse("does not permit reserved import aliases")
                modules[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in exports:
                refuse(f"does not permit import from '{node.module or '.'}'")
            for alias in node.names:
                if alias.name not in exports[node.module]:
                    refuse(f"does not permit export '{node.module}.{alias.name}'")
                local = alias.asname or alias.name
                if local in FORBIDDEN_NAMES or local.startswith("__"):
                    refuse("does not permit reserved import aliases")
                imported.add(local)

    for node in nodes:
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                refuse(f"does not permit reflective/capability name '{node.id}'")
            if node.id == "compile" and node.id not in imported:
                refuse("does not permit Python compilation")
            if node.id in modules:
                parent = parents.get(node)
                if (
                    not isinstance(parent, ast.Attribute)
                    or parent.value is not node
                    or parent.attr not in exports[modules[node.id]]
                ):
                    refuse("does not permit module aliasing or namespace access")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                refuse("does not permit private or reflective attribute access")
            if isinstance(node.value, ast.Name) and node.value.id in modules:
                if node.attr not in exports[modules[node.value.id]]:
                    refuse(
                        f"does not permit export '{modules[node.value.id]}.{node.attr}'"
                    )
            elif node.attr not in VALUE_ATTRIBUTES:
                refuse(f"does not permit attribute '{node.attr}'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS | imported | functions:
                    refuse(f"does not permit dynamic/unknown callable '{node.func.id}'")
                if node.func.id == "print" and any(
                    kw.arg == "file" for kw in node.keywords
                ):
                    refuse("does not permit redirecting print to a stream")
            elif not isinstance(node.func, ast.Attribute):
                refuse("does not permit dynamically selected callables")
        elif isinstance(
            node,
            (
                ast.ClassDef,
                ast.Global,
                ast.Nonlocal,
                ast.AsyncFunctionDef,
                ast.Await,
                ast.AsyncWith,
                ast.AsyncFor,
            ),
        ):
            refuse(f"does not permit {type(node).__name__}")
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("__"):
            refuse("does not permit special methods")
    return source
