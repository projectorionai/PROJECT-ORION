"""Validated structured workbook generation with atomic output."""

from __future__ import annotations

import asyncio
import math
import os
import re
import tempfile
from pathlib import Path

from ..data import ToolResult
from . import runtime

_FORMULA = re.compile(r"^=(SUM|AVERAGE|MIN|MAX|COUNT)\([A-Z]{1,3}[1-9][0-9]{0,5}:[A-Z]{1,3}[1-9][0-9]{0,5}\)$")


def create_workbook(spec: dict, filename: str) -> Path:
    from openpyxl import Workbook, load_workbook
    from openpyxl.chart import BarChart, LineChart, Reference
    from openpyxl.styles import Alignment, Font, PatternFill

    if not isinstance(spec, dict):
        raise TypeError("Workbook specification must be an object")
    sheets = spec.get("sheets")
    if not isinstance(sheets, list) or not 1 <= len(sheets) <= 12:
        raise ValueError("Supply between one and twelve sheets")
    clean_name = re.sub(r"[^\w .-]+", "_", Path(filename or "ORION workbook").name).strip(" .")[:100]
    if not clean_name:
        clean_name = "ORION workbook"
    clean_name = Path(clean_name).stem + ".xlsx"
    output = runtime.context().directory / "exports" / clean_name
    if output.exists():
        raise ValueError(f"{clean_name} already exists; choose another filename")
    book = Workbook()
    book.remove(book.active)
    names = set()
    for index, sheet in enumerate(sheets):
        if not isinstance(sheet, dict):
            raise TypeError("Each sheet must be an object")
        title = re.sub(r"[\\/*?:\[\]]", "_", str(sheet.get("name") or f"Sheet {index+1}"))[:31]
        if not title or title.casefold() in names:
            raise ValueError("Sheet names must be distinct and non-empty")
        names.add(title.casefold())
        columns = sheet.get("columns")
        rows = sheet.get("rows", [])
        if not isinstance(columns, list) or not 1 <= len(columns) <= 50:
            raise ValueError("Each sheet needs one to fifty column headings")
        if not isinstance(rows, list) or len(rows) > 5000:
            raise ValueError("A sheet may contain at most 5,000 rows")
        ws = book.create_sheet(title)
        headings = [str(h)[:300] for h in columns]
        ws.append(["'" + h if h.startswith(("=", "+", "-", "@")) else h for h in headings])
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError("Every row must have the same number of cells as its headings")
            values = []
            for value in row:
                if isinstance(value, dict):
                    formula = str(value.get("formula") or "").upper()
                    if not _FORMULA.fullmatch(formula):
                        raise ValueError("Only explicit SUM, AVERAGE, MIN, MAX and COUNT range formulas are supported")
                    values.append(formula)
                elif value is None or isinstance(value, (str, bool, int, float)):
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError("Workbook numbers must be finite")
                    if isinstance(value, str):
                        value = value[:32000]
                        if value.startswith(("=", "+", "-", "@")):
                            value = "'" + value
                    values.append(value)
                else:
                    raise ValueError("Unsupported workbook cell value")
            ws.append(values)
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="8C263C")
            cell.alignment = Alignment(vertical="center")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for column in ws.columns:
            ws.column_dimensions[column[0].column_letter].width = min(45, max(14, max(len(str(c.value or "")) for c in column[:80]) + 2))
        chart_spec = sheet.get("chart")
        if chart_spec is not None:
            if not isinstance(chart_spec, dict):
                raise ValueError("Chart specification must be an object")
            kind = chart_spec.get("type", "bar")
            if kind not in {"bar", "line"} or len(columns) < 2 or not rows:
                raise ValueError("A bar or line chart needs categories and at least one data column")
            chart = BarChart() if kind == "bar" else LineChart()
            chart.title = str(chart_spec.get("title") or title)[:200]
            chart.add_data(Reference(ws, min_col=2, max_col=len(columns), min_row=1, max_row=len(rows)+1), titles_from_data=True)
            chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(rows)+1))
            ws.add_chart(chart, f"A{len(rows)+4}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(suffix=".xlsx", dir=output.parent)
    os.close(fd)
    staging = Path(raw)
    try:
        book.save(staging)
        verification = load_workbook(staging, read_only=True)
        try:
            if len(verification.sheetnames) != len(sheets):
                raise RuntimeError("Workbook verification failed")
        finally:
            verification.close()
        # Same-directory hard linking publishes the verified file atomically
        # and refuses overwrites even if another request won the race.
        os.link(staging, output)
    finally:
        book.close()
        staging.unlink(missing_ok=True)
    return output


async def excel_writer(args: dict) -> ToolResult:
    spec = args.get("workbook")
    if spec is None:
        request = str(args.get("request") or "").strip()
        if not request:
            return ToolResult("Supply a workbook specification or describe the workbook and its source data.", ok=False)
        spec = await runtime.text_json(
            "Create a workbook specification for this request. Never invent figures or facts. "
            "Use blank cells for unknown values; include sample data only if explicitly requested and label the sheet Sample. "
            'JSON: {"sheets":[{"name":"...","columns":["..."],"rows":[["..."]],'
            '"chart":{"type":"bar|line","title":"..."}}]}. A chart is optional. '
            "A formula cell is an object such as {\"formula\":\"=SUM(B2:B8)\"}; only simple SUM/AVERAGE/MIN/MAX/COUNT ranges.\n" + request[:15000],
            task="excel_writer")
    output = await asyncio.to_thread(create_workbook, spec, str(args.get("filename") or "ORION workbook"))
    runtime.emit("jv_workbook", {"path": str(output), "name": output.name})
    return ToolResult(f"Workbook created and verified: {output}", evidence=[{"path": str(output), "format": "xlsx"}])
