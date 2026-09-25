"""Inventory deletions require intentional review; additions remain visible."""
import json
from pathlib import Path

import pytest

from tools.check_capabilities import compare, inventory

ROOT = Path(__file__).resolve().parents[1]


def test_release_retains_recorded_capabilities():
    baseline = json.loads((ROOT / "docs/capability_baseline.json").read_text())
    result = compare(baseline, inventory(ROOT, baseline))
    assert result["ok"], result


@pytest.mark.parametrize("category", ["tools", "handlers", "pages", "plugins", "entrypoints", "fallbacks"])
def test_missing_capability_blocks_release(category):
    assert not compare({category: ["required"]}, {category: []})["ok"]


def test_additions_are_reported_without_erasing_expected_capabilities():
    result = compare({"tools": ["one"]}, {"tools": ["one", "two"]})
    assert result["ok"] and result["added"] == {"tools": ["two"]}


def test_duplicate_names_and_local_plugin_loss_are_detected():
    assert not compare({"tools": ["one"]}, {"tools": ["one", "one"]})["ok"]
    assert not compare({"local_plugins": ["one"]}, {"local_plugins": []}, include_local=True)["ok"]
    assert compare({"local_plugins": ["one"]}, {"local_plugins": []})["ok"]
