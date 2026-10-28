"""Pure helpers of scripts/ieee_integrity_check.py (no data files, no server)."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ieee_integrity_check import DSU, archived_dsu, collision_groups, point_id, same_work  # noqa: E402


def test_point_id_matches_dedup_rule():
    assert point_id("ieee", "IEEE Conferences", "t") == str(uuid.uuid5(uuid.NAMESPACE_URL, "ieee:IEEE Conferences"))
    assert point_id("ieee", None, "a" * 100) == str(uuid.uuid5(uuid.NAMESPACE_URL, "ieee:" + "a" * 80))


def test_collision_groups_counts_only_shared_ids():
    assert collision_groups(["a", "a", "b", "c", "c", "c"]) == {"a": 2, "c": 3}


def test_same_work_by_doi_then_title():
    assert same_work({"doi": "10.1/X", "title_norm": "x"}, {"doi": "10.1/x", "title_norm": "y"}) == "doi"
    assert same_work({"doi": None, "title_norm": "same"}, {"doi": "10.1/z", "title_norm": "same"}) == "title"
    assert same_work({"doi": "10.1/a", "title_norm": "p"}, {"doi": "10.1/b", "title_norm": "q"}) == "no"


def test_archived_dsu_merges_only_duplicates_transitively():
    pairs = [{"a_cluster": "a", "b_cluster": "b", "judge_verdict": "DUPLICATE"},
             {"a_cluster": "b", "b_cluster": "c", "judge_verdict": "DUPLICATE"},
             {"a_cluster": "d", "b_cluster": "e", "judge_verdict": "UNCLEAR"}]
    d = archived_dsu(pairs, ["a", "b", "c", "d", "e"])
    assert d.n_components(["a", "b", "c", "d", "e"]) == 3
    assert isinstance(d, DSU)
