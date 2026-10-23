"""The evaluation runner: the dev cases keep meeting their thresholds, the held-out
seal and the number checks catch tampering, and RESULTS.md comes only from result files."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


@pytest.fixture(scope="module")
def dev_run():
    return run_eval.evaluate("dev", "rules", 1, "rules", "low", 0, 0.0)


def test_dev_cases_meet_every_threshold(dev_run):
    unmet = {name: entry["failures"] for name, entry in dev_run["metrics"].items() if entry["met"] is False}
    assert unmet == {}
    assert dev_run["report_only"]["model_calls_per_question"] == 0


def test_injection_cases_reach_the_planted_note(dev_run):
    runs = [case["runs"][0] for case in dev_run["cases"] if case["expect"].get("injection")]
    answered = [run for run in runs if run["kind"] == "answer"]
    assert answered and all(run["planted_note_would_rank"] and not run["planted_note_retrieved"] for run in answered)


def test_the_run_restores_the_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-key")
    monkeypatch.setattr(run_eval, "load_cases", lambda name: ([], "digest"))
    monkeypatch.setattr(run_eval, "summarize_run", lambda outcomes, thresholds, run: run)
    run_eval.evaluate("dev", "rules", 1, "rules", "low", 0, 0.0)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-real-key"


def test_the_sealed_heldout_file_is_unchanged():
    digest = hashlib.sha256((ROOT / "eval" / "heldout_cases.jsonl").read_bytes()).hexdigest()
    assert digest == (ROOT / "eval" / "heldout_cases.sha256").read_text().split()[0]


def test_a_changed_heldout_file_is_refused(tmp_path, monkeypatch):
    cases = tmp_path / "heldout_cases.jsonl"
    cases.write_text('{"id": "h01", "question": "q", "expect": {"kind": "decline"}}\n')
    seal = tmp_path / "heldout_cases.sha256"
    seal.write_text(hashlib.sha256(cases.read_bytes()).hexdigest() + "  heldout_cases.jsonl\n")
    monkeypatch.setitem(run_eval.CASE_FILES, "heldout", cases)
    monkeypatch.setattr(run_eval, "HELDOUT_HASH", seal)
    assert len(run_eval.load_cases("heldout")[0]) == 1
    cases.write_text(cases.read_text().replace('"q"', '"an easier question"'))
    with pytest.raises(SystemExit, match="sealed hash"):
        run_eval.load_cases("heldout")


def test_number_checks_catch_a_wrong_expectation(fixture_dir):
    case = json.loads((ROOT / "eval" / "dev_cases.jsonl").read_text().splitlines()[0])
    assert run_eval.numbers_check(case, fixture_dir) == []
    case["expect"]["numbers"]["total_b"] = 1226
    assert run_eval.numbers_check(case, fixture_dir) == ["total_b: 1225, expected 1226 (by hand)"]


def test_the_report_is_built_from_result_files(dev_run, tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path)
    (tmp_path / "results").mkdir()
    assert "No runs yet." in run_eval.write_report().read_text()
    (tmp_path / "results" / "run.json").write_text(json.dumps(dev_run))
    report = run_eval.write_report().read_text()
    assert "## " + dev_run["run"]["date"] + " · dev cases · rules and templates only (no model)" in report
    assert "| tool_selection | 100% (15/15) | 90% | yes |" in report
