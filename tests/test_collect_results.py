import importlib.util
import os

from conftest import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "collect_results", os.path.join(REPO_ROOT, "experiments", "reproduce_xxs_scaling", "collect_results.py")
)
collect_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect_results)

CURVES = {
    "Transformer++": [
        {"step": 14999, "valid_loss": 4.0, "valid_perplexity": 55.0},
        {"step": 29999, "valid_loss": 3.9, "valid_perplexity": 50.0},
    ],
    "EBT": [
        {"step": 14999, "valid_loss": 4.2, "valid_final_step_loss": 4.1, "valid_perplexity": 61.0},
        {"step": 29999, "valid_loss": 3.95, "valid_final_step_loss": 3.8, "valid_perplexity": 45.0},
    ],
}
BENCH = {"EBT": {"steps_per_s": 4.0, "tokens_per_s": 32768, "peak_mem_mib": 30000, "eta_hours_105k": 7.29}}


def test_read_valid_rows_merges_resumed_versions(tmp_path):
    v0, v1 = tmp_path / "v0.csv", tmp_path / "v1.csv"
    v0.write_text("epoch,step,train_loss,valid_loss,valid_final_step_loss,valid_perplexity\n0,49,7.1,,,\n0,14999,,4.20,4.10,61.0\n")
    v1.write_text("epoch,step,valid_loss,valid_final_step_loss,valid_perplexity\n0,14999,4.20,4.10,61.0\n0,29999,3.95,3.80,45.0\n")
    rows = collect_results.read_valid_rows([str(v1), str(v0)])
    assert [r["step"] for r in rows] == [14999, 29999]
    assert rows[1] == {"step": 29999, "valid_loss": 3.95, "valid_final_step_loss": 3.80, "valid_perplexity": 45.0}


def test_comparable_loss_uses_final_mcmc_step_for_ebt():
    assert collect_results.comparable_loss(CURVES["EBT"][1]) == 3.8
    assert collect_results.comparable_loss(CURVES["Transformer++"][1]) == 3.9


def test_result_md_has_only_tables_with_expected_rows():
    md = collect_results.build_result_md(CURVES, BENCH)
    assert "| Transformer++ | 29999 | 3.9000 | 49.40 | – | – | – | – |" in md
    assert "| EBT | 29999 | 3.8000 | 44.70 | 4.0 | 32768 | 30000 | 7.29 |" in md
    assert "| step | Transformer++ loss | Transformer++ ppl | EBT loss | EBT ppl |" in md
    assert "| 14999 | 4.0000 | 54.60 | 4.1000 | 60.34 |" in md
    prose = [l for l in md.splitlines() if l.strip() and not l.startswith(("|", "#"))]
    assert prose == []  # result.md contains headings and tables only


def test_result_md_survives_huge_loss():
    curves = {"EBT": [{"step": 30, "valid_loss": 3623.5, "valid_final_step_loss": 2150.0}]}
    md = collect_results.build_result_md(curves, {})
    assert "| EBT | 30 | 2150.0000 | inf | – | – | – | – |" in md
    assert "| 30 | 2150.0000 | inf |" in md
