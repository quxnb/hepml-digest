import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hepml_digest.cli import _demo_papers, main
from hepml_digest.evaluation import EvaluationSet, evaluate, export_evaluation
from hepml_digest.llm import DemoAnalyzer
from hepml_digest.models import Record, State
from hepml_digest.store import save_state


def evaluation_state():
    state = State()
    analyzer = DemoAnalyzer()
    for index in range(1, 6):
        paper = _demo_papers()[0].model_copy(deep=True)
        paper.arxiv_id = f"2607.{index:05d}"
        paper.link = f"https://arxiv.org/abs/{paper.arxiv_id}"
        if index == 3:
            state.pending_papers[paper.version_key] = paper
            continue
        screening = analyzer.screen(paper)
        if index == 2:
            screening.evidence_level = "irrelevant"
        if index == 5:
            screening.relevance = 0.1
        state.records[paper.version_key] = Record(
            paper=paper, screening=screening,
            review=analyzer.review(paper, screening), review_status="complete",
            processed_at=paper.updated, screening_model="demo-screen",
            review_model="demo-review",
            screening_fingerprint=analyzer.screening_fingerprint,
            review_fingerprint=analyzer.review_fingerprint,
        )
    return state


def test_export_samples_recommended_rejected_and_unprocessed(tmp_path: Path):
    state = evaluation_state()
    dataset = export_evaluation(state, tmp_path / "labels.json", 0.55, limit=3)
    assert len(dataset.items) == 3
    assert dataset.items[0].screening.relevance >= 0.55
    assert (
        dataset.items[1].screening.relevance < 0.55
        or dataset.items[1].screening.evidence_level == "irrelevant"
    )
    assert dataset.items[2].screening is None
    assert all(item.human_keep is None for item in dataset.items)
    assert all(item.hep_mapping_overstated is None for item in dataset.items)


def test_export_does_not_overwrite_human_annotations(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text("human annotations", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_evaluation(evaluation_state(), path, 0.55)
    assert path.read_text(encoding="utf-8") == "human annotations"


def test_evaluation_counts_rejections_and_unprocessed_positive_labels(tmp_path: Path):
    state = evaluation_state()
    dataset = export_evaluation(state, tmp_path / "labels.json", 0.55)
    for item in dataset.items:
        item.human_keep = item.paper.arxiv_id in {"2607.00001", "2607.00002", "2607.00003"}
        if item.paper.arxiv_id in {"2607.00001", "2607.00002"}:
            item.hep_mapping_overstated = item.paper.arxiv_id == "2607.00002"
    result = evaluate(state, dataset, 0.55)
    assert result["labeled"] == 5
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 2
    assert result["true_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["sample_recall"] == pytest.approx(1 / 3)
    assert result["unprocessed_relevant"] == 1
    assert result["overstatement_rate"] == 0.5


def test_unlabeled_samples_do_not_produce_quality_claims(tmp_path: Path):
    state = evaluation_state()
    dataset = export_evaluation(state, tmp_path / "labels.json", 0.55)
    result = evaluate(state, dataset, 0.55)
    assert result["labeled"] == 0
    assert result["unlabeled"] == 5
    assert result["precision"] is None
    assert result["sample_recall"] is None
    assert result["overstatement_rate"] is None


def test_stale_mapping_labels_are_excluded_after_reanalysis(tmp_path: Path):
    state = evaluation_state()
    dataset = export_evaluation(state, tmp_path / "labels.json", 0.55)
    for item in dataset.items:
        if item.paper.arxiv_id == "2607.00001":
            item.hep_mapping_overstated = False
    state.records["2607.00001:v1"].review.summary_cn = "Changed review"
    result = evaluate(state, dataset, 0.55)
    assert result["mapping_labeled"] == 0
    assert result["stale_mapping_labels"] == 1
    assert result["overstatement_rate"] is None


def test_export_uses_latest_version_and_evaluation_skips_superseded_labels(tmp_path: Path):
    state = evaluation_state()
    dataset = export_evaluation(state, tmp_path / "old.json", 0.55)
    new_version = state.records["2607.00001:v1"].paper.model_copy(deep=True)
    new_version.version = 2
    state.pending_papers[new_version.version_key] = new_version
    del state.records["2607.00002:v1"]
    for item in dataset.items:
        item.human_keep = True
    result = evaluate(state, dataset, 0.55)
    assert result["superseded"] == 1
    assert result["missing"] == 1
    assert result["labeled"] == 3
    updated = export_evaluation(state, tmp_path / "new.json", 0.55)
    versions = {item.paper.version_key for item in updated.items}
    assert "2607.00001:v2" in versions
    assert "2607.00001:v1" not in versions


def test_evaluation_rejects_duplicate_versions_and_string_booleans(tmp_path: Path):
    dataset = export_evaluation(evaluation_state(), tmp_path / "labels.json", 0.55)
    data = dataset.model_dump(mode="json")
    data["items"].append(data["items"][0])
    with pytest.raises(ValidationError, match="duplicate paper versions"):
        EvaluationSet.model_validate(data)
    data["items"].pop()
    data["items"][0]["human_keep"] = "false"
    with pytest.raises(ValidationError):
        EvaluationSet.model_validate(data)


def test_evaluation_cli_is_offline_and_does_not_modify_state(tmp_path: Path, monkeypatch, capsys):
    state_file = tmp_path / "state.json"
    labels_file = tmp_path / "labels.json"
    save_state(state_file, evaluation_state())
    original = state_file.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not initialize a model client")

    monkeypatch.setattr("hepml_digest.cli.DeepSeekAnalyzer", forbidden)
    monkeypatch.setattr("sys.argv", [
        "hepml-digest", "--state-file", str(state_file),
        "--export-evaluation", str(labels_file),
    ])
    main()
    assert json.loads(capsys.readouterr().out) == {"exported": 5}
    monkeypatch.setattr("sys.argv", [
        "hepml-digest", "--state-file", str(state_file), "--evaluate", str(labels_file),
    ])
    main()
    assert json.loads(capsys.readouterr().out)["labeled"] == 0
    assert state_file.read_bytes() == original
