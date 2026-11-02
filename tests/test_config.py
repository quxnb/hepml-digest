import pytest

from hepml_digest.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_candidates", -1),
        ("max_deep_reviews", -1),
        ("feed_max_items", -1),
        ("publish_threshold", 1.1),
        ("publish_threshold", float("nan")),
        ("review_threshold", -0.1),
        ("request_timeout_seconds", 0),
        ("request_timeout_seconds", float("inf")),
        ("api_time_budget_seconds", float("nan")),
        ("max_consecutive_api_failures", 0),
        ("max_prompt_tokens", -1),
        ("reanalysis_slots", -1),
        ("checkpoint_interval", 0),
        ("backfill_lookback_days", 0),
        ("backfill_max_pages", 0),
        ("backfill_max_pages", float("inf")),
        ("screening_budget_fraction", 0),
        ("screening_budget_fraction", 1.1),
        ("screening_budget_fraction", float("nan")),
        ("aging_slots", -1),
        ("max_paper_failures", 0),
        ("alert_after_runs", 0),
        ("alert_backlog_days", 0),
        ("review_max_tokens", 0),
        ("review_max_tokens", -1),
        ("review_reasoning_effort", "none"),
        ("review_reasoning_effort", "unknown"),
        ("llm_api_style", "auto"),
        ("llm_api_style", ""),
    ],
)
def test_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        Settings(**{field: value})


def test_env_loader_uses_declared_defaults_and_preserves_aliases(monkeypatch, tmp_path):
    from dataclasses import fields

    aliases = {"categories": "ARXIV_CATEGORIES", "user_agent": "ARXIV_USER_AGENT"}
    for item in fields(Settings):
        monkeypatch.delenv(aliases.get(item.name, item.name.upper()), raising=False)
    assert Settings.from_env() == Settings()
    for name, value in {
        "ARXIV_CATEGORIES": " cs.LG, hep-ex, ",
        "ARXIV_USER_AGENT": "test-agent",
        "SITE_URL": "https://example.test/digest/",
        "FEEDBACK_REPOSITORY": "/example/digest/",
        "STATE_FILE": str(tmp_path / "state.json"),
        "MAX_CANDIDATES": "7",
        "REQUEST_TIMEOUT_SECONDS": "1.5",
        "AGING_SLOTS": "2",
        "REVIEW_MAX_TOKENS": "4096",
        "REVIEW_REASONING_EFFORT": "high",
        "LLM_API_STYLE": "openai",
    }.items():
        monkeypatch.setenv(name, value)
    settings = Settings.from_env()
    assert settings.categories == ("cs.LG", "hep-ex")
    assert settings.user_agent == "test-agent"
    assert settings.site_url == "https://example.test/digest"
    assert settings.feedback_repository == "example/digest"
    assert settings.state_file == tmp_path / "state.json"
    assert settings.max_candidates == 7
    assert settings.request_timeout_seconds == 1.5
    assert settings.aging_slots == 2
    assert settings.review_max_tokens == 4096
    assert settings.review_reasoning_effort == "high"
    assert settings.llm_api_style == "openai"


def test_request_timeout_defaults_match_local_example_and_workflow(monkeypatch):
    from pathlib import Path

    from hepml_digest.config import DEFAULT_REQUEST_TIMEOUT_SECONDS

    monkeypatch.delenv("REQUEST_TIMEOUT_SECONDS", raising=False)
    assert Settings().request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS == 120
    assert Settings.from_env().request_timeout_seconds == 120
    assert "REQUEST_TIMEOUT_SECONDS=120" in Path(".env.example").read_text().splitlines()
    workflow = Path(".github/workflows/daily.yml").read_text()
    assert "REQUEST_TIMEOUT_SECONDS: ${{ vars.REQUEST_TIMEOUT_SECONDS || '120' }}" in workflow
