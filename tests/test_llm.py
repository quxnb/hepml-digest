from types import SimpleNamespace

from hepml_digest.cli import _demo_papers
import pytest

from hepml_digest.llm import APICallBudgetExceeded, DeepSeekAnalyzer


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        content = (
            '{"relevance":0.8,"method":"校准","hep_tasks":[], '
            '"reason":"可迁移","needs_deep_review":false,'
            '"evidence_level":"transferable"}'
        )
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=120,
                prompt_cache_hit_tokens=100,
                prompt_cache_miss_tokens=20,
                completion_tokens=30,
            ),
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content))
            ],
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class FailingCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise ValueError("invalid response")


class FailingClient:
    def __init__(self):
        self.completions = FailingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_deepseek_analyzer_tracks_usage_and_prompt_fingerprint(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = DeepSeekAnalyzer(
        "screen-model",
        "review-model",
        "screen prompt",
        "review prompt",
    )
    changed_prompt = DeepSeekAnalyzer(
        "screen-model",
        "review-model",
        "changed screen prompt",
        "review prompt",
    )
    assert analyzer.client.max_retries == 0
    analyzer.client = FakeClient()

    analyzer.screen(_demo_papers()[0])

    assert analyzer.usage.screening_calls == 1
    assert analyzer.usage.prompt_tokens == 120
    assert analyzer.usage.screening_cache_hit_tokens == 100
    assert analyzer.usage.screening_cache_miss_tokens == 20
    assert analyzer.usage.completion_tokens == 30
    assert (
        analyzer.screening_fingerprint
        != changed_prompt.screening_fingerprint
    )
    assert analyzer.review_fingerprint == changed_prompt.review_fingerprint


def test_request_guard_blocks_request_without_retry(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = DeepSeekAnalyzer(
        "screen-model",
        "review-model",
        "screen prompt",
        "review prompt",
    )
    client = FakeClient()
    analyzer.client = client
    analyzer.set_request_guard(lambda: False)

    with pytest.raises(APICallBudgetExceeded):
        analyzer.screen(_demo_papers()[0])

    assert client.completions.calls == 0
    assert analyzer.usage.screening_calls == 0


def test_failure_callback_can_stop_internal_retry(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = DeepSeekAnalyzer(
        "screen-model",
        "review-model",
        "screen prompt",
        "review prompt",
    )
    client = FailingClient()
    analyzer.client = client
    failures = [0]

    def record_failure():
        failures[0] += 1

    analyzer.set_request_guard(lambda: failures[0] < 1)
    analyzer.set_request_failure_callback(record_failure)

    with pytest.raises(APICallBudgetExceeded):
        analyzer.screen(_demo_papers()[0])

    assert failures == [1]
    assert client.completions.calls == 1
    assert analyzer.usage.screening_calls == 1
