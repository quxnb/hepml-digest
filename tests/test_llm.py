from types import SimpleNamespace

from hepml_digest.cli import _demo_papers
import pytest

from hepml_digest.llm import APICallBudgetExceeded, DeepSeekAnalyzer, DemoAnalyzer, OutputLimitExceeded
from hepml_digest.config import Settings


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


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "https://api.deepseek.com"),
        ("", "https://api.deepseek.com"),
        ("   ", "https://api.deepseek.com"),
        ("https://relay.example/v1", "https://relay.example/v1/"),
        (" https://relay.example/v1/ ", "https://relay.example/v1/"),
    ],
)
def test_deepseek_analyzer_uses_configured_endpoint(monkeypatch, base_url, expected):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "relay-test-key")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    if base_url is not None:
        monkeypatch.setenv("DEEPSEEK_BASE_URL", base_url)

    analyzer = DeepSeekAnalyzer(
        "relay-screen-model",
        "relay-review-model",
        "screen prompt",
        "review prompt",
    )

    assert str(analyzer.client.base_url) == expected
    assert analyzer.client.api_key == "relay-test-key"
    assert analyzer.screening_model == "relay-screen-model"
    assert analyzer.review_model == "relay-review-model"


def test_deepseek_analyzer_requires_key_for_relay(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://relay.example/v1")

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is missing"):
        DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")


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

    assert analyzer.usage.screening.calls == 1
    assert analyzer.usage.prompt_tokens == 120
    assert analyzer.usage.screening.cache_hit_tokens == 100
    assert analyzer.usage.screening.cache_miss_tokens == 20
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
    analyzer.request_guard = lambda: False

    with pytest.raises(APICallBudgetExceeded):
        analyzer.screen(_demo_papers()[0])

    assert client.completions.calls == 0
    assert analyzer.usage.screening.calls == 0


def test_attempt_count_can_stop_internal_retry(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = DeepSeekAnalyzer(
        "screen-model",
        "review-model",
        "screen prompt",
        "review prompt",
    )
    client = FailingClient()
    analyzer.client = client
    analyzer.request_guard = lambda: analyzer.usage.failed_calls < 1

    with pytest.raises(APICallBudgetExceeded):
        analyzer.screen(_demo_papers()[0])

    assert analyzer.usage.failed_calls == 1
    assert client.completions.calls == 1
    assert analyzer.usage.screening.calls == 1


def test_fingerprint_separates_services_and_revisions_but_not_keys(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "first-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://one.example/v1")
    monkeypatch.setenv("ANALYSIS_REVISION", "1")
    first = DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "rotated-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://one.example/v1/")
    rotated = DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")
    assert first.screening_fingerprint == rotated.screening_fingerprint
    assert first.review_fingerprint == rotated.review_fingerprint
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://two.example/v1")
    relay = DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")
    assert first.screening_fingerprint != relay.screening_fingerprint
    assert first.review_fingerprint != relay.review_fingerprint
    monkeypatch.setenv("ANALYSIS_REVISION", "2")
    revised = DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")
    assert relay.screening_fingerprint != revised.screening_fingerprint
    assert relay.review_fingerprint != revised.review_fingerprint


@pytest.mark.parametrize("deepseek_fields", [True, False])
def test_usage_normalizes_cache_fields_once(deepseek_fields):
    from hepml_digest.llm import StageUsage

    raw = SimpleNamespace(prompt_tokens=30, completion_tokens=7)
    if deepseek_fields:
        raw.prompt_cache_hit_tokens = 10
        raw.prompt_cache_miss_tokens = 20
    else:
        raw.prompt_tokens_details = SimpleNamespace(cached_tokens=10)
    usage = StageUsage()
    usage.add(raw)
    assert usage.cache_hit_tokens == 10
    assert usage.cache_miss_tokens == 20
    assert usage.cost(1, 2, 3) == pytest.approx(71 / 1_000_000)


def test_paid_invalid_response_is_counted_before_retry_budget_check(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")
    monkeypatch.setattr(DeepSeekAnalyzer._request_json.retry, "wait", lambda _: 0)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=250000),
            choices=[SimpleNamespace(message=SimpleNamespace(content="invalid JSON"))],
        )

    analyzer.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = Settings()
    analyzer.request_guard = lambda: analyzer.usage.cost(settings) < 1
    with pytest.raises(APICallBudgetExceeded):
        analyzer.screen(_demo_papers()[0])
    assert len(calls) == 1
    assert analyzer.usage.cost(settings) == 1
    assert analyzer.usage.failed_calls == 1


@pytest.mark.parametrize("change", ["schema", "thinking", "length"])
def test_fingerprint_covers_request_contract(change):
    from hepml_digest.llm import _analysis_fingerprint
    from hepml_digest.models import Review, Screening

    args = ["model", "prompt", "https://relay.example/v1", "1", Screening, False, 900]
    first = _analysis_fingerprint(*args)
    if change == "schema":
        args[4] = Review
    elif change == "thinking":
        args[5] = True
    else:
        args[6] = 901
    assert _analysis_fingerprint(*args) != first


@pytest.mark.parametrize("missing", ["all", "input", "output"])
def test_missing_usage_is_detected_at_the_api_boundary(monkeypatch, missing):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")
    fake = FakeCompletions()

    def create(**kwargs):
        response = fake.create(**kwargs)
        if missing == "all":
            response.usage = None
        else:
            delattr(response.usage, "prompt_tokens" if missing == "input" else "completion_tokens")
        return response

    analyzer.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    analyzer.screen(_demo_papers()[0])
    assert analyzer.usage.unmetered_calls == 1


@pytest.mark.parametrize("stage", ["screening", "review"])
@pytest.mark.parametrize("with_json", [False, True])
def test_truncated_response_is_metered_and_not_retried(monkeypatch, stage, with_json):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")
    monkeypatch.setattr(DeepSeekAnalyzer._request_json.retry, "wait", lambda _: 0)
    paper = _demo_papers()[0]
    screening = DemoAnalyzer().screen(paper)
    model = screening if stage == "screening" else DemoAnalyzer().review(paper, screening)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=2200,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=2100),
            ),
            choices=[SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content=model.model_dump_json() if with_json else None),
            )],
        )

    analyzer.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(OutputLimitExceeded):
        if stage == "screening":
            analyzer.screen(paper)
        else:
            analyzer.review(paper, screening)
    usage = getattr(analyzer.usage, stage)
    assert len(calls) == usage.calls == 1
    assert usage.last_finish_reason == "length"
    assert usage.reasoning_tokens == 2100
    assert analyzer.usage.completion_tokens == 2200
    assert analyzer.usage.cost(Settings()) > 0
    assert analyzer.usage.failed_calls == 1
    assert analyzer.usage.last_failure_kind == "output_limit"


def test_review_limits_and_effort_are_sent_and_fingerprinted_without_rescreening(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    default = DeepSeekAnalyzer("screen", "review", "screen prompt", "review prompt")
    limited = DeepSeekAnalyzer(
        "screen", "review", "screen prompt", "review prompt",
        review_max_tokens=4096,
    )
    stronger = DeepSeekAnalyzer(
        "screen", "review", "screen prompt", "review prompt",
        review_max_tokens=4096, review_reasoning_effort="high",
    )
    assert default.review_max_tokens == Settings().review_max_tokens == 8192
    assert default.review_reasoning_effort == Settings().review_reasoning_effort == "low"
    assert default.screening_fingerprint == limited.screening_fingerprint == stronger.screening_fingerprint
    assert len({default.review_fingerprint, limited.review_fingerprint, stronger.review_fingerprint}) == 3
    paper = _demo_papers()[0]
    screening = DemoAnalyzer().screen(paper)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        output = DemoAnalyzer().review(paper, screening) if kwargs["model"] == "review" else screening
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=100),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=output.model_dump_json()))],
        )

    stronger.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    stronger.review(paper, screening)
    stronger.screen(paper)
    assert calls[0]["max_tokens"] == 4096
    assert calls[0]["reasoning_effort"] == "high"
    assert calls[0]["extra_body"]["thinking"]["type"] == "enabled"
    assert "reasoning_effort" not in calls[1]
    assert stronger.usage.review.last_finish_reason == "stop"
    assert stronger.usage.screening.last_finish_reason == "stop"


def test_review_evidence_boundary_is_sent_and_changes_only_review_fingerprint(monkeypatch):
    from pathlib import Path

    from hepml_digest.config import DEFAULT_REQUEST_TIMEOUT_SECONDS

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    screen_prompt = Path("prompts/screen.txt").read_text(encoding="utf-8")
    review_prompt = Path("prompts/review.txt").read_text(encoding="utf-8")
    assert "摘要未说明不等于论文未做" in review_prompt
    assert "需查阅全文确认" in review_prompt
    assert "区分模拟结果与真实数据结果" in review_prompt
    prefix, boundary_and_rest = review_prompt.split("证据边界：", 1)
    _, rest = boundary_and_rest.split("只输出 JSON", 1)
    without_boundary = prefix + "只输出 JSON" + rest
    previous = DeepSeekAnalyzer("screen", "review", screen_prompt, without_boundary)
    current = DeepSeekAnalyzer("screen", "review", screen_prompt, review_prompt)
    assert current.client.timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert previous.screening_fingerprint == current.screening_fingerprint
    assert previous.review_fingerprint != current.review_fingerprint
    paper = _demo_papers()[0]
    screening = DemoAnalyzer().screen(paper)
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=100),
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=DemoAnalyzer().review(paper, screening).model_dump_json()),
            )],
        )

    current.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    current.review(paper, screening)
    assert sent[0]["messages"][0] == {"role": "system", "content": review_prompt}
