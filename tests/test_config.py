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
        ("screening_cache_hit_price_per_million_rmb", -0.01),
        ("review_output_price_per_million_rmb", float("inf")),
    ],
)
def test_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        Settings(**{field: value})


def test_settings_reject_review_minimum_above_maximum():
    with pytest.raises(ValueError):
        Settings(min_deep_reviews=6, max_deep_reviews=5)
