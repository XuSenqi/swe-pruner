from unittest.mock import MagicMock

from minisweagent.agents.default import (
    DefaultAgent,
    RepeatedAction,
    _extract_read_paths,
    _is_read_command,
    _prune_fallback_reason,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel
from minisweagent.utils.pruner import PruneResponse, PrunerConfig


def test_extract_read_paths_normalizes_testbed_prefix():
    command = "cat -n /testbed/astropy/io/ascii/rst.py"
    assert _extract_read_paths(command) == ["astropy/io/ascii/rst.py"]


def test_is_read_command_ignores_edits():
    assert not _is_read_command("sed -i 's/a/b/' foo.py")
    assert _is_read_command("cat -n foo.py")


def test_prune_fallback_reason_low_score():
    config = PrunerConfig(url="http://example.com", threshold=0.5, min_keep_ratio=0.35)
    result = PruneResponse(
        score=0.006,
        pruned_code="x",
        token_scores=[],
        kept_frags=[1],
        origin_token_cnt=1000,
        left_token_cnt=100,
        model_input_token_cnt=1100,
    )
    assert _prune_fallback_reason(result, config) == "low_score"


def test_prune_fallback_reason_low_keep_ratio():
    config = PrunerConfig(url="http://example.com", threshold=0.5, min_keep_ratio=0.35, min_output_chars=2400)
    result = PruneResponse(
        score=0.99,
        pruned_code="x",
        token_scores=[],
        kept_frags=[1],
        origin_token_cnt=1200,
        left_token_cnt=300,
        model_input_token_cnt=1300,
    )
    assert _prune_fallback_reason(result, config) == "low_keep_ratio"


def test_apply_pruner_skips_small_output_before_call():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        pruner={
            "url": "http://example.com",
            "threshold": 0.5,
            "min_output_chars": 2400,
        },
    )
    mock_prune = MagicMock()
    agent.pruner_client.prune = mock_prune

    action = {"action": "cat -n foo.py", "context_focus_question": "What is foo?"}
    output = {"output": "x" * 1200}
    agent._apply_pruner(action, output)

    mock_prune.assert_not_called()
    assert output["output"] == "x" * 1200
    assert output["cfq_stats"]["prune_skipped"] == "small_output"


def test_apply_pruner_does_not_generate_cfq_on_small_output():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        pruner={"url": "http://example.com", "threshold": 0.5, "min_output_chars": 1600},
        cfq_generator={
            "url": "http://example.com",
            "model": "test-model",
            "min_output_chars": 3200,
        },
    )
    agent.cfq_generator.generate = MagicMock(return_value="What is foo?")
    agent.pruner_client.prune = MagicMock()

    action = {
        "action": "grep -r RST /testbed/foo --include='*.py' | head -30",
        "extra": {"response": {"choices": [{"message": {"reasoning_content": "Find tests."}}]}},
    }
    output = {"output": "x" * 2000}
    agent._apply_pruner(action, output)

    agent.cfq_generator.generate.assert_not_called()
    agent.pruner_client.prune.assert_not_called()
    assert "cfq_stats" not in output


def test_apply_pruner_uses_lower_threshold_for_agent_cfq():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        pruner={"url": "http://example.com", "threshold": 0.5, "min_output_chars": 1600},
        cfq_generator={"url": "http://example.com", "model": "test-model", "min_output_chars": 3200},
    )
    agent.pruner_client.prune = MagicMock(
        return_value=PruneResponse(
            score=0.99,
            pruned_code="filtered",
            token_scores=[],
            kept_frags=[1],
            origin_token_cnt=500,
            left_token_cnt=300,
            model_input_token_cnt=600,
        )
    )

    action = {"action": "cat -n foo.py", "context_focus_question": "What is foo?"}
    output = {"output": "x" * 2000}
    agent._apply_pruner(action, output)

    agent.pruner_client.prune.assert_called_once()
    assert output["cfq_stats"]["source"] == "agent"


def test_apply_pruner_generates_cfq_only_above_auto_threshold():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        pruner={"url": "http://example.com", "threshold": 0.5, "min_output_chars": 1600},
        cfq_generator={"url": "http://example.com", "model": "test-model", "min_output_chars": 3200},
    )
    agent.cfq_generator.generate = MagicMock(return_value="What is foo?")
    agent.pruner_client.prune = MagicMock(
        return_value=PruneResponse(
            score=0.99,
            pruned_code="filtered",
            token_scores=[],
            kept_frags=[1],
            origin_token_cnt=1000,
            left_token_cnt=500,
            model_input_token_cnt=1100,
        )
    )

    action = {
        "action": "cat -n foo.py",
        "extra": {"response": {"choices": [{"message": {"reasoning_content": "Read foo."}}]}},
    }
    output = {"output": "x" * 3500}
    agent._apply_pruner(action, output)

    agent.cfq_generator.generate.assert_called_once()
    agent.pruner_client.prune.assert_called_once()
    assert output["cfq_stats"]["source"] == "cfq_generator"


def test_apply_pruner_skips_reread():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        pruner={
            "url": "http://example.com",
            "threshold": 0.5,
            "min_output_chars": 0,
            "skip_prune_on_reread": True,
        },
    )
    mock_prune = MagicMock()
    agent.pruner_client.prune = mock_prune
    agent._file_read_counts["astropy/io/ascii/rst.py"] = 1

    action = {"action": "cat -n /testbed/astropy/io/ascii/rst.py", "context_focus_question": "What is RST?"}
    output = {"output": "x" * 5000}
    agent._apply_pruner(action, output)

    mock_prune.assert_not_called()
    assert output["cfq_stats"]["prune_skipped"] == "reread"


def test_apply_pruner_falls_back_on_low_score():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        pruner={
            "url": "http://example.com",
            "threshold": 0.5,
            "min_output_chars": 0,
        },
    )
    original = "line\n" * 400
    agent.pruner_client.prune = MagicMock(
        return_value=PruneResponse(
            score=0.006,
            pruned_code="filtered",
            token_scores=[],
            kept_frags=[1],
            origin_token_cnt=1000,
            left_token_cnt=100,
            model_input_token_cnt=1100,
        )
    )

    action = {"action": "cat -n foo.py", "context_focus_question": "What is foo?"}
    output = {"output": original}
    agent._apply_pruner(action, output)

    assert output["output"] == original
    assert output["pruned_stats"]["fallback"] is True
    assert output["pruned_stats"]["fallback_reason"] == "low_score"


def test_repeated_action_raises_after_limit():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        max_repeat_steps=3,
    )
    for _ in range(3):
        agent._track_repeat("cat -n foo.py")
    try:
        agent._track_repeat("cat -n foo.py")
        raise AssertionError("expected RepeatedAction")
    except RepeatedAction:
        pass


def test_repeated_action_resets_on_change():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        max_repeat_steps=3,
    )
    for _ in range(3):
        agent._track_repeat("cat -n foo.py")
    agent._track_repeat("cat -n bar.py")
    agent._track_repeat("cat -n bar.py")
    assert agent._repeat_count == 2


def test_repeated_action_disabled_by_default():
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
    )
    for _ in range(100):
        agent._track_repeat("cat -n foo.py")
    assert agent._repeat_count == 0
