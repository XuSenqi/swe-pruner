from unittest.mock import MagicMock, patch

from minisweagent.utils.cfq_generator import CFQGenerator, CFQGeneratorConfig

MIN_OUTPUT_CHARS = 500


def test_cfq_generator_config_default_min_output_chars():
    gen = CFQGenerator(CFQGeneratorConfig())
    assert gen.config.min_output_chars == 3200


def test_should_generate_for_large_cat_output():
    gen = CFQGenerator(CFQGeneratorConfig())
    assert gen.should_generate("cat -n foo.py", "x" * 600, min_output_chars=MIN_OUTPUT_CHARS)


def test_should_not_generate_for_edit_or_small_output():
    gen = CFQGenerator(CFQGeneratorConfig())
    assert not gen.should_generate("sed -i 's/a/b/' foo.py", "x" * 600, min_output_chars=MIN_OUTPUT_CHARS)
    assert not gen.should_generate("cat -n foo.py", "short", min_output_chars=MIN_OUTPUT_CHARS)


def test_should_generate_respects_min_output_chars():
    gen = CFQGenerator(CFQGeneratorConfig())
    assert not gen.should_generate("cat -n foo.py", "x" * 499, min_output_chars=500)
    assert gen.should_generate("cat -n foo.py", "x" * 500, min_output_chars=500)


def test_build_reasoning_block_uses_prior_when_current_is_short():
    gen = CFQGenerator(CFQGeneratorConfig())
    block, used_prior = gen.build_reasoning_block(
        "Let me continue looking for the __eq__ method.",
        [
            "The bug is UnrecognizedUnit == None raises TypeError.",
            "Let me look at UnitBase.__eq__ to see how it handles None.",
        ],
    )
    assert used_prior is True
    assert "Prior agent reasoning" in block
    assert "handles None" in block
    assert "continue looking" in block


def test_build_reasoning_block_skips_prior_when_current_is_long():
    gen = CFQGenerator(CFQGeneratorConfig())
    long_reasoning = "x" * 200
    block, used_prior = gen.build_reasoning_block(long_reasoning, ["older context"])
    assert used_prior is False
    assert block.startswith("Agent reasoning:")


def test_generate_parses_openai_response():
    gen = CFQGenerator(CFQGeneratorConfig())
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "How does __eq__ handle None?"}}]
    }
    with patch.object(gen.session, "post", return_value=mock_response) as post:
        cfq = gen.generate(
            reasoning="Let me continue looking for __eq__.",
            command="cat -n foo.py",
            prior_reasoning=["Need to compare with UnitBase.__eq__ when other is None."],
        )
    assert cfq == "How does __eq__ handle None?"
    post.assert_called_once()
    payload = post.call_args.kwargs["json"]
    assert "Prior agent reasoning" in payload["messages"][1]["content"]


def test_generate_returns_none_for_skip():
    gen = CFQGenerator(CFQGeneratorConfig())
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"choices": [{"message": {"content": "SKIP"}}]}
    with patch.object(gen.session, "post", return_value=mock_response):
        assert (
            gen.generate(reasoning="Apply the sed edit.", command="sed -i 's/a/b/' foo.py") is None
        )


def test_generate_returns_none_without_reasoning():
    gen = CFQGenerator(CFQGeneratorConfig())
    assert gen.generate(reasoning="", command="cat -n foo.py") is None


def test_normalize_output_rejects_line_number_questions():
    gen = CFQGenerator(CFQGeneratorConfig())
    assert gen._normalize_output("What is the code at lines 1708 to 1735?") is None
