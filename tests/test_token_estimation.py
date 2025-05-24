from typing import Optional
from gitingest.output_formatters import _calculate_and_format_token_count
from gitingest.utils.tokenizer import Tokenizer

def test_format_openai_token_count() -> None:
    test_text: str = "This is an example text for OpenAI."
    # Assuming _calculate_token_count internally results in 9 tokens for this text with OpenAI tokenizer
    expected_formatted_count: str = "9"

    actual_formatted_count: Optional[str] = _calculate_and_format_token_count(test_text, Tokenizer.OPEN_AI)

    assert actual_formatted_count is not None, "OpenAI token formatting should not return None for valid text"
    assert actual_formatted_count == expected_formatted_count, \
        f"Incorrect OpenAI token formatting. Expected: {expected_formatted_count}, Got: {actual_formatted_count}"

def test_format_gemini_token_count() -> None:
    test_text: str = "This is an example text for Gemini."
    # Assuming _calculate_token_count internally results in 8 tokens for this text with Gemini tokenizer
    expected_formatted_count: str = "8"

    actual_formatted_count: Optional[str] = _calculate_and_format_token_count(test_text, Tokenizer.GEMINI_1_5_PRO)

    assert actual_formatted_count is not None, "Gemini token formatting should not return None for valid text"
    assert actual_formatted_count == expected_formatted_count, \
        f"Incorrect Gemini token formatting. Expected: {expected_formatted_count}, Got: {actual_formatted_count}"

def test_format_token_count_none_tokenizer() -> None:
    test_text: str = "Some arbitrary text."
    # If the tokenizer is None, _calculate_and_format_token_count should return None.
    expected_output: None = None

    actual_output: Optional[str] = _calculate_and_format_token_count(test_text, None)
    assert actual_output is expected_output, \
        f"Token formatting with a None tokenizer should return None, got: {actual_output}"

def test_format_token_count_large_numbers(mocker) -> None: # mocker is a pytest fixture
    # Mock the internal _calculate_token_count function to directly test reformatting logic
    mock_calculate = mocker.patch("gitingest.output_formatters._calculate_token_count")

    test_cases: list[tuple[int, str]] = [
        (120, "120"),
        (999, "999"),
        (1000, "1.0k"),
        (1234, "1.2k"),    # 1.234 -> 1.2k
        (9999, "10.0k"),   # 9.999 -> 10.0k
        (999999, "1000.0k"),# 999.999 -> 1000.0k
        (1000000, "1.0M"),
        (123456789, "123.5M")# 123.456789 -> 123.5M (rounds .45 up to .5)
    ]

    for token_count, expected_format in test_cases:
        mock_calculate.return_value = token_count
        # The choice of tokenizer doesn't matter here as _calculate_token_count is mocked.
        # Using Tokenizer.OPEN_AI as a placeholder.
        actual_formatted_count: Optional[str] = _calculate_and_format_token_count("some text", Tokenizer.OPEN_AI)
        assert actual_formatted_count == expected_format, \
            f"For token count {token_count}, expected format '{expected_format}', but got '{actual_formatted_count}'"

    # Test case for when _calculate_token_count returns -1 (simulating an error)
    mock_calculate.return_value = -1
    actual_formatted_count_error: Optional[str] = _calculate_and_format_token_count("some text", Tokenizer.OPEN_AI)
    assert actual_formatted_count_error is None, \
        f"Expected None when _calculate_token_count returns -1, but got {actual_formatted_count_error}"