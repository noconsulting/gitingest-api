"""Enum for tokenizer types and functions to count tokens."""

from enum import Enum

import tiktoken
from vertexai.preview.tokenization import get_tokenizer_for_model


class Tokenizer(Enum):
    """
    Enum class representing supported tokenizer configurations.

    Each tokenizer configuration consists of:
    - value: string identifier for the tokenizer/model
    - encoding_function: function that returns tokenizer encoding
    - token_count_function: function that counts tokens given encoding and text

    Currently supports:
    - GEMINI_1_5_PRO: Google's Gemini 1.5 Pro model tokenizer
    - OPEN_AI: OpenAI's cl100k_base tokenizer (used by GPT models)
    """

    GEMINI_1_5_PRO = (
        "gemini-1.5-pro-002",
        lambda: get_tokenizer_for_model(Tokenizer.GEMINI_1_5_PRO.value),
        lambda encoding, text: encoding.count_tokens(text).total_tokens,
    )

    OPEN_AI = (
        "cl100k_base",
        lambda: tiktoken.get_encoding(Tokenizer.OPEN_AI.value),
        lambda encoding, text: len(encoding.encode(text, disallowed_special=())),
    )

    def __new__(cls, value, encoding_function, token_count_function):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.encoding_function = encoding_function
        obj.token_count_function = token_count_function
        return obj
