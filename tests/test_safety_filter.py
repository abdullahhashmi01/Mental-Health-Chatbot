"""
tests/test_safety_filter.py
------------------------------
Unit tests for the crisis-keyword safety net. This is the most
important component to test thoroughly in a mental-health-adjacent
chatbot.

Run:
    pytest tests/test_safety_filter.py -v
"""

import pytest
from src.components.safety_filter import SafetyFilter


@pytest.fixture
def safety_filter():
    return SafetyFilter()


def test_detects_explicit_crisis_language(safety_filter):
    assert safety_filter.is_crisis_message("I want to kill myself") is True


def test_detects_suicide_keyword(safety_filter):
    assert safety_filter.is_crisis_message("I've been thinking about suicide a lot") is True


def test_detects_case_insensitive(safety_filter):
    assert safety_filter.is_crisis_message("I WANT TO DIE") is True


def test_does_not_flag_normal_message(safety_filter):
    assert safety_filter.is_crisis_message("I'm just feeling a bit stressed about work") is False


def test_does_not_flag_unrelated_message(safety_filter):
    assert safety_filter.is_crisis_message("I had a great day today, thanks for asking!") is False


def test_crisis_response_contains_helpline(safety_filter):
    response = safety_filter.get_crisis_response()
    assert "988" in response or "findahelpline.com" in response


def test_custom_keywords_can_be_injected():
    custom_filter = SafetyFilter(crisis_keywords=["red flag phrase"])
    assert custom_filter.is_crisis_message("this is a red flag phrase") is True
    assert custom_filter.is_crisis_message("kill myself") is False  # not in custom list
