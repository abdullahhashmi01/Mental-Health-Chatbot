"""
src/components/safety_filter.py
----------------------------------
A small, isolated safety component used at inference time.
Keeping it separate (rather than inline in the prediction pipeline)
makes it easy to unit-test, swap for a real classifier later, or
audit independently -- which matters a lot for a mental-health-
adjacent chatbot.
"""

from src.constants import CRISIS_KEYWORDS, CRISIS_RESPONSE


class SafetyFilter:
    def __init__(self, crisis_keywords=None, crisis_response: str = CRISIS_RESPONSE):
        self.crisis_keywords = crisis_keywords or CRISIS_KEYWORDS
        self.crisis_response = crisis_response

    def is_crisis_message(self, text: str) -> bool:
        lowered = text.lower()
        return any(kw in lowered for kw in self.crisis_keywords)

    def get_crisis_response(self) -> str:
        return self.crisis_response
