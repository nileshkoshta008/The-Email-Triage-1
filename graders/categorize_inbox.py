"""Grader for categorize_inbox task."""
from src.models import Email
import random


def grade_categorize_inbox(emails: list[Email]) -> float:
    """
    Grade how well emails were categorized.
    Score is strictly between 0 and 1 (never 0.0 or 1.0).
    """
    if not emails:
        return 0.5
    
    score = 0.0
    total = len(emails)
    
    expected = {
        1: "work",
        2: "personal",
        3: "spam",
        4: "work",
        5: "spam",
    }
    
    for email in emails:
        if email.category != "inbox":
            if email.id in expected and email.category == expected[email.id]:
                score += 1.0
            elif email.id in expected:
                score += 0.5
            else:
                score += 0.25
    
    result = (score / total) if total > 0 else 0.5
    
    if result <= 0.0 or result >= 1.0 or result < 0.001 or result > 0.999:
        result = 0.5
    
    return result