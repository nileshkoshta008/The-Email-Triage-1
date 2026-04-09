"""Grader for prioritize_urgent task."""
from src.models import Email


def grade_prioritize_urgent(emails: list[Email]) -> float:
    """
    Grade how well urgent emails were prioritized.
    Score is strictly between 0 and 1 (never 0.0 or 1.0).
    """
    if not emails:
        return 0.5
    
    correct = 0
    total = len(emails)
    
    urgent_senders = ["boss@company.com"]
    urgent_keywords = ["urgent", "asap", "important", "deadline"]
    
    for email in emails:
        should_be_urgent = (
            email.sender in urgent_senders or
            any(kw in email.subject.lower() for kw in urgent_keywords)
        )
        
        is_actually_urgent = email.priority <= 2
        
        if should_be_urgent and is_actually_urgent:
            correct += 1
        elif not should_be_urgent and not is_actually_urgent:
            correct += 1
        elif should_be_urgent and not is_actually_urgent:
            correct += 0.25
    
    result = (correct / total) if total > 0 else 0.5
    
    if result <= 0.0 or result >= 1.0 or result < 0.001 or result > 0.999:
        result = 0.5
    
    return result