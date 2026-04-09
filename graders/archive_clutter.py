"""Grader for archive_clutter task."""
from datetime import datetime, timedelta
from src.models import Email


def grade_archive_clutter(emails: list[Email]) -> float:
    """
    Grade how well old non-urgent emails were archived.
    Score is strictly between 0 and 1 (never 0.0 or 1.0).
    """
    if not emails:
        return 0.5
    
    try:
        cutoff = datetime.now() - timedelta(days=7)
    except:
        cutoff = datetime(2026, 4, 1)
    
    old_non_urgent = 0
    correctly_archived = 0
    incorrectly_archived = 0
    
    for email in emails:
        try:
            email_date = datetime.fromisoformat(email.timestamp.replace("Z", "+00:00"))
            if email_date.tzinfo:
                email_date = email_date.replace(tzinfo=None)
        except:
            email_date = datetime(2026, 3, 25)
        
        is_old = email_date < cutoff
        is_urgent = email.priority <= 2
        is_archived = email.category == "archived"
        
        if is_old and not is_urgent:
            old_non_urgent += 1
            if is_archived:
                correctly_archived += 1
        elif is_urgent and is_archived:
            incorrectly_archived += 1
    
    # Handle edge cases
    if old_non_urgent == 0:
        if incorrectly_archived > 0:
            result = 0.5
        else:
            result = 0.5
    elif correctly_archived + incorrectly_archived == 0:
        result = 0.5
    else:
        precision = correctly_archived / (correctly_archived + incorrectly_archived)
        recall = correctly_archived / old_non_urgent
        
        if precision + recall == 0:
            result = 0.5
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            result = f1
    
    # Ensure strictly between 0 and 1
    if result <= 0.0 or result >= 1.0 or result < 0.001 or result > 0.999:
        result = 0.5
    
    return result
