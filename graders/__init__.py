from .categorize_inbox import grade_categorize_inbox
from .prioritize_urgent import grade_prioritize_urgent
from .archive_clutter import grade_archive_clutter

GRADERS = {
    "categorize_inbox": grade_categorize_inbox,
    "prioritize_urgent": grade_prioritize_urgent,
    "archive_clutter": grade_archive_clutter,
}

__all__ = ["GRADERS", "grade_categorize_inbox", "grade_prioritize_urgent", "grade_archive_clutter"]
