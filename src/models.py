from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class Email(BaseModel):
    id: int
    sender: str
    subject: str
    body: str
    timestamp: str
    category: str = "inbox"
    priority: int = Field(default=3, ge=1, le=5)


class Action(BaseModel):
    action_type: Literal["categorize", "prioritize", "archive"]
    email_id: int
    category: Optional[str] = None
    priority: Optional[int] = Field(default=None, ge=1, le=5)


class Observation(BaseModel):
    inbox_count: int
    emails: list[Email]
    current_email_index: int
    session_score: float


class Reward(BaseModel):
    total: float
    breakdown: dict[str, float]


class TaskInfo(BaseModel):
    task_id: str
    description: str
    difficulty: str
    max_score: float


SAMPLE_EMAILS = [
    {"id": 1, "sender": "boss@company.com", "subject": "URGENT: Client meeting at 3pm", "body": "We need to discuss the quarterly report before the client arrives. Please be in the conference room.", "timestamp": "2026-04-07T14:00:00", "category": "inbox", "priority": 1},
    {"id": 2, "sender": "mom@example.com", "subject": "Dinner on Sunday", "body": "Hey dear, are you coming over for dinner this Sunday? Your sister will be here too.", "timestamp": "2026-04-07T12:00:00", "category": "inbox", "priority": 3},
    {"id": 3, "sender": "newsletter@spam.com", "subject": "YOU WON A MILLION DOLLARS!!!", "body": "Click here to claim your prize! Free money guaranteed!", "timestamp": "2026-04-07T10:00:00", "category": "inbox", "priority": 3},
    {"id": 4, "sender": "colleague@company.com", "subject": "Code review needed", "body": "Can you review my PR when you get a chance? It's for the new feature.", "timestamp": "2026-04-06T16:00:00", "category": "inbox", "priority": 2},
    {"id": 5, "sender": "netflix@email.com", "subject": "New shows this week", "body": "Check out the new releases on Netflix this week!", "timestamp": "2026-04-05T08:00:00", "category": "inbox", "priority": 5},
]
