import random
from typing import Any
from datetime import datetime, timedelta

from .models import Email, Action, Observation, SAMPLE_EMAILS


def clamp(value: float) -> float:
    """Ensure value is strictly between 0 and 1."""
    if value <= 0.0 or value >= 1.0:
        return 0.5
    return value


class EmailTriageEnv:
    def __init__(self, seed: int = None):
        self.seed = seed
        self.emails: list[Email] = []
        self.current_email_index: int = 0
        self.session_score: float = 0.0
        self.task_id: str = "categorize_inbox"
        self.done: bool = False
        self._action_history: list[dict] = []
        
    def reset(self, task_id: str = "categorize_inbox") -> dict:
        self.task_id = task_id
        self.emails = [Email(**e) for e in SAMPLE_EMAILS]
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(self.emails)
        for i, email in enumerate(self.emails):
            email.id = i + 1
        self.current_email_index = 0
        self.session_score = 0.0
        self.done = False
        self._action_history = []
        return self._get_observation()
    
    def _get_observation(self) -> dict:
        return {
            "inbox_count": len([e for e in self.emails if e.category == "inbox"]),
            "emails": [e.model_dump() for e in self.emails],
            "current_email_index": self.current_email_index,
            "session_score": self.session_score
        }
    
    def step(self, action: dict) -> tuple[dict, float, bool]:
        action_obj = Action(**action)
        
        reward = 0.01
        feedback = ""
        
        if action_obj.action_type == "categorize":
            if action_obj.category in ["work", "personal", "spam"]:
                for email in self.emails:
                    if email.id == action_obj.email_id:
                        old_cat = email.category
                        email.category = action_obj.category
                        
                        if self.task_id == "categorize_inbox":
                            if action_obj.category in ["work", "personal", "spam"]:
                                raw_reward = 1.0 / len(self.emails)
                                reward = clamp(raw_reward)
                            else:
                                reward = 0.01
                        
                        feedback = f"Categorized email {action_obj.email_id} as {action_obj.category}"
                        break
                        
        elif action_obj.action_type == "prioritize":
            if action_obj.priority and 1 <= action_obj.priority <= 5:
                for email in self.emails:
                    if email.id == action_obj.email_id:
                        email.priority = action_obj.priority
                        
                        if self.task_id == "prioritize_urgent":
                            if action_obj.priority <= 2:
                                raw_reward = 1.0 / len([e for e in self.emails if "boss" in e.sender or "urgent" in e.subject.lower()])
                                reward = clamp(raw_reward)
                            else:
                                reward = 0.01
                        
                        feedback = f"Set priority of email {action_obj.email_id} to {action_obj.priority}"
                        break
                        
        elif action_obj.action_type == "archive":
            for email in self.emails:
                if email.id == action_obj.email_id:
                    email.category = "archived"
                    
                    if self.task_id == "archive_clutter":
                        if email.priority > 2:
                            reward = 0.49
                        else:
                            reward = 0.01
                    
                    feedback = f"Archived email {action_obj.email_id}"
                    break
        
        self.session_score += reward
        self._action_history.append({"action": action_obj.model_dump(), "reward": reward, "feedback": feedback})
        
        self.current_email_index = (self.current_email_index + 1) % len(self.emails)
        
        inbox_count = len([e for e in self.emails if e.category == "inbox"])
        if inbox_count == 0:
            self.done = True
        
        return self._get_observation(), reward, self.done
    
    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "done": self.done,
            "observation": self._get_observation(),
            "action_history": self._action_history
        }
    
    def get_email_by_id(self, email_id: int) -> Email | None:
        for email in self.emails:
            if email.id == email_id:
                return email
        return None
