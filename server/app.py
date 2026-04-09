#!/usr/bin/env python
"""
OpenEnv Email Triage Server
Serves the environment via REST API for OpenEnv validation.
"""

from src.environment import EmailTriageEnv
from src.models import Email
from graders import GRADERS
from typing import Optional

env = EmailTriageEnv(seed=42)
current_task = "categorize_inbox"


def clamp_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    if score <= 0.0 or score >= 1.0:
        return 0.5
    return score


def reset(task_id: str = "categorize_inbox") -> dict:
    global current_task
    current_task = task_id
    obs = env.reset(task_id)
    return {"status": "ok", "observation": obs, "task_id": task_id}


def step(action: dict) -> dict:
    global current_task
    obs, reward, done = env.step(action)
    return {"status": "ok", "observation": obs, "reward": reward, "done": done}


def state() -> dict:
    global current_task
    st = env.state()
    emails = [Email(**e) for e in st["observation"]["emails"]]
    grader = GRADERS.get(current_task, GRADERS["categorize_inbox"])
    raw_score = grader(emails)
    score = clamp_score(raw_score)
    return {
        "status": "ok",
        "task_id": current_task,
        "done": st["done"],
        "observation": st["observation"],
        "score": score,
    }


def main():
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI(title="Email Triage OpenEnv")
    
    class ResetRequest(BaseModel):
        task_id: str = "categorize_inbox"
    
    class StepRequest(BaseModel):
        action_type: str
        email_id: int
        category: Optional[str] = None
        priority: Optional[int] = None
    
    @app.get("/")
    def root():
        return {"status": "ok", "message": "Email Triage Environment - OpenEnv"}
    
    @app.post("/reset")
    def api_reset(request: ResetRequest = None):
        task_id = request.task_id if request else "categorize_inbox"
        return reset(task_id)
    
    @app.post("/step")
    def api_step(request: StepRequest):
        action = {"action_type": request.action_type, "email_id": request.email_id}
        if request.category:
            action["category"] = request.category
        if request.priority:
            action["priority"] = request.priority
        return step(action)
    
    @app.get("/state")
    def api_state():
        return state()
    
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()