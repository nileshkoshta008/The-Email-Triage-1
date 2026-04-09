"""Hugging Face Space application for Email Triage Environment with OpenEnv API."""
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.environment import EmailTriageEnv
from src.models import Email
from graders import GRADERS

app = FastAPI(title="Email Triage Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnv(seed=42)
current_task = "categorize_inbox"


class ResetRequest(BaseModel):
    task_id: str = "categorize_inbox"


class StepRequest(BaseModel):
    action_type: str
    email_id: int
    category: Optional[str] = None
    priority: Optional[int] = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Email Triage Environment - OpenEnv compatible"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    global current_task
    task_id = request.task_id if request else "categorize_inbox"
    current_task = task_id
    obs = env.reset(task_id)
    return {"status": "ok", "observation": obs, "task_id": task_id}


@app.post("/step")
def step(request: StepRequest):
    global current_task
    
    action = {
        "action_type": request.action_type,
        "email_id": request.email_id,
    }
    if request.category:
        action["category"] = request.category
    if request.priority:
        action["priority"] = request.priority
    
    obs, reward, done = env.step(action)
    
    return {
        "status": "ok",
        "observation": obs,
        "reward": reward,
        "done": done
    }


@app.get("/state")
def state():
    st = env.state()
    emails = [Email(**e) for e in st["observation"]["emails"]]
    grader = GRADERS.get(current_task, GRADERS["categorize_inbox"])
    raw_score = grader(emails)
    
    if raw_score <= 0.0 or raw_score >= 1.0:
        score = 0.5
    else:
        score = raw_score
    
    return {
        "status": "ok",
        "task_id": current_task,
        "done": st["done"],
        "observation": st["observation"],
        "score": score
    }


def format_observation(obs: dict) -> str:
    emails = obs["emails"]
    inbox_emails = [e for e in emails if e.get("category", "inbox") == "inbox"]
    
    if not inbox_emails:
        return "No emails in inbox. Task complete!"
    
    lines = ["## Emails\n"]
    for e in inbox_emails:
        lines.append(f"**ID {e['id']}** | From: {e['sender']}")
        lines.append(f"- Subject: {e['subject']}")
        lines.append(f"- Priority: {e.get('priority', 3)} | Category: {e.get('category', 'inbox')}")
        lines.append("")
    
    lines.append(f"\n**Inbox Count:** {obs['inbox_count']}")
    lines.append(f"**Session Score:** {obs['session_score']:.2f}")
    
    return "\n".join(lines)


def reset_task(task_id: str):
    global current_task
    current_task = task_id
    obs = env.reset(task_id)
    return format_observation(obs), ""


def take_action(action_type: str, email_id: int, category: str, priority: int):
    action = {"action_type": action_type, "email_id": email_id}
    
    if action_type == "categorize" and category:
        action["category"] = category
    elif action_type == "prioritize" and priority:
        action["priority"] = priority
    
    obs, reward, done = env.step(action)
    
    state = env.state()
    emails = [Email(**e) for e in state["observation"]["emails"]]
    grader = GRADERS.get(current_task, GRADERS["categorize_inbox"])
    raw_final_score = grader(emails)
    
    if raw_final_score <= 0.0 or raw_final_score >= 1.0:
        final_score = 0.5
    else:
        final_score = raw_final_score
    
    result = f"**Action:** {action_type} on email {email_id}\n"
    result += f"**Reward:** {reward:.2f}\n"
    result += f"**Done:** {done}\n"
    result += f"**Final Score:** {final_score:.2f}\n"
    
    return format_observation(obs), result


def get_current_observation():
    obs = env._get_observation()
    return format_observation(obs)


with gr.Blocks(title="Email Triage Environment") as demo:
    gr.Markdown("# Email Triage Environment")
    gr.Markdown("An OpenEnv environment for AI agents to learn email triage tasks.")
    
    with gr.Row():
        with gr.Column():
            task_dropdown = gr.Dropdown(
                choices=["categorize_inbox", "prioritize_urgent", "archive_clutter"],
                value="categorize_inbox",
                label="Select Task"
            )
            reset_btn = gr.Button("Reset Environment")
        
        with gr.Column():
            gr.Markdown("### Action")
            action_type = gr.Radio(["categorize", "prioritize", "archive"], value="categorize", label="Action Type")
            email_id = gr.Number(value=1, minimum=1, label="Email ID")
            category = gr.Dropdown(["work", "personal", "spam"], label="Category (for categorize)")
            priority = gr.Slider(1, 5, value=3, step=1, label="Priority (for prioritize)")
            submit_btn = gr.Button("Take Action")
    
    with gr.Row():
        with gr.Column(scale=2):
            obs_display = gr.Markdown()
        with gr.Column(scale=1):
            result_display = gr.Textbox(label="Result", lines=10)
    
    reset_btn.click(reset_task, inputs=[task_dropdown], outputs=[obs_display, result_display])
    submit_btn.click(
        take_action, 
        inputs=[action_type, email_id, category, priority],
        outputs=[obs_display, result_display]
    )
    demo.load(get_current_observation, outputs=[obs_display])

app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
