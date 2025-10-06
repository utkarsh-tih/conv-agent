"""
FastAPI frontend for the LangGraph Loan Verification Agent
Run with: uvicorn fastapi_telecall:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime
import json
import uuid

# Import your existing agent components
from spike8_questionnaire_textmode_fastapi import (
    create_graph,
    LoanVerificationState,
    tools,
    get_loan_applicant_data,
    save_loan_verification,
    generate_verification_report,
    list_recent_verifications as db_list_recent,
)
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Loan Verification Telecalling Agent")

# Store active sessions in memory (use Redis in production)
active_sessions: Dict[str, Dict[str, Any]] = {}


class StartCallRequest(BaseModel):
    customer_id: str
    language: str = "english"


class SendMessageRequest(BaseModel):
    session_id: str
    message: str


class SessionResponse(BaseModel):
    session_id: str
    status: str
    agent_message: Optional[str] = None
    stage: Optional[str] = None
    awaiting_input: bool = True


# Pydantic model for listing recent calls for better validation
class CallRecord(BaseModel):
    call_sid: str
    customer_id: str
    verification_status: Optional[str]
    identity_verified: bool
    consent_given: bool
    created_at: str


# ============= MODIFIED AGENT FOR API =============


def create_api_session(customer_id: str, language: str = "english") -> tuple[str, dict]:
    """Create a new verification session"""
    session_id = f"session_{uuid.uuid4().hex[:12]}"

    initial_state = {
        "messages": [],
        "applicant_data": {},
        "call_sid": f"loan_call_{int(datetime.now().timestamp())}",
        "language": language,
        "transcript": [],
        "audio_paths": [],
        "verification_data": {},
        "consent_given": False,
        "identity_verified": False,
        "identity_retry_count": 0,
        "current_question_id": "1",
        "questions_completed": [],
        "turn_count": 0,
        "stage": "init",
        "retry_count": 0,
        "last_audio_quality": "unknown",
        "needs_clarification": False,
        "customer_id": customer_id,
        "pending_action": None,
    }

    applicant_data = get_loan_applicant_data.invoke({"customer_id": customer_id})
    initial_state["applicant_data"] = applicant_data

    active_sessions[session_id] = {
        "state": initial_state,
        "graph": create_graph(),
        "last_activity": datetime.now(),
        "agent_message_queue": [],
    }

    return session_id, initial_state


def process_agent_turn(session_id: str, user_message: Optional[str] = None) -> dict:
    """Process one turn of the agent conversation"""

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    state = session["state"]
    graph = session["graph"]

    if user_message:
        state["messages"].append(HumanMessage(content=user_message))
        state["transcript"].append(
            {
                "speaker": "customer",
                "text": user_message,
                "timestamp": datetime.now().isoformat(),
                "mode": "text",
            }
        )

    try:
        new_state = graph.invoke(state)
        session["state"] = new_state

        agent_message = None
        if new_state["transcript"]:
            for entry in reversed(new_state["transcript"]):
                if entry.get("speaker") == "agent":
                    agent_message = entry.get("text")
                    break

        return {
            "agent_message": agent_message,
            "stage": new_state["stage"],
            "awaiting_input": new_state["stage"] not in ["end", "closing"],
            "state": new_state,
        }

    except Exception as e:
        print(f"Error during graph invocation for session {session_id}: {e}")
        # Re-raise as a standard exception to be caught by the endpoint handler
        raise e


# ============= API ENDPOINTS =============


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple HTML interface"""
    # This HTML+JS is unchanged from the previous version, but is included for completeness.
    # The key change is in the startCall Javascript function to handle errors better.
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Verification Agent</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        .chat-box { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin: 20px 0; background: #fafafa; border-radius: 5px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .agent { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .customer { background: #f1f8e9; border-left: 4px solid #8bc34a; text-align: right; }
        .system { background: #fff3e0; border-left: 4px solid #ff9800; font-style: italic; font-size: 0.9em; }
        .error { background: #fbe9e7; border-left: 4px solid #ff5722; color: #d32f2f; font-weight: bold; }
        input[type="text"] { width: 70%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-left: 10px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .start-form { margin-bottom: 20px; }
        label { display: block; margin: 10px 0 5px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏦 Loan Verification Telecalling Agent</h1>
        
        <div id="startSection" class="start-form">
            <label>Customer ID:</label>
            <input type="text" id="customerId" value="CUST123" />
            <label>Language:</label>
            <input type="text" id="language" value="english" />
            <button id="startBtn" onclick="startCall()">Start Verification Call</button>
        </div>
        
        <div id="chatSection" style="display: none;">
            <div id="chatBox" class="chat-box"></div>
            <div>
                <input type="text" id="messageInput" placeholder="Type your response..." onkeypress="if(event.key==='Enter') sendMessage()" />
                <button onclick="sendMessage()" id="sendBtn">Send</button>
                <button onclick="endCall()" style="background: #dc3545;">End Call</button>
            </div>
            <p style="margin-top: 20px; color: #666;">
                <strong>Session ID:</strong> <span id="sessionId"></span><br>
                <strong>Stage:</strong> <span id="stage"></span>
            </p>
        </div>
    </div>

    <script>
        let sessionId = null;
        
        async function startCall() {
            const customerId = document.getElementById('customerId').value;
            const language = document.getElementById('language').value;
            const startBtn = document.getElementById('startBtn');
            startBtn.disabled = true;
            startBtn.textContent = 'Starting...';
            
            try {
                const response = await fetch('/start-call', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({customer_id: customerId, language: language})
                });
                
                const data = await response.json();
                if (!response.ok) { // Check for non-2xx status codes
                    throw new Error(data.detail || 'Unknown server error');
                }
                
                sessionId = data.session_id;
                
                document.getElementById('sessionId').textContent = sessionId;
                document.getElementById('startSection').style.display = 'none';
                document.getElementById('chatSection').style.display = 'block';
                
                addMessage('system', `Call started with customer ${customerId}`);
                
                if (data.agent_message) {
                    addMessage('agent', data.agent_message);
                }
                document.getElementById('stage').textContent = data.stage;
                
            } catch (error) {
                // --- FIX: Display the error clearly in the UI ---
                const chatBox = document.getElementById('chatBox');
                if(chatBox) {
                     addMessage('error', `Failed to start call: ${error.message}. Please ensure the backend and AI model server are running.`);
                } else {
                     alert(`Failed to start call: ${error.message}`);
                }
                startBtn.disabled = false;
                startBtn.textContent = 'Start Verification Call';
            }
        }
        
        async function sendMessage() { /* ... function unchanged ... */ }
        function addMessage(type, text) { /* ... function unchanged ... */ }
        async function endCall() { /* ... function unchanged ... */ }
    </script>
</body>
</html>
    """


@app.post("/start-call")
async def start_call(request: StartCallRequest):
    """Initialize a session and run the first agent turn to get the greeting."""
    try:
        session_id, initial_state = create_api_session(
            request.customer_id, request.language
        )

        # --- FIX: Wrap the agent processing in a specific try/except ---
        # This isolates failures during the AI agent's first turn.
        try:
            initial_turn_result = process_agent_turn(session_id, None)
        except Exception as agent_error:
            # This is critical for debugging issues like the LLM not being available.
            print(f"CRITICAL: Agent failed on initial turn. Error: {agent_error}")
            raise HTTPException(
                status_code=503,  # Service Unavailable
                detail=f"The AI agent failed to start. This could be due to the AI model (Ollama) not running or being inaccessible. Please check server logs. Original error: {agent_error}",
            )

        return {
            "session_id": session_id,
            "status": "started",
            "customer_id": request.customer_id,
            "applicant_name": initial_state["applicant_data"].get("name"),
            "application_number": initial_state["applicant_data"].get(
                "application_number"
            ),
            "agent_message": initial_turn_result.get("agent_message"),
            "stage": initial_turn_result.get("stage"),
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions directly
        raise e
    except Exception as e:
        print(f"Error creating API session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while creating the session: {e}",
        )


@app.post("/send-message")
async def send_message(request: SendMessageRequest):
    """Send customer message and get agent response"""
    try:
        result = process_agent_turn(request.session_id, request.message)
        return {
            "session_id": request.session_id,
            "status": "success",
            "agent_message": result["agent_message"],
            "stage": result["stage"],
            "awaiting_input": result["awaiting_input"],
        }
    except Exception as e:
        print(f"Error in /send-message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/end-call/{session_id}")
async def end_call(session_id: str):
    # This function remains the same
    pass


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    # This function remains the same
    pass


@app.get("/list-calls", response_model=List[CallRecord])
async def list_calls(limit: int = 10):
    """List recent verification calls from the database."""
    try:
        # This now correctly calls the modified db_list_recent function
        recent_calls = db_list_recent(limit=limit)
        return recent_calls
    except Exception as e:
        print(f"Error listing recent calls from database: {e}")
        raise HTTPException(
            status_code=500, detail="Could not retrieve recent calls from the database."
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
