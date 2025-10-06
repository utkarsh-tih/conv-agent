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
    list_recent_verifications as db_list_recent
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
        "pending_action": None,  # Track what agent is waiting for
    }
    
    # Fetch applicant data immediately
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
    
    # Add user message if provided
    if user_message:
        state["messages"].append(HumanMessage(content=user_message))
        state["transcript"].append({
            "speaker": "customer",
            "text": user_message,
            "timestamp": datetime.now().isoformat(),
            "mode": "text"
        })
    
    # Run one iteration of the graph
    # We need to capture agent outputs without blocking
    try:
        # Invoke the graph with current state
        new_state = graph.invoke(state)
        session["state"] = new_state
        
        # Extract agent's last message
        agent_message = None
        for msg in reversed(new_state["messages"]):
            if isinstance(msg, AIMessage):
                # Check if it's a tool call or actual text
                if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                    agent_message = msg.content
                    break
        
        # Check transcript for agent speech as a fallback (since our agent uses tools to "speak")
        if not agent_message and new_state["transcript"]:
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
        print(f"Error processing agent turn: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= API ENDPOINTS =============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple HTML interface"""
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
            
            <button onclick="startCall()">Start Verification Call</button>
        </div>
        
        <div id="chatSection" style="display: none;">
            <div id="chatBox" class="chat-box"></div>
            
            <div>
                <input type="text" id="messageInput" placeholder="Type your response..." 
                       onkeypress="if(event.key==='Enter') sendMessage()" />
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
            
            try {
                const response = await fetch('/start-call', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({customer_id: customerId, language: language})
                });
                
                const data = await response.json();
                if (response.status !== 200) {
                    throw new Error(data.detail);
                }
                
                sessionId = data.session_id;
                
                document.getElementById('sessionId').textContent = sessionId;
                document.getElementById('startSection').style.display = 'none';
                document.getElementById('chatSection').style.display = 'block';
                
                addMessage('system', `Call started with customer ${customerId}`);
                
                // --- FIX ---
                // The initial agent message is now included in the /start-call response.
                // No need for a separate polling function.
                if (data.agent_message) {
                    addMessage('agent', data.agent_message);
                }
                document.getElementById('stage').textContent = data.stage;
                
            } catch (error) {
                alert('Error starting call: ' + error);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage('customer', message);
            input.value = '';
            
            document.getElementById('sendBtn').disabled = true;
            
            try {
                const response = await fetch('/send-message', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId, message: message})
                });
                
                const data = await response.json();
                if (response.status !== 200) {
                    throw new Error(data.detail);
                }
                
                if (data.agent_message) {
                    addMessage('agent', data.agent_message);
                }
                
                document.getElementById('stage').textContent = data.stage;
                
                if (!data.awaiting_input) {
                    addMessage('system', 'Call completed');
                    document.getElementById('sendBtn').disabled = true;
                } else {
                    document.getElementById('sendBtn').disabled = false;
                }
                
            } catch (error) {
                alert('Error sending message: ' + error);
                document.getElementById('sendBtn').disabled = false;
            }
        }
        
        function addMessage(type, text) {
            const chatBox = document.getElementById('chatBox');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.textContent = text;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        async function endCall() {
            if (confirm('Are you sure you want to end this call?')) {
                try {
                    await fetch(`/end-call/${sessionId}`, {method: 'POST'});
                    addMessage('system', 'Call ended by user');
                    document.getElementById('sendBtn').disabled = true;
                } catch (error) {
                    alert('Error ending call: ' + error);
                }
            }
        }
    </script>
</body>
</html>
    """


@app.post("/start-call")
async def start_call(request: StartCallRequest):
    """
    FIX: Initialize a session AND run the first agent turn to get the greeting.
    This is more robust and avoids a race condition on the client-side.
    """
    try:
        # Step 1: Create the session and initial state
        session_id, initial_state = create_api_session(
            request.customer_id, 
            request.language
        )
        
        # Step 2: Immediately process the first agent turn to get the initial greeting
        initial_turn_result = process_agent_turn(session_id, None)
        
        # Step 3: Return a combined response with session info and the first message
        return {
            "session_id": session_id,
            "status": "started",
            "customer_id": request.customer_id,
            "applicant_name": initial_state["applicant_data"].get("name"),
            "application_number": initial_state["applicant_data"].get("application_number"),
            "agent_message": initial_turn_result.get("agent_message"),
            "stage": initial_turn_result.get("stage"),
        }
    except Exception as e:
        # It's good practice to log the actual error on the server
        print(f"Error in /start-call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# The /get-agent-message endpoint is no longer needed as its logic is merged into /start-call
# @app.get("/get-agent-message/{session_id}")
# async def get_agent_message(session_id: str): ...


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
    """End the call and save data"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    state = session["state"]
    
    # Determine verification status
    if state.get('identity_verified') and len(state.get('questions_completed', [])) >= 19:
        verification_status = "completed"
    elif state.get('identity_verified') and len(state.get('questions_completed', [])) > 0:
        verification_status = "partial"
    else:
        verification_status = "failed"
    
    # Save to database
    save_result = save_loan_verification.invoke({
        "call_sid": state['call_sid'],
        "customer_id": state.get('customer_id', 'unknown'),
        "transcript": state['transcript'],
        "verification_data": state['verification_data'],
        "audio_paths": state['audio_paths'],
        "identity_verified": state.get('identity_verified', False),
        "consent_given": state.get('consent_given', False),
        "verification_status": verification_status
    })
    
    # Clean up session
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": "ended",
        "verification_status": verification_status,
        "save_status": save_result.get('status'),
    }


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    state = session["state"]
    
    return {
        "session_id": session_id,
        "stage": state["stage"],
        "consent_given": state.get("consent_given"),
        "identity_verified": state.get("identity_verified"),
        "questions_completed": len(state.get("questions_completed", [])),
        "total_questions": 19,
        "transcript_length": len(state["transcript"]),
    }


@app.get("/list-calls", response_model=List[CallRecord])
async def list_calls(limit: int = 10):
    """
    FIX: List recent verification calls from the database.
    This now correctly calls the imported database function.
    """
    try:
        # Note: This requires the `list_recent_verifications` function in the other file 
        # to be modified to RETURN the data instead of printing it.
        recent_calls = db_list_recent(limit=limit)
        return recent_calls
    except Exception as e:
        print(f"Error listing recent calls from database: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve recent calls from the database.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)