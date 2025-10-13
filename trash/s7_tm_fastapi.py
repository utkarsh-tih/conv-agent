"""
FastAPI + WebSocket Interface for LangGraph Loan Verification Telecalling Agent
Real-time bidirectional communication without polling
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import uuid
import sys
from io import StringIO
import os
from typing import Dict, Optional
import json
from datetime import datetime

# Set text mode before importing the main module
os.environ["TEXT_MODE"] = "true"

# Import your existing verification system
from spike7_questionnaire_textmode import (
    create_graph,
    save_loan_verification,
)
from langchain_core.messages import HumanMessage
import time

app = FastAPI(title="Loan Verification Telecalling API")

# Store active sessions
active_sessions: Dict[str, "SessionManager"] = {}


class StartSessionRequest(BaseModel):
    customer_id: str
    language: str = "english"


class SessionManager:
    """Manages individual verification sessions with async support"""

    def __init__(self, customer_id: str, language: str = "english"):
        self.customer_id = customer_id
        self.language = language
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.call_sid = f"loan_call_{int(time.time())}"

        # WebSocket connection
        self.websocket: Optional[WebSocket] = None

        # Async queues for communication
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()

        # Session state
        self.state = self._initialize_state()
        self.graph = create_graph()
        self.task: Optional[asyncio.Task] = None
        self.running = False

        # Chat history
        self.chat_history = []

    def _initialize_state(self):
        """Initialize LangGraph state"""
        return {
            "messages": [
                HumanMessage(
                    content=f"Start loan verification call with customer ID: {self.customer_id}"
                )
            ],
            "applicant_data": {},
            "call_sid": self.call_sid,
            "language": self.language,
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
        }

    async def add_to_chat(self, speaker: str, message: str, metadata: dict = None):
        """Add message to chat history and send via WebSocket"""
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "message": message,
            "metadata": metadata or {},
        }
        self.chat_history.append(chat_entry)

        # Send via WebSocket if connected
        if self.websocket:
            try:
                await self.websocket.send_json({"type": "message", "data": chat_entry})
            except Exception as e:
                print(f"WebSocket send error: {e}")

    async def send_status_update(self):
        """Send status update via WebSocket"""
        if self.websocket:
            try:
                status = self.get_status()
                await self.websocket.send_json({"type": "status", "data": status})
            except Exception as e:
                print(f"WebSocket status update error: {e}")

    async def run_verification(self):
        """Run the verification workflow asynchronously"""
        self.running = True

        try:
            # Monkey-patch input() to use our async queue
            original_input = __builtins__.input

            def queued_input(prompt=""):
                # Agent is asking for input
                if "You (customer):" in prompt or "CUSTOMER INPUT REQUIRED" in prompt:
                    try:
                        # Create a new event loop task to wait for input
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        user_input = loop.run_until_complete(
                            asyncio.wait_for(self.input_queue.get(), timeout=300)
                        )
                        return user_input
                    except asyncio.TimeoutError:
                        return ""
                return ""

            __builtins__.input = queued_input

            # Capture print statements to extract agent messages
            captured_output = []
            original_print = print

            # def custom_print(*args, **kwargs):
            #     output = StringIO()
            #     original_print(*args, file=output, **kwargs)
            #     text = output.getvalue().strip()
            #     captured_output.append(text)

            #     # Check if this is an agent message
            #     if "Agent:" in text:
            #         # Extract the agent's message
            #         agent_msg = text.split("Agent:", 1)[1].strip()
            #         # Send it via WebSocket
            #         asyncio.create_task(
            #             self.add_to_chat("agent", agent_msg, {"from_print": True})
            #         )

            def custom_print(*args, **kwargs):
                # If the print call is already being directed to a specific file,
                # let it pass through without capturing it for the WebSocket.
                if "file" in kwargs and kwargs["file"] is not None:
                    return original_print(*args, **kwargs)

                # Capture output destined for stdout
                output = StringIO()
                original_print(*args, file=output, **kwargs)
                text = output.getvalue().strip()

                if not text:
                    return

                captured_output.append(text)

                # Check if this is an agent message to be sent via WebSocket
                if "Agent:" in text:
                    # Extract the agent's message
                    agent_msg = text.split("Agent:", 1)[1].strip()
                    # Send it via WebSocket
                    asyncio.create_task(
                        self.add_to_chat("agent", agent_msg, {"from_print": True})
                    )

            # Replace print temporarily
            import builtins

            builtins.print = custom_print

            # Run the graph in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(
                None, lambda: self.graph.invoke(self.state)
            )

            # Restore original functions
            builtins.print = original_print
            __builtins__.input = original_input

            # Update state
            self.state = final_state

            # Determine verification status
            if (
                final_state.get("identity_verified")
                and len(final_state.get("questions_completed", [])) >= 19
            ):
                verification_status = "completed"
            elif (
                final_state.get("identity_verified")
                and len(final_state.get("questions_completed", [])) > 0
            ):
                verification_status = "partial"
            else:
                verification_status = "failed"

            # Save to database
            await loop.run_in_executor(
                None,
                lambda: save_loan_verification.invoke(
                    {
                        "call_sid": final_state["call_sid"],
                        "customer_id": self.customer_id,
                        "transcript": final_state["transcript"],
                        "verification_data": final_state["verification_data"],
                        "audio_paths": final_state["audio_paths"],
                        "identity_verified": final_state.get(
                            "identity_verified", False
                        ),
                        "consent_given": final_state.get("consent_given", False),
                        "verification_status": verification_status,
                    }
                ),
            )

            # Send completion message
            await self.add_to_chat(
                "system",
                f"✅ Verification completed with status: {verification_status}",
                {"status": verification_status, "final_state": True},
            )

        except Exception as e:
            await self.add_to_chat("system", f"❌ Error: {str(e)}", {"error": True})
        finally:
            self.running = False
            await self.send_status_update()

    def start(self):
        """Start verification in background task"""
        self.task = asyncio.create_task(self.run_verification())

    async def send_input(self, text: str):
        """Send user input to the verification workflow"""
        await self.input_queue.put(text)
        await self.add_to_chat("customer", text)
        await self.send_status_update()

    def get_status(self):
        """Get current session status"""
        return {
            "session_id": self.session_id,
            "customer_id": self.customer_id,
            "running": self.running,
            "stage": self.state.get("stage", "unknown"),
            "identity_verified": self.state.get("identity_verified", False),
            "consent_given": self.state.get("consent_given", False),
            "questions_completed": len(self.state.get("questions_completed", [])),
            "total_questions": 19,
            "call_sid": self.call_sid,
        }


# ============= REST API ENDPOINTS =============


@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Serve the main chat interface"""
    return HTML_TEMPLATE


@app.post("/api/start_session")
async def start_session(request: StartSessionRequest):
    """Start a new verification session"""
    if not request.customer_id:
        raise HTTPException(status_code=400, detail="customer_id is required")

    # Create new session
    session_manager = SessionManager(request.customer_id, request.language)
    active_sessions[session_manager.session_id] = session_manager

    # Start the verification workflow
    session_manager.start()

    return {
        "session_id": session_manager.session_id,
        "customer_id": request.customer_id,
        "call_sid": session_manager.call_sid,
    }


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """Get current session status"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_manager = active_sessions[session_id]
    return session_manager.get_status()


# ============= WEBSOCKET ENDPOINT =============


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()

    if session_id not in active_sessions:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    session_manager = active_sessions[session_id]
    session_manager.websocket = websocket

    try:
        # Send initial status
        await session_manager.send_status_update()

        # Send chat history
        for msg in session_manager.chat_history:
            await websocket.send_json({"type": "message", "data": msg})

        # Listen for incoming messages
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "message":
                message = data.get("message", "").strip()
                if message:
                    await session_manager.send_input(message)

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        session_manager.websocket = None


# ============= HTML TEMPLATE WITH WEBSOCKET =============

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Verification Telecalling Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 1200px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 600;
        }

        .status-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }

        .status-dot.disconnected {
            background: #ef4444;
            animation: none;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-content {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        .sidebar {
            width: 300px;
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
            padding: 20px;
            overflow-y: auto;
        }

        .sidebar h3 {
            font-size: 16px;
            margin-bottom: 15px;
            color: #334155;
        }

        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .stat-label {
            font-size: 12px;
            color: #64748b;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 18px;
            font-weight: 600;
            color: #1e293b;
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #ffffff;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .message {
            display: flex;
            gap: 15px;
            max-width: 80%;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.agent {
            align-self: flex-start;
        }

        .message.customer {
            align-self: flex-end;
            flex-direction: row-reverse;
        }

        .message.system {
            align-self: center;
            max-width: 60%;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            flex-shrink: 0;
            font-size: 20px;
        }

        .message.agent .message-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .message.customer .message-avatar {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }

        .message.system .message-avatar {
            background: #94a3b8;
            color: white;
        }

        .message-content {
            flex: 1;
        }

        .message-bubble {
            padding: 15px 20px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.5;
            word-wrap: break-word;
        }

        .message.agent .message-bubble {
            background: #f1f5f9;
            color: #1e293b;
            border-bottom-left-radius: 4px;
        }

        .message.customer .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.system .message-bubble {
            background: #fef3c7;
            color: #92400e;
            text-align: center;
            border-radius: 10px;
            font-size: 13px;
        }

        .message-time {
            font-size: 11px;
            color: #94a3b8;
            margin-top: 5px;
        }

        .input-container {
            padding: 20px 30px;
            background: #f8fafc;
            border-top: 1px solid #e2e8f0;
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .input-field {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 25px;
            font-size: 15px;
            outline: none;
            transition: border-color 0.3s;
        }

        .input-field:focus {
            border-color: #667eea;
        }

        .send-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }

        .send-button:disabled {
            background: #cbd5e1;
            cursor: not-allowed;
            transform: none;
        }

        .start-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            padding: 40px;
            text-align: center;
        }

        .start-screen h2 {
            font-size: 28px;
            color: #1e293b;
            margin-bottom: 20px;
        }

        .start-screen p {
            color: #64748b;
            margin-bottom: 30px;
            font-size: 16px;
        }

        .start-form {
            width: 100%;
            max-width: 400px;
        }

        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }

        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #334155;
        }

        .form-input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 15px;
            outline: none;
            transition: border-color 0.3s;
        }

        .form-input:focus {
            border-color: #667eea;
        }

        .start-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }

        .start-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .hidden {
            display: none;
        }

        .connection-indicator {
            font-size: 12px;
            color: #64748b;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 Loan Verification Telecalling</h1>
            <div class="status-badge" id="statusBadge">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">Ready</span>
            </div>
        </div>

        <div class="main-content">
            <!-- Start Screen -->
            <div class="chat-container" id="startScreen">
                <div class="start-screen">
                    <h2>Welcome to Loan Verification System</h2>
                    <p>Enter customer details to start the verification call</p>
                    <div class="start-form">
                        <div class="form-group">
                            <label class="form-label">Customer ID</label>
                            <input type="text" class="form-input" id="customerId" placeholder="Enter customer ID" value="Aditya">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Language</label>
                            <select class="form-input" id="language">
                                <option value="english">English</option>
                                <option value="hindi">Hindi</option>
                            </select>
                        </div>
                        <button class="start-button" id="startButton" onclick="startSession()">
                            Start Verification Call
                        </button>
                        <div class="connection-indicator" id="connectionIndicator"></div>
                    </div>
                </div>
            </div>

            <!-- Chat Interface (hidden initially) -->
            <div class="sidebar hidden" id="sidebar">
                <h3>Verification Status</h3>
                <div class="stat-item">
                    <div class="stat-label">Stage</div>
                    <div class="stat-value" id="statStage">-</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Identity Verified</div>
                    <div class="stat-value" id="statIdentity">❌</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Consent Given</div>
                    <div class="stat-value" id="statConsent">❌</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Questions Completed</div>
                    <div class="stat-value" id="statQuestions">0/19</div>
                </div>
            </div>

            <div class="chat-container hidden" id="chatInterface">
                <div class="chat-messages" id="chatMessages">
                    <!-- Messages will be inserted here -->
                </div>
                <div class="input-container">
                    <div class="input-wrapper">
                        <input type="text" class="input-field" id="messageInput" placeholder="Type your response..." onkeypress="handleKeyPress(event)">
                        <button class="send-button" id="sendButton" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let ws = null;
        let reconnectAttempts = 0;
        const MAX_RECONNECT_ATTEMPTS = 5;

        async function startSession() {
            const customerId = document.getElementById('customerId').value.trim();
            const language = document.getElementById('language').value;

            if (!customerId) {
                alert('Please enter a customer ID');
                return;
            }

            const startButton = document.getElementById('startButton');
            startButton.disabled = true;
            startButton.innerHTML = '<span class="loading"></span> Starting...';

            try {
                const response = await fetch('/api/start_session', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ customer_id: customerId, language: language })
                });

                const data = await response.json();

                if (response.ok) {
                    sessionId = data.session_id;
                    
                    // Hide start screen, show chat
                    document.getElementById('startScreen').classList.add('hidden');
                    document.getElementById('sidebar').classList.remove('hidden');
                    document.getElementById('chatInterface').classList.remove('hidden');

                    // Connect WebSocket
                    connectWebSocket();

                    // Update status
                    updateConnectionStatus('Connected', true);
                } else {
                    alert('Error: ' + (data.detail || 'Unknown error'));
                    startButton.disabled = false;
                    startButton.textContent = 'Start Verification Call';
                }
            } catch (error) {
                alert('Connection error: ' + error.message);
                startButton.disabled = false;
                startButton.textContent = 'Start Verification Call';
            }
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('WebSocket connected');
                reconnectAttempts = 0;
                updateConnectionStatus('Connected', true);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'message') {
                    displayMessage(data.data);
                } else if (data.type === 'status') {
                    updateStatus(data.data);
                } else if (data.type === 'error') {
                    console.error('WebSocket error:', data.message);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                updateConnectionStatus('Connection error', false);
            };

            ws.onclose = () => {
                console.log('WebSocket closed');
                updateConnectionStatus('Disconnected', false);
                
                // Attempt to reconnect
                if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                    reconnectAttempts++;
                    console.log(`Reconnecting... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);
                    setTimeout(connectWebSocket, 2000);
                }
            };
        }

        function updateConnectionStatus(status, connected) {
            const statusText = document.getElementById('statusText');
            const statusDot = document.getElementById('statusDot');
            
            statusText.textContent = status;
            
            if (connected) {
                statusDot.classList.remove('disconnected');
            } else {
                statusDot.classList.add('disconnected');
            }
        }

        function displayMessage(message) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${message.speaker}`;

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = message.speaker === 'agent' ? '🤖' : 
                                message.speaker === 'customer' ? '👤' : 'ℹ️';

            const content = document.createElement('div');
            content.className = 'message-content';

            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            bubble.textContent = message.message;

            const time = document.createElement('div');
            time.className = 'message-time';
            const timestamp = new Date(message.timestamp);
            time.textContent = timestamp.toLocaleTimeString();

            content.appendChild(bubble);
            content.appendChild(time);

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);

            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function updateStatus(status) {
            document.getElementById('statStage').textContent = status.stage || '-';
            document.getElementById('statIdentity').textContent = status.identity_verified ? '✅' : '❌';
            document.getElementById('statConsent').textContent = status.consent_given ? '✅' : '❌';
            document.getElementById('statQuestions').textContent = 
                `${status.questions_completed || 0}/${status.total_questions || 19}`;
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message || !ws || ws.readyState !== WebSocket.OPEN) {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('WebSocket not connected. Please wait...');
                }
                return;
            }

            // Send via WebSocket
            ws.send(JSON.stringify({
                type: 'message',
                message: message
            }));

            input.value = '';
            input.focus();
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        // Keep connection alive with periodic pings
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);

        // Focus input when chat becomes visible
        document.addEventListener('DOMContentLoaded', () => {
            const observer = new MutationObserver(() => {
                const chatInterface = document.getElementById('chatInterface');
                if (!chatInterface.classList.contains('hidden')) {
                    document.getElementById('messageInput').focus();
                }
            });

            observer.observe(document.getElementById('chatInterface'), {
                attributes: true,
                attributeFilter: ['class']
            });
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 LOAN VERIFICATION FASTAPI + WEBSOCKET INTERFACE")
    print("=" * 70)
    print("\n✨ Features:")
    print("  - FastAPI with ASGI for async performance")
    print("  - WebSocket for real-time bidirectional communication")
    print("  - No polling - instant message updates")
    print("  - Auto-reconnection on connection loss")
    print("  - Clean separation of logs and chat")
    print("\n📱 Open your browser and go to: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\n⚠️  Make sure your PostgreSQL database is running!")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
