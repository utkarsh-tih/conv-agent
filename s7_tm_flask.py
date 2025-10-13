"""
Flask Web Interface for LangGraph Loan Verification Telecalling Agent
Provides a clean chat interface separated from system logs
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import threading
import queue
import uuid
import sys
from io import StringIO
import os

# Set text mode before importing the main module
os.environ["TEXT_MODE"] = "true"

# Import your existing verification system
from spike7_questionnaire_textmode import (
    create_graph,
    LoanVerificationState,
    save_loan_verification,
)
from langchain_core.messages import HumanMessage
import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = "loan_verification_secret_key_2025"
CORS(app)

# Store active sessions
active_sessions = {}


class SessionManager:
    """Manages individual verification sessions"""

    def __init__(self, customer_id, language="english"):
        self.customer_id = customer_id
        self.language = language
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.call_sid = f"loan_call_{int(time.time())}"

        # Message queue for communication
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Session state
        self.state = self._initialize_state()
        self.graph = create_graph()
        self.thread = None
        self.running = False

        # Chat history for UI
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

    def add_to_chat(self, speaker, message, metadata=None):
        """Add message to chat history"""
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "message": message,
            "metadata": metadata or {},
        }
        self.chat_history.append(chat_entry)
        self.output_queue.put(chat_entry)

    def run_verification(self):
        """Run the verification workflow in background thread"""
        self.running = True

        try:
            # Monkey-patch input() to use our queue
            original_input = __builtins__.input

            def queued_input(prompt=""):
                # Agent is asking for input
                if "You (customer):" in prompt or "CUSTOMER INPUT REQUIRED" in prompt:
                    try:
                        # Wait for user input from web interface
                        user_input = self.input_queue.get(timeout=300)  # 5 min timeout
                        return user_input
                    except queue.Empty:
                        return ""
                return ""

            __builtins__.input = queued_input

            # Capture print statements
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            # Run the graph
            final_state = self.graph.invoke(self.state)

            # Restore stdout
            sys.stdout = old_stdout
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
            save_loan_verification.invoke(
                {
                    "call_sid": final_state["call_sid"],
                    "customer_id": self.customer_id,
                    "transcript": final_state["transcript"],
                    "verification_data": final_state["verification_data"],
                    "audio_paths": final_state["audio_paths"],
                    "identity_verified": final_state.get("identity_verified", False),
                    "consent_given": final_state.get("consent_given", False),
                    "verification_status": verification_status,
                }
            )

            # Send completion message
            self.add_to_chat(
                "system",
                f"Verification completed with status: {verification_status}",
                {"status": verification_status, "final_state": True},
            )

        except Exception as e:
            self.add_to_chat("system", f"Error: {str(e)}", {"error": True})
        finally:
            self.running = False

    def start(self):
        """Start verification in background thread"""
        self.thread = threading.Thread(target=self.run_verification, daemon=True)
        self.thread.start()

    def send_input(self, text):
        """Send user input to the verification workflow"""
        self.input_queue.put(text)
        self.add_to_chat("customer", text)

    def get_messages(self, last_index=0):
        """Get new messages since last_index"""
        return self.chat_history[last_index:]

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
        }


# ============= ROUTES =============


@app.route("/")
def index():
    """Serve the main chat interface"""
    return render_template("index.html")


@app.route("/api/start_session", methods=["POST"])
def start_session():
    """Start a new verification session"""
    data = request.json
    customer_id = data.get("customer_id", "")
    language = data.get("language", "english")

    if not customer_id:
        return jsonify({"error": "customer_id is required"}), 400

    # Create new session
    session_manager = SessionManager(customer_id, language)
    active_sessions[session_manager.session_id] = session_manager

    # Store session_id in Flask session
    session["session_id"] = session_manager.session_id

    # Start the verification workflow
    session_manager.start()

    return jsonify(
        {
            "session_id": session_manager.session_id,
            "customer_id": customer_id,
            "call_sid": session_manager.call_sid,
        }
    )


@app.route("/api/send_message", methods=["POST"])
def send_message():
    """Send customer message to the verification workflow"""
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "No active session"}), 400

    data = request.json
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    session_manager = active_sessions[session_id]
    session_manager.send_input(message)

    return jsonify({"status": "sent", "message": message})


@app.route("/api/get_messages")
def get_messages():
    """Get new messages from the session"""
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "No active session"}), 400

    last_index = int(request.args.get("last_index", 0))
    session_manager = active_sessions[session_id]

    messages = session_manager.get_messages(last_index)

    return jsonify(
        {
            "messages": messages,
            "last_index": len(session_manager.chat_history),
            "status": session_manager.get_status(),
        }
    )


@app.route("/api/get_status")
def get_status():
    """Get current session status"""
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "No active session"}), 400

    session_manager = active_sessions[session_id]
    return jsonify(session_manager.get_status())


@app.route("/api/end_session", methods=["POST"])
def end_session():
    """End current session"""
    session_id = session.get("session_id")
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
        session.pop("session_id", None)
        return jsonify({"status": "session_ended"})

    return jsonify({"error": "No active session"}), 400


# ============= HTML TEMPLATE =============


@app.route("/templates/index.html")
def serve_template():
    """Serve the HTML template"""
    return """
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 Loan Verification Telecalling</h1>
            <div class="status-badge" id="statusBadge">
                <span class="status-dot"></span>
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
        let lastMessageIndex = 0;
        let pollInterval = null;

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

                    // Start polling for messages
                    startPolling();

                    // Update status
                    document.getElementById('statusText').textContent = 'Active';
                } else {
                    alert('Error: ' + data.error);
                    startButton.disabled = false;
                    startButton.textContent = 'Start Verification Call';
                }
            } catch (error) {
                alert('Connection error: ' + error.message);
                startButton.disabled = false;
                startButton.textContent = 'Start Verification Call';
            }
        }

        function startPolling() {
            pollInterval = setInterval(fetchMessages, 1000);
        }

        async function fetchMessages() {
            if (!sessionId) return;

            try {
                const response = await fetch(`/api/get_messages?last_index=${lastMessageIndex}`);
                const data = await response.json();

                if (response.ok && data.messages.length > 0) {
                    data.messages.forEach(msg => displayMessage(msg));
                    lastMessageIndex = data.last_index;

                    // Update status
                    if (data.status) {
                        updateStatus(data.status);
                    }
                }
            } catch (error) {
                console.error('Error fetching messages:', error);
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

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message || !sessionId) return;

            const sendButton = document.getElementById('sendButton');
            sendButton.disabled = true;

            try {
                const response = await fetch('/api/send_message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });

                if (response.ok) {
                    input.value = '';
                }
            } catch (error) {
                console.error('Error sending message:', error);
            } finally {
                sendButton.disabled = false;
                input.focus();
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

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
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)

    # Save the HTML template
    html_content = """{{ DO NOT COPY - Will be auto-generated }}"""

    print("\n" + "=" * 70)
    print("🌐 LOAN VERIFICATION WEB INTERFACE")
    print("=" * 70)
    print("\n🚀 Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("\n💡 Features:")
    print("  - Clean chat interface separated from logs")
    print("  - Real-time message updates")
    print("  - Live verification status tracking")
    print("  - Agent messages displayed clearly")
    print("\n⚠️  Make sure your PostgreSQL database is running!")
    print("=" * 70 + "\n")

    app.run(debug=True, port=5000, use_reloader=False)
