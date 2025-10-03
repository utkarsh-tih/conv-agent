"""
LangGraph Telecalling AI Agent with Twilio Integration
Supports: TTS/STT, Multilingual (English + Indic), Local deployment, Scoring

Key: Uses LOCAL LLM (Llama 3.1, Mistral, or Gemma) for tool selection and reasoning
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import operator
import json
from datetime import datetime
import os

# ============= LOCAL LLM CONFIGURATION =============

def initialize_local_llm(model_choice: str = "llama3.1"):
    """
    Initialize local LLM for tool calling and decision making
    
    Options:
    1. Ollama (Recommended for India deployment):
       - Llama 3.1 8B/70B (Meta)
       - Mistral 7B/8x7B
       - Gemma 2 9B/27B (Google)
       - Qwen 2.5 (Alibaba - good for multilingual)
    
    2. vLLM (for production scale):
       - Self-hosted inference server
       - Better performance for high-throughput
    
    3. LocalAI:
       - OpenAI-compatible API
       - Runs on CPU/GPU
    """
    
    if model_choice == "llama3.1":
        # Using Ollama with Llama 3.1 8B (good balance of speed/quality)
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.3,  # Lower for consistent tool calling
            base_url="http://localhost:11434",  # Ollama default
            # For tool calling, need structured output
            format="json" if "tool" in model_choice else None
        )
    
    elif model_choice == "mistral":
        llm = ChatOllama(
            model="mistral:7b",
            temperature=0.3,
            base_url="http://localhost:11434"
        )
    
    elif model_choice == "qwen":
        # Qwen 2.5 is excellent for multilingual (including Indic languages)
        llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0.3,
            base_url="http://localhost:11434"
        )
    
    elif model_choice == "vllm":
        # For production deployment with vLLM
        llm = ChatOpenAI(
            base_url="http://localhost:8000/v1",  # vLLM server
            api_key="EMPTY",  # vLLM doesn't need API key for local
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.3
        )
    
    else:
        # Default to Llama 3.1
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.3
        )
    
    return llm


# Create global LLM instance
LOCAL_LLM = initialize_local_llm("llama3.1")


# ============= STATE DEFINITION =============
class AgentState(TypedDict):
    """Complete state for the telecalling agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_data: dict  # JSON customer information
    call_sid: str  # Twilio call identifier
    language: str  # Current conversation language
    transcript: list  # Full conversation transcript
    audio_chunks: list  # Stored audio segments
    customer_score: dict  # Scoring metrics
    current_question: str  # Current question being asked
    retry_count: int  # Number of retries for current question
    audio_quality: str  # "clear" | "unclear" | "very_poor"
    conversation_stage: str  # "greeting" | "questioning" | "closing"
    needs_clarification: bool
    error_type: str  # "unclear_audio" | "wrong_answer" | "no_response" | None


# ============= TOOLS DEFINITION =============

@tool
def get_customer_data(customer_id: str) -> dict:
    """
    Fetch customer information from database
    Args:
        customer_id: Unique customer identifier
    Returns:
        Customer data as JSON
    """
    # Mock implementation - replace with actual DB call
    return {
        "customer_id": customer_id,
        "name": "Sample Customer",
        "phone": "+91XXXXXXXXXX",
        "language_preference": "hindi",
        "previous_interactions": 2,
        "risk_category": "medium"
    }


@tool
def text_to_speech(text: str, language: str) -> dict:
    """
    Convert text to speech using local TTS engine
    Supports English and Indic languages (Hindi, Tamil, Telugu, etc.)
    Args:
        text: Text to convert to speech
        language: Language code (en, hi, ta, te, etc.)
    Returns:
        Audio file path and metadata
    """
    # Use local TTS engines like:
    # - Coqui TTS (open source, multilingual)
    # - Mozilla TTS
    # - Indic TTS models from AI4Bharat
    
    return {
        "audio_path": f"audio/{datetime.now().timestamp()}.wav",
        "duration": 3.5,
        "language": language,
        "text": text
    }


@tool
def speech_to_text(audio_path: str, language: str) -> dict:
    """
    Convert speech to text using local STT engine
    Args:
        audio_path: Path to audio file
        language: Expected language
    Returns:
        Transcribed text with confidence score
    """
    # Use local STT engines like:
    # - Whisper (OpenAI, runs locally)
    # - Wav2Vec2 models from AI4Bharat for Indic languages
    # - Vosk (offline STT)
    
    return {
        "text": "Sample transcribed text",
        "confidence": 0.87,
        "language": language,
        "audio_quality": "clear"  # or "unclear" or "very_poor"
    }


@tool
def store_transcript(call_sid: str, transcript: list) -> bool:
    """
    Store conversation transcript to secure local database
    Args:
        call_sid: Twilio call identifier
        transcript: List of conversation turns
    Returns:
        Success status
    """
    # Store in local database (PostgreSQL/MongoDB)
    # Ensure encryption for sensitive data
    return True


@tool
def store_audio(call_sid: str, audio_path: str, metadata: dict) -> bool:
    """
    Store audio file to secure local storage
    Args:
        call_sid: Twilio call identifier
        audio_path: Path to audio file
        metadata: Audio metadata
    Returns:
        Success status
    """
    # Store in local file system with encryption
    # Or use local object storage like MinIO
    return True


@tool
def calculate_customer_score(transcript: list, customer_data: dict) -> dict:
    """
    Calculate customer scoring based on conversation
    Args:
        transcript: Full conversation transcript
        customer_data: Customer information
    Returns:
        Scoring metrics
    """
    # Implement scoring logic based on:
    # - Response clarity
    # - Engagement level
    # - Sentiment analysis
    # - Specific answer validation
    
    return {
        "overall_score": 75,
        "engagement_score": 80,
        "clarity_score": 70,
        "compliance_score": 75,
        "risk_level": "medium",
        "next_action": "follow_up_required"
    }


@tool
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text between languages
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
    Returns:
        Translated text
    """
    # Use local translation models:
    # - AI4Bharat IndicTrans2 for Indic languages
    # - MarianMT models
    return f"Translated: {text}"


@tool
def validate_audio_quality(audio_metrics: dict) -> dict:
    """
    Validate audio quality and determine if retry needed
    Args:
        audio_metrics: Audio quality metrics
    Returns:
        Quality assessment and action
    """
    confidence = audio_metrics.get("confidence", 0)
    
    if confidence < 0.5:
        return {"quality": "very_poor", "action": "retry", "message": "unclear"}
    elif confidence < 0.7:
        return {"quality": "unclear", "action": "clarify", "message": "partial"}
    else:
        return {"quality": "clear", "action": "proceed", "message": "good"}


@tool
def twilio_make_call(phone_number: str, callback_url: str) -> dict:
    """
    Initiate outbound call via Twilio
    Args:
        phone_number: Customer phone number
        callback_url: Webhook URL for call events
    Returns:
        Call SID and status
    """
    # Integrate with Twilio API
    return {
        "call_sid": "CA1234567890",
        "status": "initiated",
        "timestamp": datetime.now().isoformat()
    }


@tool
def twilio_send_sms(phone_number: str, message: str, language: str) -> bool:
    """
    Send SMS via Twilio (for follow-ups)
    Args:
        phone_number: Customer phone number
        message: SMS content
        language: Message language
    Returns:
        Success status
    """
    return True


# ============= NODE FUNCTIONS =============

def initiate_call_node(state: AgentState) -> AgentState:
    """Initialize call and greet customer (agent-first approach)"""
    customer_data = state["customer_data"]
    language = customer_data.get("language_preference", "english")
    
    # Use LLM to generate personalized greeting
    system_prompt = f"""You are a professional telecalling AI agent for a financial/business organization in India.
Customer Details: {json.dumps(customer_data)}
Language: {language}
Task: Generate a professional, warm greeting to start the call. Keep it brief (2-3 sentences).
The agent speaks first, not the customer."""

    response = LOCAL_LLM.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Generate the initial greeting for this customer.")
    ])
    
    greeting = response.content
    
    # If language is not English, translate
    if language != "english":
        greeting = translate_text.invoke({
            "text": greeting,
            "source_lang": "english",
            "target_lang": language
        })
    
    # Convert to speech and initiate call
    tts_result = text_to_speech.invoke({"text": greeting, "language": language})
    
    state["messages"].append(AIMessage(content=greeting))
    state["language"] = language
    state["conversation_stage"] = "greeting"
    state["current_question"] = greeting
    state["retry_count"] = 0
    
    return state


def listen_and_transcribe_node(state: AgentState) -> AgentState:
    """Capture customer response and transcribe"""
    # Simulate receiving audio from Twilio
    audio_path = "temp_audio.wav"  # From Twilio stream
    
    # Transcribe audio
    stt_result = speech_to_text.invoke({
        "audio_path": audio_path,
        "language": state["language"]
    })
    
    # Validate audio quality
    quality_check = validate_audio_quality.invoke({"audio_metrics": stt_result})
    
    state["audio_quality"] = quality_check["quality"]
    state["transcript"].append({
        "timestamp": datetime.now().isoformat(),
        "speaker": "customer",
        "text": stt_result["text"],
        "confidence": stt_result["confidence"],
        "language": state["language"]
    })
    
    # Store audio
    store_audio.invoke({
        "call_sid": state["call_sid"],
        "audio_path": audio_path,
        "metadata": stt_result
    })
    
    if quality_check["quality"] in ["unclear", "very_poor"]:
        state["needs_clarification"] = True
        state["error_type"] = "unclear_audio"
    else:
        state["needs_clarification"] = False
        state["messages"].append(HumanMessage(content=stt_result["text"]))
    
    return state


def handle_unclear_audio_node(state: AgentState) -> AgentState:
    """Handle unclear audio by asking customer to repeat"""
    state["retry_count"] += 1
    
    if state["retry_count"] > 3:
        # Too many retries, move to closing
        state["conversation_stage"] = "closing"
        return state
    
    clarification_messages = {
        "english": "I'm sorry, I couldn't hear you clearly. Could you please repeat that?",
        "hindi": "क्षमा करें, मैं आपको स्पष्ट रूप से नहीं सुन सका। क्या आप कृपया दोहरा सकते हैं?",
        "tamil": "மன்னிக்கவும், உங்களை தெளிவாகக் கேட்க முடியவில்லை. தயவுசெய்து மீண்டும் சொல்ல முடியுமா?",
    }
    
    message = clarification_messages.get(state["language"], clarification_messages["english"])
    
    # Convert to speech
    text_to_speech.invoke({"text": message, "language": state["language"]})
    
    state["messages"].append(AIMessage(content=message))
    state["needs_clarification"] = False
    
    return state


def validate_answer_node(state: AgentState) -> AgentState:
    """Validate customer's answer against expected response using LLM"""
    last_customer_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_customer_message = msg.content
            break
    
    # Use LLM to validate the answer
    validation_prompt = f"""You are validating a customer's response in a telecalling conversation.

Current Question: {state['current_question']}
Customer's Answer: {last_customer_message}
Conversation Stage: {state['conversation_stage']}

Task: Determine if the answer is:
1. Valid and relevant to the question
2. Requires clarification
3. Completely off-topic or nonsensical

Respond with ONLY a JSON object:
{{"is_valid": true/false, "reason": "brief explanation", "needs_clarification": true/false}}"""

    response = LOCAL_LLM.invoke([
        SystemMessage(content="You are a validation assistant. Always respond with valid JSON only."),
        HumanMessage(content=validation_prompt)
    ])
    
    try:
        validation_result = json.loads(response.content)
        is_valid = validation_result.get("is_valid", False)
        needs_clarification = validation_result.get("needs_clarification", False)
        
        if not is_valid or needs_clarification:
            state["error_type"] = "wrong_answer"
            state["needs_clarification"] = True
            state["retry_count"] += 1
        else:
            state["error_type"] = None
            state["retry_count"] = 0
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        is_valid = len(last_customer_message or "") > 5
        if not is_valid:
            state["error_type"] = "wrong_answer"
            state["needs_clarification"] = True
            state["retry_count"] += 1
    
    return state


def ask_question_node(state: AgentState) -> AgentState:
    """Ask next question based on conversation flow - LLM decides what to ask"""
    
    # Build conversation context for LLM
    conversation_history = "\n".join([
        f"{'Agent' if isinstance(msg, AIMessage) else 'Customer'}: {msg.content}"
        for msg in state["messages"][-6:]  # Last 3 exchanges
    ])
    
    system_prompt = f"""You are a professional telecalling AI agent conducting a customer call.

Customer Data: {json.dumps(state['customer_data'])}
Current Stage: {state['conversation_stage']}
Language: {state['language']}
Previous Conversation:
{conversation_history}

Task: Generate the NEXT question to ask the customer. 
- Keep questions clear, concise, and professional
- One question at a time
- If previous answer was unclear, rephrase for clarity
- Progress the conversation towards completion
- If error_type is 'wrong_answer', politely ask for clarification

{"IMPORTANT: The customer gave an unclear or wrong answer. Politely ask them to clarify or provide the correct information." if state.get('error_type') == 'wrong_answer' else ''}

Respond with ONLY the question text, nothing else."""

    response = LOCAL_LLM.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="What should I ask next?")
    ])
    
    question = response.content.strip()
    
    # Translate if needed
    if state["language"] != "english":
        question = translate_text.invoke({
            "text": question,
            "source_lang": "english",
            "target_lang": state["language"]
        })
    
    # Convert to speech
    text_to_speech.invoke({"text": question, "language": state["language"]})
    
    state["messages"].append(AIMessage(content=question))
    state["current_question"] = question
    state["transcript"].append({
        "timestamp": datetime.now().isoformat(),
        "speaker": "agent",
        "text": question,
        "language": state["language"]
    })
    
    # Update conversation stage based on conversation flow
    if len(state["messages"]) > 10:  # After multiple exchanges
        state["conversation_stage"] = "closing"
    elif state["conversation_stage"] == "greeting":
        state["conversation_stage"] = "questioning"
    
    return state


def score_and_store_node(state: AgentState) -> AgentState:
    """Calculate final score using LLM and store all data"""
    
    # Use LLM to generate comprehensive customer score
    conversation_text = "\n".join([
        f"{'Agent' if isinstance(msg, AIMessage) else 'Customer'}: {msg.content}"
        for msg in state["messages"]
    ])
    
    scoring_prompt = f"""Analyze this completed telecalling conversation and provide a comprehensive customer score.

Conversation:
{conversation_text}

Customer Data: {json.dumps(state['customer_data'])}

Evaluate based on:
1. Engagement: How engaged was the customer? (0-100)
2. Clarity: How clear were their responses? (0-100)
3. Cooperation: How cooperative were they? (0-100)
4. Information Quality: Quality of information provided (0-100)
5. Overall Risk Level: low/medium/high

Respond with ONLY this JSON structure:
{{
    "overall_score": <0-100>,
    "engagement_score": <0-100>,
    "clarity_score": <0-100>,
    "cooperation_score": <0-100>,
    "information_quality_score": <0-100>,
    "risk_level": "low/medium/high",
    "next_action": "suggested next action",
    "key_insights": ["insight1", "insight2"],
    "red_flags": ["flag1", "flag2"] or []
}}"""

    response = LOCAL_LLM.invoke([
        SystemMessage(content="You are a customer interaction analyst. Respond with valid JSON only."),
        HumanMessage(content=scoring_prompt)
    ])
    
    try:
        score = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback scoring if LLM fails
        score = {
            "overall_score": 70,
            "engagement_score": 70,
            "clarity_score": 70,
            "cooperation_score": 70,
            "information_quality_score": 70,
            "risk_level": "medium",
            "next_action": "manual_review_required",
            "key_insights": [],
            "red_flags": []
        }
    
    state["customer_score"] = score
    
    # Store transcript with scores
    store_transcript.invoke({
        "call_sid": state["call_sid"],
        "transcript": state["transcript"]
    })
    
    return state


def routing_function(state: AgentState) -> Literal["handle_unclear", "validate", "ask_question", "decide_next", "end"]:
    """Route to next node based on state - can also use LLM for complex routing"""
    
    # Handle unclear audio first
    if state["needs_clarification"] and state["error_type"] == "unclear_audio":
        return "handle_unclear"
    
    # If conversation stage is closing, end
    if state["conversation_stage"] == "closing" and len(state["messages"]) > 8:
        return "end"
    
    # If we have a customer response, validate it
    if len(state["messages"]) > 0 and isinstance(state["messages"][-1], HumanMessage):
        return "validate"
    
    # Check if we need to re-ask due to wrong answer
    if state["error_type"] == "wrong_answer" and state["retry_count"] < 3:
        return "ask_question"
    
    # Use LLM to decide if we should continue or end (for complex cases)
    if len(state["messages"]) > 6:
        return "decide_next"
    
    return "ask_question"


def llm_decide_next_node(state: AgentState) -> AgentState:
    """Use LLM to decide whether to continue or end conversation"""
    
    conversation_summary = "\n".join([
        f"{'Agent' if isinstance(msg, AIMessage) else 'Customer'}: {msg.content}"
        for msg in state["messages"]
    ])
    
    decision_prompt = f"""Analyze this telecalling conversation and decide if it should continue or end.

Conversation:
{conversation_summary}

Customer Data: {json.dumps(state['customer_data'])}
Current Stage: {state['conversation_stage']}

Respond with JSON only:
{{"decision": "continue" or "end", "reason": "brief explanation"}}"""

    response = LOCAL_LLM.invoke([
        SystemMessage(content="You are a conversation flow manager. Respond with valid JSON only."),
        HumanMessage(content=decision_prompt)
    ])
    
    try:
        decision = json.loads(response.content)
        if decision.get("decision") == "end":
            state["conversation_stage"] = "closing"
    except json.JSONDecodeError:
        pass  # Continue normally if parsing fails
    
    return state


# ============= GRAPH CONSTRUCTION =============

def create_telecalling_graph():
    """Construct the complete LangGraph workflow"""
    
    # Create tools list
    tools = [
        get_customer_data,
        text_to_speech,
        speech_to_text,
        store_transcript,
        store_audio,
        calculate_customer_score,
        translate_text,
        validate_audio_quality,
        twilio_make_call,
        twilio_send_sms
    ]
    
    # Create tool node for tool execution
    tool_node = ToolNode(tools)
    
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("initiate_call", initiate_call_node)
    workflow.add_node("listen", listen_and_transcribe_node)
    workflow.add_node("handle_unclear", handle_unclear_audio_node)
    workflow.add_node("validate", validate_answer_node)
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("decide_next", llm_decide_next_node)  # New LLM decision node
    workflow.add_node("score_and_store", score_and_store_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point (agent-first approach)
    workflow.set_entry_point("initiate_call")
    
    # Add edges
    workflow.add_edge("initiate_call", "listen")
    workflow.add_conditional_edges(
        "listen",
        routing_function,
        {
            "handle_unclear": "handle_unclear",
            "validate": "validate",
            "ask_question": "ask_question",
            "decide_next": "decide_next",
            "end": "score_and_store"
        }
    )
    workflow.add_edge("handle_unclear", "listen")
    workflow.add_edge("validate", "ask_question")
    workflow.add_edge("ask_question", "listen")
    workflow.add_conditional_edges(
        "decide_next",
        lambda s: "end" if s["conversation_stage"] == "closing" else "ask_question",
        {
            "ask_question": "ask_question",
            "end": "score_and_store"
        }
    )
    workflow.add_edge("score_and_store", END)
    
    return workflow.compile()


# ============= USAGE EXAMPLE =============

def run_telecalling_agent(customer_id: str, phone_number: str):
    """Main function to run the telecalling agent"""
    
    # Fetch customer data
    customer_data = get_customer_data.invoke({"customer_id": customer_id})
    
    # Initiate Twilio call
    call_result = twilio_make_call.invoke({
        "phone_number": phone_number,
        "callback_url": "https://your-server.in/twilio/callback"
    })
    
    # Initialize state
    initial_state = {
        "messages": [],
        "customer_data": customer_data,
        "call_sid": call_result["call_sid"],
        "language": customer_data.get("language_preference", "english"),
        "transcript": [],
        "audio_chunks": [],
        "customer_score": {},
        "current_question": "",
        "retry_count": 0,
        "audio_quality": "clear",
        "conversation_stage": "greeting",
        "needs_clarification": False,
        "error_type": None
    }
    
    # Create and run graph
    app = create_telecalling_graph()
    final_state = app.invoke(initial_state)
    
    return final_state


# Example usage
if __name__ == "__main__":
    result = run_telecalling_agent(
        customer_id="CUST123",
        phone_number="+91XXXXXXXXXX"
    )
    print(f"Call completed. Customer score: {result['customer_score']}")


# ============= DEPLOYMENT SETUP GUIDE =============
"""
STEP-BY-STEP LOCAL DEPLOYMENT IN INDIA:

1. INSTALL OLLAMA (Local LLM Server):
   ```bash
   # Linux/Mac
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Pull recommended model (Llama 3.1 8B - ~5GB)
   ollama pull llama3.1:8b
   
   # Alternative: Qwen 2.5 for better Indic language support
   ollama pull qwen2.5:7b
   ```

2. INSTALL DEPENDENCIES:
   ```bash
   pip install langgraph langchain langchain-community
   pip install twilio
   pip install openai-whisper  # For STT
   pip install TTS  # Coqui TTS for text-to-speech
   pip install indic-nlp-library  # For Indic language support
   pip install sqlalchemy psycopg2  # For database
   ```

3. LOCAL TTS/STT SETUP:
   
   A. For Speech-to-Text (Whisper):
   ```python
   import whisper
   model = whisper.load_model("base")  # or "small", "medium", "large"
   result = model.transcribe("audio.wav", language="hi")  # hi=Hindi
   ```
   
   B. For Text-to-Speech (Coqui TTS):
   ```python
   from TTS.api import TTS
   # For English
   tts_en = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
   
   # For Indic languages (using AI4Bharat models)
   # Download from: https://github.com/AI4Bharat/Indic-TTS
   ```

4. DATABASE SETUP (PostgreSQL):
   ```bash
   # Install PostgreSQL
   sudo apt install postgresql
   
   # Create database
   sudo -u postgres psql
   CREATE DATABASE telecalling_db;
   CREATE USER telecall_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE telecalling_db TO telecall_user;
   ```
   
   Schema:
   ```sql
   CREATE TABLE call_transcripts (
       call_sid VARCHAR(50) PRIMARY KEY,
       customer_id VARCHAR(50),
       transcript JSONB,
       audio_paths TEXT[],
       customer_score JSONB,
       created_at TIMESTAMP DEFAULT NOW()
   );
   
   CREATE INDEX idx_customer_id ON call_transcripts(customer_id);
   CREATE INDEX idx_created_at ON call_transcripts(created_at);
   ```

5. TWILIO CONFIGURATION:
   ```python
   # Set environment variables
   export TWILIO_ACCOUNT_SID="your_account_sid"
   export TWILIO_AUTH_TOKEN="your_auth_token"
   export TWILIO_PHONE_NUMBER="+91XXXXXXXXXX"
   
   # Webhook URL (use ngrok for testing)
   ngrok http 8000
   # Set this URL in Twilio console
   ```

6. SERVER SETUP (India):
   
   Options:
   - AWS Mumbai (ap-south-1) region
   - Azure India regions
   - Self-hosted on-premise server
   
   Recommended specs for production:
   - CPU: 8+ cores (or GPU: 16GB+ VRAM for faster inference)
   - RAM: 32GB+
   - Storage: 500GB+ SSD (for audio storage)
   - OS: Ubuntu 22.04 LTS

7. RUNNING THE APPLICATION:
   ```bash
   # Start Ollama
   ollama serve
   
   # Start your FastAPI/Flask server
   python app.py
   
   # For production: Use Gunicorn + Nginx
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
   ```

8. MONITORING & LOGGING:
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('telecalling_agent.log'),
           logging.StreamHandler()
       ]
   )
   ```

9. SECURITY CONSIDERATIONS:
   - Encrypt audio files at rest (AES-256)
   - Use VPN for Twilio connection if needed
   - Implement rate limiting
   - Regular security audits
   - GDPR/Data protection compliance

10. PERFORMANCE OPTIMIZATION:
    - Use vLLM instead of Ollama for production (3-5x faster)
    - Implement audio file compression
    - Use Redis for caching customer data
    - Batch processing for transcripts
    - Load balancer for multiple concurrent calls

COST ESTIMATE (Monthly):
- Server (AWS/Azure India): ₹15,000 - ₹50,000
- Twilio calls: ~₹1-2 per minute
- Storage: ₹500 - ₹2,000
- Total: ~₹20,000 - ₹60,000/month for 1000 calls/month

No API costs for LLM since it's running locally!
"""