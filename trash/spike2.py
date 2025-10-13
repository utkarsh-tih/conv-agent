"""
Complete LangGraph Telecalling AI Agent with Twilio Integration
Updated for LangGraph v0.2+ with CAM-based questioning
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
import json
from datetime import datetime
import logging
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============= ENUMS AND CONSTANTS =============

class ConversationStage(str, Enum):
    INIT = "init"
    CONSENT = "consent"
    IDENTITY = "identity"
    QUESTIONING = "questioning"
    SCORING = "scoring"
    CLOSING = "closing"
    HANDOFF = "handoff"


class ResponseType(str, Enum):
    CLEAR = "clear"
    UNCLEAR = "unclear"
    OFF_TOPIC = "off_topic"
    HUMAN_REQUEST = "human_request"
    ANGER = "anger"
    LANGUAGE_ISSUE = "language_issue"


# Credit Assessment Model (CAM) - Sample question bank
CAM_QUESTIONS = {
    "financial_stability": [
        "What is your current monthly income?",
        "Do you have any existing loans or debts?",
        "What is your current employment status?"
    ],
    "repayment_history": [
        "Have you ever defaulted on a loan payment?",
        "How would you rate your payment discipline?",
        "Do you have an active credit card?"
    ],
    "assets": [
        "Do you own any property or vehicle?",
        "What is your primary source of income?",
        "Do you have any savings or investments?"
    ],
    "personal_info": [
        "Can you confirm your date of birth?",
        "What is your current residential address?",
        "How long have you been at your current residence?"
    ]
}


# ============= STATE DEFINITION =============

class AgentState(TypedDict):
    """Complete state for telecalling agent following the flowchart"""
    # Core state
    messages: list[BaseMessage]
    customer_data: dict
    call_sid: str
    language: str
    
    # Conversation tracking
    transcript: list[dict]
    audio_chunks: list[str]
    conversation_stage: ConversationStage
    
    # Consent tracking
    consent_given: bool
    consent_attempts: int
    
    # Identity verification
    identity_score: float
    identity_attempts: int
    identity_verified: bool
    
    # Question management
    current_question: str
    current_category: str
    questions_asked: list[dict]
    questions_remaining: list[dict]
    retry_count: int
    
    # Response analysis
    last_response_type: ResponseType
    audio_quality: str
    needs_clarification: bool
    
    # Scoring and results
    credit_score: dict
    final_report: dict
    
    # Error handling
    error_type: str | None
    handoff_reason: str | None


# ============= LLM INITIALIZATION =============

def initialize_llm(model: str = "llama3.1:8b"):
    """Initialize local LLM"""
    return ChatOllama(
        model=model,
        temperature=0.3,
        base_url="http://localhost:11434"
    )

LOCAL_LLM = initialize_llm()


# ============= TOOL IMPLEMENTATIONS =============

def get_customer_data(customer_id: str) -> dict:
    """Fetch customer data from database"""
    # Mock implementation - replace with actual DB query
    logger.info(f"Fetching data for customer: {customer_id}")
    return {
        "customer_id": customer_id,
        "name": "Rajesh Kumar",
        "phone": "+919876543210",
        "language_preference": "english",
        "date_of_birth": "1985-06-15",
        "address": "Mumbai, Maharashtra",
        "previous_interactions": 2,
        "risk_category": "medium"
    }


def text_to_speech(text: str, language: str) -> dict:
    """Convert text to speech - integrate with local TTS"""
    logger.info(f"TTS: {text[:50]}...")
    # In production, use Coqui TTS or similar
    return {
        "audio_path": f"audio/tts_{datetime.now().timestamp()}.wav",
        "duration": len(text) / 15,  # Rough estimate
        "language": language
    }


def speech_to_text(audio_path: str, language: str) -> dict:
    """Convert speech to text - integrate with Whisper or similar"""
    logger.info(f"STT for audio: {audio_path}")
    # Mock response - in production, use Whisper
    return {
        "text": "Yes, I agree to proceed with the call",
        "confidence": 0.89,
        "language": language,
        "audio_quality": "clear"
    }


def store_transcript(call_sid: str, transcript: list) -> bool:
    """Store transcript to database"""
    logger.info(f"Storing transcript for call: {call_sid}")
    # In production, store to PostgreSQL
    return True


def store_audio(call_sid: str, audio_path: str) -> bool:
    """Store audio file with encryption"""
    logger.info(f"Storing audio: {audio_path}")
    # In production, encrypt and store
    return True


def twilio_make_call(phone_number: str, callback_url: str) -> dict:
    """Initiate Twilio call"""
    logger.info(f"Initiating call to: {phone_number}")
    return {
        "call_sid": f"CA{datetime.now().timestamp()}",
        "status": "initiated"
    }


def send_sms_notification(phone_number: str, message: str) -> bool:
    """Send SMS via Twilio"""
    logger.info(f"Sending SMS to: {phone_number}")
    return True


# ============= NODE FUNCTIONS =============

def initialize_session_node(state: AgentState) -> AgentState:
    """Initialize call session"""
    logger.info("=== INITIALIZING SESSION ===")
    
    state["conversation_stage"] = ConversationStage.INIT
    state["consent_given"] = False
    state["consent_attempts"] = 0
    state["identity_verified"] = False
    state["identity_attempts"] = 0
    state["identity_score"] = 0.0
    state["questions_asked"] = []
    state["retry_count"] = 0
    state["transcript"] = []
    state["audio_chunks"] = []
    
    # Generate initial greeting
    greeting = f"Hello, this is an automated call from [Company Name]. Am I speaking with {state['customer_data']['name']}?"
    
    if state["language"] != "english":
        greeting = translate_text(greeting, "english", state["language"])
    
    text_to_speech(greeting, state["language"])
    
    state["messages"].append(AIMessage(content=greeting))
    state["transcript"].append({
        "timestamp": datetime.now().isoformat(),
        "speaker": "agent",
        "text": greeting
    })
    
    return state


def consent_node(state: AgentState) -> AgentState:
    """Handle consent collection"""
    logger.info("=== CONSENT NODE ===")
    
    state["conversation_stage"] = ConversationStage.CONSENT
    
    if state["consent_attempts"] == 0:
        consent_msg = """I'm calling regarding your loan application. This call will be recorded for quality and training purposes. 
The conversation will take approximately 5-7 minutes. Do you consent to proceed with this call?"""
        
        if state["language"] != "english":
            consent_msg = translate_text(consent_msg, "english", state["language"])
        
        text_to_speech(consent_msg, state["language"])
        
        state["messages"].append(AIMessage(content=consent_msg))
        state["current_question"] = consent_msg
        state["consent_attempts"] += 1
    
    return state


def analyze_consent_response(state: AgentState) -> AgentState:
    """Analyze customer's consent response using LLM"""
    
    last_response = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_response = msg.content
            break
    
    if not last_response:
        return state
    
    prompt = f"""Analyze if the customer gave consent to proceed with the call.

Customer's response: "{last_response}"

Respond with JSON only:
{{"consent_given": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    response = LOCAL_LLM.invoke([
        SystemMessage(content="You are a consent analyzer. Respond with valid JSON only."),
        HumanMessage(content=prompt)
    ])
    
    try:
        result = json.loads(response.content)
        if result["consent_given"] and result["confidence"] > 0.7:
            state["consent_given"] = True
            logger.info("✓ Consent obtained")
        else:
            state["consent_given"] = False
            logger.info("✗ Consent not given")
    except json.JSONDecodeError:
        state["consent_given"] = False
    
    return state


def identity_verification_node(state: AgentState) -> AgentState:
    """Verify customer identity"""
    logger.info("=== IDENTITY VERIFICATION ===")
    
    state["conversation_stage"] = ConversationStage.IDENTITY
    
    # Generate identity verification questions
    questions = [
        f"Can you confirm your date of birth?",
        f"What is your current residential address?",
        f"Can you confirm the last 4 digits of your registered mobile number?"
    ]
    
    question = questions[min(state["identity_attempts"], len(questions)-1)]
    
    if state["language"] != "english":
        question = translate_text(question, "english", state["language"])
    
    text_to_speech(question, state["language"])
    
    state["messages"].append(AIMessage(content=question))
    state["current_question"] = question
    
    return state


def calculate_identity_score(state: AgentState) -> AgentState:
    """Calculate identity verification score using LLM"""
    
    last_response = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_response = msg.content
            break
    
    prompt = f"""Verify if the customer's response matches the expected identity information.

Expected DOB: {state['customer_data']['date_of_birth']}
Expected Address: {state['customer_data']['address']}
Expected Phone: {state['customer_data']['phone']}

Customer's response: "{last_response}"
Question asked: "{state['current_question']}"

Calculate identity score (0.0-1.0) based on how well the response matches.
Respond with JSON only:
{{"identity_score": 0.0-1.0, "matches": true/false, "reason": "explanation"}}"""

    response = LOCAL_LLM.invoke([
        SystemMessage(content="You are an identity verification system. Respond with valid JSON only."),
        HumanMessage(content=prompt)
    ])
    
    try:
        result = json.loads(response.content)
        score = result.get("identity_score", 0.0)
        
        # Average with previous scores
        if state["identity_score"] > 0:
            state["identity_score"] = (state["identity_score"] + score) / 2
        else:
            state["identity_score"] = score
        
        state["identity_attempts"] += 1
        
        # Threshold for verification
        if state["identity_score"] >= 0.75:
            state["identity_verified"] = True
            logger.info(f"✓ Identity verified (score: {state['identity_score']:.2f})")
        else:
            logger.info(f"Identity score: {state['identity_score']:.2f} (attempts: {state['identity_attempts']})")
            
    except json.JSONDecodeError:
        state["identity_score"] = 0.0
    
    return state


def generate_questions_from_cam(state: AgentState) -> AgentState:
    """Generate questions from CAM (Credit Assessment Model)"""
    logger.info("=== GENERATING CAM QUESTIONS ===")
    
    # Select questions from each category
    questions = []
    for category, q_list in CAM_QUESTIONS.items():
        for q in q_list[:2]:  # Take first 2 from each category
            questions.append({
                "category": category,
                "question": q,
                "asked": False
            })
    
    state["questions_remaining"] = questions
    state["conversation_stage"] = ConversationStage.QUESTIONING
    
    logger.info(f"Generated {len(questions)} questions from CAM")
    return state


def dynamic_qa_node(state: AgentState) -> AgentState:
    """Dynamic Q&A node - ask next question"""
    logger.info("=== DYNAMIC Q&A NODE ===")
    
    # Find next unanswered question
    next_question = None
    for q in state["questions_remaining"]:
        if not q["asked"]:
            next_question = q
            break
    
    if not next_question:
        # All questions asked
        state["conversation_stage"] = ConversationStage.SCORING
        return state
    
    question_text = next_question["question"]
    
    if state["language"] != "english":
        question_text = translate_text(question_text, "english", state["language"])
    
    text_to_speech(question_text, state["language"])
    
    state["messages"].append(AIMessage(content=question_text))
    state["current_question"] = question_text
    state["current_category"] = next_question["category"]
    
    # Mark as asked
    next_question["asked"] = True
    
    state["transcript"].append({
        "timestamp": datetime.now().isoformat(),
        "speaker": "agent",
        "text": question_text,
        "category": next_question["category"]
    })
    
    return state


def analyze_response_node(state: AgentState) -> AgentState:
    """Analyze customer's response"""
    logger.info("=== ANALYZING RESPONSE ===")
    
    last_response = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_response = msg.content
            break
    
    if not last_response:
        return state
    
    prompt = f"""Analyze the customer's response in a telecalling conversation.

Question asked: "{state['current_question']}"
Customer's response: "{last_response}"

Classify the response as one of:
- "clear": Clear, relevant answer
- "unclear": Audio quality issue or mumbled response
- "off_topic": Answer is not related to question
- "human_request": Customer explicitly asks for human agent
- "anger": Customer is angry or frustrated
- "language_issue": Customer is struggling with current language

Respond with JSON only:
{{
    "response_type": "clear/unclear/off_topic/human_request/anger/language_issue",
    "confidence": 0.0-1.0,
    "answer_quality": 0.0-1.0,
    "reason": "brief explanation"
}}"""

    response = LOCAL_LLM.invoke([
        SystemMessage(content="You are a response analyzer. Respond with valid JSON only."),
        HumanMessage(content=prompt)
    ])
    
    try:
        result = json.loads(response.content)
        response_type = result.get("response_type", "clear")
        
        state["last_response_type"] = ResponseType(response_type)
        
        # Store the answer with quality score
        state["questions_asked"].append({
            "question": state["current_question"],
            "answer": last_response,
            "category": state.get("current_category", "unknown"),
            "quality": result.get("answer_quality", 0.5),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Response type: {response_type}")
        
    except (json.JSONDecodeError, ValueError):
        state["last_response_type"] = ResponseType.UNCLEAR
    
    state["transcript"].append({
        "timestamp": datetime.now().isoformat(),
        "speaker": "customer",
        "text": last_response,
        "response_type": state["last_response_type"]
    })
    
    return state


def repeat_question_node(state: AgentState) -> AgentState:
    """Ask customer to repeat (unclear audio)"""
    logger.info("=== REPEAT REQUEST ===")
    
    state["retry_count"] += 1
    
    repeat_msg = "I'm sorry, I couldn't hear you clearly. Could you please repeat that?"
    
    if state["language"] != "english":
        repeat_msg = translate_text(repeat_msg, "english", state["language"])
    
    text_to_speech(repeat_msg, state["language"])
    state["messages"].append(AIMessage(content=repeat_msg))
    
    return state


def redirect_to_question_node(state: AgentState) -> AgentState:
    """Redirect customer back to the question (off-topic)"""
    logger.info("=== REDIRECTING ===")
    
    redirect_msg = f"I understand. However, I need to ask: {state['current_question']}"
    
    if state["language"] != "english":
        redirect_msg = translate_text(redirect_msg, "english", state["language"])
    
    text_to_speech(redirect_msg, state["language"])
    state["messages"].append(AIMessage(content=redirect_msg))
    
    return state


def switch_language_node(state: AgentState) -> AgentState:
    """Switch conversation language"""
    logger.info("=== SWITCHING LANGUAGE ===")
    
    # Simple language switch logic
    new_lang = "hindi" if state["language"] == "english" else "english"
    
    switch_msg = f"I can speak in {'Hindi' if new_lang == 'hindi' else 'English'}. Let me repeat the question."
    
    text_to_speech(switch_msg, new_lang)
    state["messages"].append(AIMessage(content=switch_msg))
    state["language"] = new_lang
    
    logger.info(f"Language switched to: {new_lang}")
    return state


def human_handoff_node(state: AgentState) -> AgentState:
    """Handle handoff to human agent"""
    logger.info("=== HUMAN HANDOFF ===")
    
    state["conversation_stage"] = ConversationStage.HANDOFF
    
    if state["last_response_type"] == ResponseType.ANGER:
        state["handoff_reason"] = "customer_anger"
        handoff_msg = "I understand your frustration. Let me connect you with one of our representatives who can better assist you."
    else:
        state["handoff_reason"] = "customer_request"
        handoff_msg = "I'll connect you with one of our representatives. Please hold for a moment."
    
    if state["language"] != "english":
        handoff_msg = translate_text(handoff_msg, "english", state["language"])
    
    text_to_speech(handoff_msg, state["language"])
    state["messages"].append(AIMessage(content=handoff_msg))
    
    logger.info(f"Handoff reason: {state['handoff_reason']}")
    return state


def credit_scoring_node(state: AgentState) -> AgentState:
    """Calculate credit score based on conversation"""
    logger.info("=== CREDIT SCORING ===")
    
    state["conversation_stage"] = ConversationStage.SCORING
    
    # Use LLM to analyze all answers and generate credit score
    qa_summary = "\n".join([
        f"Q: {qa['question']}\nA: {qa['answer']} (Quality: {qa['quality']:.2f})"
        for qa in state["questions_asked"]
    ])
    
    prompt = f"""Analyze this credit assessment conversation and generate a credit score.

Customer Data: {json.dumps(state['customer_data'])}

Questions and Answers:
{qa_summary}

Generate a comprehensive credit score (0-100) based on:
1. Financial stability indicators
2. Repayment capacity
3. Answer quality and honesty
4. Risk factors

Respond with JSON only:
{{
    "credit_score": 0-100,
    "risk_level": "low/medium/high",
    "financial_stability": 0-100,
    "repayment_capacity": 0-100,
    "honesty_score": 0-100,
    "recommendation": "approve/review/reject",
    "key_strengths": ["strength1", "strength2"],
    "key_concerns": ["concern1", "concern2"],
    "suggested_loan_amount": number,
    "reasoning": "detailed explanation"
}}"""

    response = LOCAL_LLM.invoke([
        SystemMessage(content="You are a credit scoring analyst. Respond with valid JSON only."),
        HumanMessage(content=prompt)
    ])
    
    try:
        credit_score = json.loads(response.content)
        state["credit_score"] = credit_score
        logger.info(f"✓ Credit score calculated: {credit_score['credit_score']}/100")
    except json.JSONDecodeError:
        state["credit_score"] = {
            "credit_score": 50,
            "risk_level": "medium",
            "recommendation": "manual_review"
        }
    
    return state


def generate_report_node(state: AgentState) -> AgentState:
    """Generate final JSON report"""
    logger.info("=== GENERATING REPORT ===")
    
    report = {
        "call_metadata": {
            "call_sid": state["call_sid"],
            "customer_id": state["customer_data"]["customer_id"],
            "customer_name": state["customer_data"]["name"],
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": len(state["transcript"]) * 30,  # Rough estimate
            "language": state["language"]
        },
        "consent": {
            "given": state["consent_given"],
            "attempts": state["consent_attempts"]
        },
        "identity_verification": {
            "verified": state["identity_verified"],
            "score": state["identity_score"],
            "attempts": state["identity_attempts"]
        },
        "conversation": {
            "total_questions": len(state["questions_asked"]),
            "questions_answered": sum(1 for q in state["questions_asked"] if q["quality"] > 0.5),
            "average_answer_quality": sum(q["quality"] for q in state["questions_asked"]) / max(len(state["questions_asked"]), 1)
        },
        "credit_assessment": state["credit_score"],
        "transcript": state["transcript"],
        "audio_files": state["audio_chunks"],
        "handoff": {
            "occurred": state["conversation_stage"] == ConversationStage.HANDOFF,
            "reason": state.get("handoff_reason")
        }
    }
    
    state["final_report"] = report
    
    # Store everything
    store_transcript(state["call_sid"], state["transcript"])
    
    logger.info("✓ Report generated successfully")
    return state


def save_audio_node(state: AgentState) -> AgentState:
    """Save all audio recordings"""
    logger.info("=== SAVING AUDIO ===")
    
    for audio_path in state["audio_chunks"]:
        store_audio(state["call_sid"], audio_path)
    
    logger.info(f"✓ Saved {len(state['audio_chunks'])} audio files")
    return state


def end_call_node(state: AgentState) -> AgentState:
    """End call gracefully"""
    logger.info("=== ENDING CALL ===")
    
    state["conversation_stage"] = ConversationStage.CLOSING
    
    closing_msg = "Thank you for your time. This concludes our call. Have a great day!"
    
    if state["language"] != "english":
        closing_msg = translate_text(closing_msg, "english", state["language"])
    
    text_to_speech(closing_msg, state["language"])
    state["messages"].append(AIMessage(content=closing_msg))
    
    logger.info("✓ Call ended")
    return state


# ============= UTILITY FUNCTIONS =============

def translate_text(text: str, source: str, target: str) -> str:
    """Translate text between languages"""
    # Mock implementation - use actual translation service
    return text


def listen_for_response(state: AgentState) -> AgentState:
    """Listen and transcribe customer response"""
    # Simulate audio capture and STT
    audio_path = f"audio/customer_{datetime.now().timestamp()}.wav"
    stt_result = speech_to_text(audio_path, state["language"])
    
    state["messages"].append(HumanMessage(content=stt_result["text"]))
    state["audio_chunks"].append(audio_path)
    state["audio_quality"] = stt_result["audio_quality"]
    
    return state


# ============= ROUTING FUNCTIONS =============

def route_after_init(state: AgentState) -> str:
    """Route after initialization"""
    return "consent"


def route_after_consent(state: AgentState) -> str:
    """Route after consent node"""
    listen_for_response(state)
    state = analyze_consent_response(state)
    
    if state["consent_given"]:
        return "identity"
    else:
        return "log_refuse"


def route_after_identity(state: AgentState) -> str:
    """Route after identity verification"""
    listen_for_response(state)
    state = calculate_identity_score(state)
    
    if state["identity_verified"]:
        return "generate_questions"
    elif state["identity_attempts"] < 2:
        return "retry_identity"
    else:
        return "fail_identity"


def route_after_qa(state: AgentState) -> str:
    """Route after Q&A node"""
    listen_for_response(state)
    state = analyze_response_node(state)
    
    response_type = state["last_response_type"]
    
    if response_type == ResponseType.UNCLEAR:
        if state["retry_count"] < 2:
            return "repeat"
        else:
            state["retry_count"] = 0
            return "next_question"
    elif response_type == ResponseType.OFF_TOPIC:
        return "redirect"
    elif response_type == ResponseType.HUMAN_REQUEST:
        return "handoff"
    elif response_type == ResponseType.ANGER:
        return "handoff"
    elif response_type == ResponseType.LANGUAGE_ISSUE:
        return "switch_language"
    else:  # CLEAR
        state["retry_count"] = 0
        return "next_question"


def check_more_questions(state: AgentState) -> str:
    """Check if more questions remain"""
    remaining = [q for q in state["questions_remaining"] if not q["asked"]]
    
    if len(remaining) > 0:
        return "continue_qa"
    else:
        return "scoring"


# ============= GRAPH CONSTRUCTION =============

def create_telecalling_graph():
    """Create the complete LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("init", initialize_session_node)
    workflow.add_node("consent", consent_node)
    workflow.add_node("log_refuse", end_call_node)
    workflow.add_node("identity", identity_verification_node)
    workflow.add_node("retry_identity", identity_verification_node)
    workflow.add_node("fail_identity", end_call_node)
    workflow.add_node("generate_questions", generate_questions_from_cam)
    workflow.add_node("qa", dynamic_qa_node)
    workflow.add_node("repeat", repeat_question_node)
    workflow.add_node("redirect", redirect_to_question_node)
    workflow.add_node("switch_language", switch_language_node)
    workflow.add_node("handoff", human_handoff_node)
    workflow.add_node("scoring", credit_scoring_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("save_audio", save_audio_node)
    workflow.add_node("end_call", end_call_node)
    
    # Set entry point
    workflow.add_edge(START, "init")
    
    # Add edges following the flowchart
    workflow.add_conditional_edges("init", route_after_init)
    workflow.add_conditional_edges("consent", route_after_consent)
    workflow.add_edge("log_refuse", END)
    workflow.add_conditional_edges("identity", route_after_identity)
    workflow.add_edge("retry_identity", "identity")
    workflow.add_edge("fail_identity", END)
    workflow.add_edge("generate_questions", "qa")
    workflow.add_conditional_edges("qa", route_after_qa)
    workflow.add_edge("repeat", "qa")
    workflow.add_edge("redirect", "qa")
    workflow.add_edge("switch_language", "qa")
    workflow.add_conditional_edges(
        "qa",
        check_more_questions,
        {
            "continue_qa": "qa",
            "scoring": "scoring"
        }
    )
    workflow.add_edge("handoff", "end_call")
    workflow.add_edge("scoring", "generate_report")
    workflow.add_edge("generate_report", "save_audio")
    workflow.add_edge("save_audio", "end_call")
    workflow.add_edge("end_call", END)
    
    return workflow.compile()


# ============= MAIN EXECUTION =============

def run_telecalling_agent(customer_id: str, phone_number: str):
    """Main function to run telecalling agent"""
    
    logger.info("=" * 60)
    logger.info("STARTING TELECALLING AGENT")
    logger.info("=" * 60)
    
    # Fetch customer data
    customer_data = get_customer_data(customer_id)
    
    # Initiate call
    call_result = twilio_make_call(
        phone_number=phone_number,
        callback_url="https://your-server.in/twilio/callback"
    )
    
    # Initialize state
    initial_state: AgentState = {
        "messages": [],
        "customer_data": customer_data,
        "call_sid": call_result["call_sid"],
        "language": customer_data.get("language_preference", "english"),
        "transcript": [],
        "audio_chunks": [],
        "conversation_stage": ConversationStage.INIT,
        "consent_given": False,
        "consent_attempts": 0,
        "identity_score": 0.0,
        "identity_attempts": 0,
        "identity_verified": False,
        "current_question": "",
        "current_category": "",
        "questions_asked": [],
        "questions_remaining": [],
        "retry_count": 0,
        "last_response_type": ResponseType.CLEAR,  # Default initial value
        "audio_quality": "unknown",
        "needs_clarification": False,
        "credit_score": {},
        "final_report": {},
        "error_type": None,
        "handoff_reason": None,
    }

    # Create and compile the graph
    app = create_telecalling_graph()

    # Stream events and print updates
    # Note: In a real-world scenario, the graph would be paused at each 'listen' step,
    # waiting for an external event (like a Twilio webhook with user's speech).
    # Here, 'listen_for_response' is mocked, so the graph runs to completion.
    final_state = None
    logger.info("--- Starting Graph Execution ---")
    for event in app.stream(initial_state):
        node_name = list(event.keys())[0]
        logger.info(f"--- Executing Node: {node_name} ---")
        if "__end__" in event:
            final_state = event["__end__"]
            break
        
    if final_state:
        logger.info("\n" + "=" * 60)
        logger.info("AGENT EXECUTION FINISHED")
        logger.info("=" * 60)
        
        # Print final report
        final_report = final_state.get("final_report", {})
        if final_report:
            logger.info("Final Report:")
            print(json.dumps(final_report, indent=2))
        else:
            logger.warning("No final report was generated.")
    else:
        logger.error("Agent execution failed to complete.")


if __name__ == "__main__":
    # Example customer ID and phone number
    CUSTOMER_ID = "CUST12345"
    PHONE_NUMBER = "+919999988888"
    
    # Run the agent for the specified customer
    run_telecalling_agent(
        customer_id=CUSTOMER_ID,
        phone_number=PHONE_NUMBER
    )