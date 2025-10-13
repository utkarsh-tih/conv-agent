"""
Text-Based Telecalling AI Agent
Simplified version for testing core logic without audio/Twilio
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import json
from datetime import datetime
import logging
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
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
    ],
    "repayment_history": [
        "Have you ever defaulted on a loan payment?",
        "How would you rate your payment discipline?",
    ],
    "assets": [
        "Do you own any property or vehicle?",
        "What is your primary source of income?",
    ],
    "personal_info": [
        "Can you confirm your date of birth?",
        "What is your current residential address?",
    ]
}


# ============= STATE DEFINITION =============

class AgentState(TypedDict):
    """Complete state for telecalling agent"""
    messages: list[BaseMessage]
    customer_data: dict
    call_sid: str
    language: str
    
    transcript: list[dict]
    conversation_stage: ConversationStage
    
    consent_given: bool
    consent_attempts: int
    
    identity_score: float
    identity_attempts: int
    identity_verified: bool
    
    current_question: str
    current_category: str
    questions_asked: list[dict]
    questions_remaining: list[dict]
    retry_count: int
    
    last_response_type: ResponseType
    needs_clarification: bool
    
    credit_score: dict
    final_report: dict
    
    error_type: str | None
    handoff_reason: str | None
    
    # For interactive mode
    waiting_for_input: bool


# ============= LLM INITIALIZATION =============

def initialize_llm(model: str = "llama3.2"):
    """Initialize local LLM"""
    return ChatOllama(
        model=model,
        temperature=0.3,
        base_url="http://localhost:11434"
    )

try:
    LOCAL_LLM = initialize_llm()
    logger.info("✓ LLM initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize LLM: {e}")
    LOCAL_LLM = None


# ============= MOCK DATA =============

MOCK_CUSTOMERS = {
    "CUST001": {
        "customer_id": "CUST001",
        "name": "Rajesh Kumar",
        "phone": "+919876543210",
        "language_preference": "english",
        "date_of_birth": "1985-06-15",
        "address": "123 MG Road, Mumbai, Maharashtra",
        "previous_interactions": 2,
        "risk_category": "medium"
    }
}


# ============= HELPER FUNCTIONS =============

def get_customer_data(customer_id: str) -> dict:
    """Fetch customer data"""
    return MOCK_CUSTOMERS.get(customer_id, MOCK_CUSTOMERS["CUST001"])


def print_agent_message(message: str, stage: str = ""):
    """Pretty print agent messages"""
    print("\n" + "=" * 70)
    if stage:
        print(f"[{stage}]")
    print(f"🤖 AGENT: {message}")
    print("=" * 70)


def print_stage_header(stage: str):
    """Print stage header"""
    print("\n" + "▓" * 70)
    print(f"   STAGE: {stage}")
    print("▓" * 70)


def get_user_input() -> str:
    """Get user input"""
    response = input("\n👤 YOU: ").strip()
    return response


def add_to_transcript(state: AgentState, speaker: str, text: str):
    """Add entry to transcript"""
    state["transcript"].append({
        "timestamp": datetime.now().isoformat(),
        "speaker": speaker,
        "text": text,
        "stage": state["conversation_stage"]
    })


# ============= NODE FUNCTIONS =============

def initialize_session_node(state: AgentState) -> AgentState:
    """Initialize call session"""
    print_stage_header("INITIALIZATION")
    
    state["conversation_stage"] = ConversationStage.INIT
    state["consent_given"] = False
    state["consent_attempts"] = 0
    state["identity_verified"] = False
    state["identity_attempts"] = 0
    state["identity_score"] = 0.0
    state["questions_asked"] = []
    state["questions_remaining"] = []
    state["retry_count"] = 0
    state["transcript"] = []
    state["waiting_for_input"] = False
    
    greeting = f"Hello! This is an automated credit assessment system. Am I speaking with {state['customer_data']['name']}?"
    
    state["messages"].append(AIMessage(content=greeting))
    add_to_transcript(state, "agent", greeting)
    print_agent_message(greeting, "INIT")
    
    # Get user response
    user_response = get_user_input()
    state["messages"].append(HumanMessage(content=user_response))
    add_to_transcript(state, "customer", user_response)
    
    return state


def consent_node(state: AgentState) -> AgentState:
    """Handle consent collection"""
    print_stage_header("CONSENT")
    
    state["conversation_stage"] = ConversationStage.CONSENT
    
    consent_msg = """I'm calling regarding your loan application. This conversation will be recorded for quality and training purposes. 
The conversation will take approximately 5-7 minutes. Do you consent to proceed with this call?"""
    
    state["messages"].append(AIMessage(content=consent_msg))
    state["current_question"] = consent_msg
    add_to_transcript(state, "agent", consent_msg)
    print_agent_message(consent_msg, "CONSENT")
    
    # Get user response
    user_response = get_user_input()
    state["messages"].append(HumanMessage(content=user_response))
    add_to_transcript(state, "customer", user_response)
    state["consent_attempts"] += 1
    
    return state


def analyze_consent_response(state: AgentState) -> AgentState:
    """Analyze customer's consent response using LLM"""
    
    last_response = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_response = msg.content
            break
    
    if not last_response or not LOCAL_LLM:
        state["consent_given"] = False
        return state
    
    prompt = f"""Analyze if the customer gave consent to proceed with the call.

Customer's response: "{last_response}"

Respond with JSON only:
{{"consent_given": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    try:
        response = LOCAL_LLM.invoke([
            SystemMessage(content="You are a consent analyzer. Respond with valid JSON only."),
            HumanMessage(content=prompt)
        ])
        
        result = json.loads(response.content)
        if result["consent_given"] and result["confidence"] > 0.7:
            state["consent_given"] = True
            logger.info(f"✓ Consent obtained (confidence: {result['confidence']:.2f})")
        else:
            state["consent_given"] = False
            logger.info(f"✗ Consent not given - {result['reason']}")
    except Exception as e:
        logger.error(f"Error analyzing consent: {e}")
        state["consent_given"] = False
    
    return state


def identity_verification_node(state: AgentState) -> AgentState:
    """Verify customer identity"""
    print_stage_header("IDENTITY VERIFICATION")
    
    state["conversation_stage"] = ConversationStage.IDENTITY
    
    questions = [
        "Can you confirm your date of birth? (Format: YYYY-MM-DD)",
        "What is your current residential address?",
        "Can you confirm your registered mobile number?",
    ]
    
    question = questions[min(state["identity_attempts"], len(questions)-1)]
    
    state["messages"].append(AIMessage(content=question))
    state["current_question"] = question
    add_to_transcript(state, "agent", question)
    print_agent_message(question, "IDENTITY")
    
    # Get user response
    user_response = get_user_input()
    state["messages"].append(HumanMessage(content=user_response))
    add_to_transcript(state, "customer", user_response)
    
    return state


def calculate_identity_score(state: AgentState) -> AgentState:
    """Calculate identity verification score using LLM"""
    
    last_response = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_response = msg.content
            break
    
    if not last_response or not LOCAL_LLM:
        return state
    
    prompt = f"""Verify if the customer's response matches the expected identity information.

Expected Information:
- Date of Birth: {state['customer_data']['date_of_birth']}
- Address: {state['customer_data']['address']}
- Phone: {state['customer_data']['phone']}

Customer's response: "{last_response}"
Question asked: "{state['current_question']}"

Calculate identity score (0.0-1.0) based on how well the response matches.
Respond with JSON only:
{{"identity_score": 0.0-1.0, "matches": true/false, "reason": "explanation"}}"""

    try:
        response = LOCAL_LLM.invoke([
            SystemMessage(content="You are an identity verification system. Respond with valid JSON only."),
            HumanMessage(content=prompt)
        ])
        
        result = json.loads(response.content)
        score = result.get("identity_score", 0.0)
        
        if state["identity_score"] > 0:
            state["identity_score"] = (state["identity_score"] + score) / 2
        else:
            state["identity_score"] = score
        
        state["identity_attempts"] += 1
        
        if state["identity_score"] >= 0.75:
            state["identity_verified"] = True
            logger.info(f"✓ Identity verified (score: {state['identity_score']:.2f})")
        else:
            logger.info(f"Identity score: {state['identity_score']:.2f} (attempts: {state['identity_attempts']})")
            
    except Exception as e:
        logger.error(f"Error calculating identity score: {e}")
        state["identity_score"] = 0.0
    
    return state


def generate_questions_from_cam(state: AgentState) -> AgentState:
    """Generate questions from CAM (Credit Assessment Model)"""
    print_stage_header("CREDIT ASSESSMENT")
    
    questions = []
    for category, q_list in CAM_QUESTIONS.items():
        for q in q_list:
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
    
    next_question = None
    for q in state["questions_remaining"]:
        if not q["asked"]:
            next_question = q
            break
    
    if not next_question:
        state["conversation_stage"] = ConversationStage.SCORING
        return state
    
    question_text = next_question["question"]
    
    state["messages"].append(AIMessage(content=question_text))
    state["current_question"] = question_text
    state["current_category"] = next_question["category"]
    
    add_to_transcript(state, "agent", question_text)
    print_agent_message(f"[{next_question['category'].upper()}] {question_text}", "QUESTION")
    
    next_question["asked"] = True
    
    # Get user response
    user_response = get_user_input()
    state["messages"].append(HumanMessage(content=user_response))
    add_to_transcript(state, "customer", user_response)
    
    return state


def analyze_response_node(state: AgentState) -> AgentState:
    """Analyze customer's response"""
    
    last_response = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_response = msg.content
            break
    
    if not last_response or not LOCAL_LLM:
        state["last_response_type"] = ResponseType.CLEAR
        return state
    
    prompt = f"""Analyze the customer's response in a telecalling conversation.

Question asked: "{state['current_question']}"
Customer's response: "{last_response}"

Classify the response as one of:
- "clear": Clear, relevant answer
- "unclear": Vague or incomplete response
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

    try:
        response = LOCAL_LLM.invoke([
            SystemMessage(content="You are a response analyzer. Respond with valid JSON only."),
            HumanMessage(content=prompt)
        ])
        
        result = json.loads(response.content)
        response_type = result.get("response_type", "clear")
        
        state["last_response_type"] = ResponseType(response_type)
        
        state["questions_asked"].append({
            "question": state["current_question"],
            "answer": last_response,
            "category": state.get("current_category", "unknown"),
            "quality": result.get("answer_quality", 0.5),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Response type: {response_type} (quality: {result.get('answer_quality', 0.5):.2f})")
        
    except Exception as e:
        logger.error(f"Error analyzing response: {e}")
        state["last_response_type"] = ResponseType.CLEAR
    
    return state


def repeat_question_node(state: AgentState) -> AgentState:
    """Ask customer to clarify"""
    state["retry_count"] += 1
    
    repeat_msg = "I'm sorry, I didn't quite understand that. Could you please clarify?"
    
    state["messages"].append(AIMessage(content=repeat_msg))
    add_to_transcript(state, "agent", repeat_msg)
    print_agent_message(repeat_msg, "CLARIFICATION")
    
    # Get user response
    user_response = get_user_input()
    state["messages"].append(HumanMessage(content=user_response))
    add_to_transcript(state, "customer", user_response)
    
    return state


def redirect_to_question_node(state: AgentState) -> AgentState:
    """Redirect customer back to the question"""
    
    redirect_msg = f"I understand. However, I need to ask: {state['current_question']}"
    
    state["messages"].append(AIMessage(content=redirect_msg))
    add_to_transcript(state, "agent", redirect_msg)
    print_agent_message(redirect_msg, "REDIRECT")
    
    # Get user response
    user_response = get_user_input()
    state["messages"].append(HumanMessage(content=user_response))
    add_to_transcript(state, "customer", user_response)
    
    return state


def human_handoff_node(state: AgentState) -> AgentState:
    """Handle handoff to human agent"""
    print_stage_header("HUMAN HANDOFF")
    
    state["conversation_stage"] = ConversationStage.HANDOFF
    
    if state["last_response_type"] == ResponseType.ANGER:
        state["handoff_reason"] = "customer_anger"
        handoff_msg = "I understand your frustration. Let me connect you with one of our representatives who can better assist you."
    else:
        state["handoff_reason"] = "customer_request"
        handoff_msg = "I'll connect you with one of our representatives. Please hold for a moment."
    
    state["messages"].append(AIMessage(content=handoff_msg))
    add_to_transcript(state, "agent", handoff_msg)
    print_agent_message(handoff_msg, "HANDOFF")
    
    logger.info(f"Handoff reason: {state['handoff_reason']}")
    return state


def credit_scoring_node(state: AgentState) -> AgentState:
    """Calculate credit score based on conversation"""
    print_stage_header("CREDIT SCORING")
    
    state["conversation_stage"] = ConversationStage.SCORING
    
    if not LOCAL_LLM:
        state["credit_score"] = {"credit_score": 50, "risk_level": "medium"}
        return state
    
    qa_summary = "\n".join([
        f"Category: {qa['category']}\nQ: {qa['question']}\nA: {qa['answer']} (Quality: {qa['quality']:.2f})"
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

    try:
        response = LOCAL_LLM.invoke([
            SystemMessage(content="You are a credit scoring analyst. Respond with valid JSON only."),
            HumanMessage(content=prompt)
        ])
        
        credit_score = json.loads(response.content)
        state["credit_score"] = credit_score
        logger.info(f"✓ Credit score calculated: {credit_score['credit_score']}/100")
        
        # Show score to user
        print("\n" + "🎯" * 35)
        print(f"   CREDIT SCORE: {credit_score['credit_score']}/100")
        print(f"   RISK LEVEL: {credit_score['risk_level'].upper()}")
        print(f"   RECOMMENDATION: {credit_score['recommendation'].upper()}")
        print("🎯" * 35)
        
    except Exception as e:
        logger.error(f"Error calculating credit score: {e}")
        state["credit_score"] = {
            "credit_score": 50,
            "risk_level": "medium",
            "recommendation": "manual_review"
        }
    
    return state


def generate_report_node(state: AgentState) -> AgentState:
    """Generate final JSON report"""
    print_stage_header("GENERATING REPORT")
    
    report = {
        "call_metadata": {
            "call_sid": state["call_sid"],
            "customer_id": state["customer_data"]["customer_id"],
            "customer_name": state["customer_data"]["name"],
            "timestamp": datetime.now().isoformat(),
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
        "handoff": {
            "occurred": state["conversation_stage"] == ConversationStage.HANDOFF,
            "reason": state.get("handoff_reason")
        }
    }
    
    state["final_report"] = report
    logger.info("✓ Report generated successfully")
    
    return state


def end_call_node(state: AgentState) -> AgentState:
    """End call gracefully"""
    print_stage_header("CALL ENDING")
    
    state["conversation_stage"] = ConversationStage.CLOSING
    
    closing_msg = "Thank you for your time. This concludes our conversation. Have a great day!"
    
    state["messages"].append(AIMessage(content=closing_msg))
    add_to_transcript(state, "agent", closing_msg)
    print_agent_message(closing_msg, "CLOSING")
    
    logger.info("✓ Call ended")
    return state


# ============= ROUTING FUNCTIONS =============

def route_after_consent(state: AgentState) -> str:
    """Route after consent node"""
    state = analyze_consent_response(state)
    
    if state["consent_given"]:
        return "identity"
    else:
        return "log_refuse"


def route_after_identity(state: AgentState) -> str:
    """Route after identity verification"""
    state = calculate_identity_score(state)
    
    if state["identity_verified"]:
        return "generate_questions"
    elif state["identity_attempts"] < 3:
        return "identity"
    else:
        return "fail_identity"


def route_after_qa(state: AgentState) -> str:
    """Route after Q&A node"""
    state = analyze_response_node(state)
    
    response_type = state["last_response_type"]
    
    if response_type == ResponseType.UNCLEAR:
        if state["retry_count"] < 2:
            return "repeat"
        else:
            state["retry_count"] = 0
            return "check_more"
    elif response_type == ResponseType.OFF_TOPIC:
        return "redirect"
    elif response_type == ResponseType.HUMAN_REQUEST:
        return "handoff"
    elif response_type == ResponseType.ANGER:
        return "handoff"
    else:  # CLEAR
        state["retry_count"] = 0
        return "check_more"


def check_more_questions(state: AgentState) -> str:
    """Check if more questions remain"""
    remaining = [q for q in state["questions_remaining"] if not q["asked"]]
    
    if len(remaining) > 0:
        return "qa"
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
    workflow.add_node("fail_identity", end_call_node)
    workflow.add_node("generate_questions", generate_questions_from_cam)
    workflow.add_node("qa", dynamic_qa_node)
    workflow.add_node("repeat", repeat_question_node)
    workflow.add_node("redirect", redirect_to_question_node)
    workflow.add_node("check_more", check_more_questions)
    workflow.add_node("handoff", human_handoff_node)
    workflow.add_node("scoring", credit_scoring_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("end_call", end_call_node)
    
    # Set entry point
    workflow.add_edge(START, "init")
    
    # Add edges
    workflow.add_edge("init", "consent")
    workflow.add_conditional_edges("consent", route_after_consent)
    workflow.add_edge("log_refuse", END)
    workflow.add_conditional_edges("identity", route_after_identity)
    workflow.add_edge("fail_identity", END)
    workflow.add_edge("generate_questions", "qa")
    workflow.add_conditional_edges("qa", route_after_qa, {
        "repeat": "repeat",
        "redirect": "redirect",
        "handoff": "handoff",
        "check_more": "check_more"
    })
    workflow.add_edge("repeat", "qa")
    workflow.add_edge("redirect", "qa")
    workflow.add_conditional_edges("check_more", check_more_questions)
    workflow.add_edge("handoff", "generate_report")
    workflow.add_edge("scoring", "generate_report")
    workflow.add_edge("generate_report", "end_call")
    workflow.add_edge("end_call", END)
    
    return workflow.compile()


# ============= MAIN EXECUTION =============

def run_text_telecalling_agent(customer_id: str = "CUST001"):
    """Main function to run text-based telecalling agent"""
    
    print("\n" + "█" * 70)
    print("   TEXT-BASED TELECALLING AI AGENT")
    print("   Credit Assessment System")
    print("█" * 70)
    
    if not LOCAL_LLM:
        print("\n⚠️  WARNING: Could not connect to Ollama LLM")
        print("Please ensure Ollama is running: ollama serve")
        print("And the model is installed: ollama pull llama3.1:8b\n")
        return
    
    # Fetch customer data
    customer_data = get_customer_data(customer_id)
    
    print(f"\n📋 Customer: {customer_data['name']}")
    print(f"📞 Phone: {customer_data['phone']}")
    print(f"🔤 Language: {customer_data['language_preference']}\n")
    
    # Initialize state
    initial_state: AgentState = {
        "messages": [],
        "customer_data": customer_data,
        "call_sid": f"TEXT_{datetime.now().timestamp()}",
        "language": customer_data.get("language_preference", "english"),
        "transcript": [],
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
        "last_response_type": ResponseType.CLEAR,
        "needs_clarification": False,
        "credit_score": {},
        "final_report": {},
        "error_type": None,
        "handoff_reason": None,
        "waiting_for_input": False
    }
    
    # Create and run graph
    try:
        graph = create_telecalling_graph()
        final_state = graph.invoke(initial_state)
        
        # Print final report
        print("\n\n" + "=" * 70)
        print("📊 FINAL REPORT")
        print("=" * 70)
        print(json.dumps(final_state["final_report"], indent=2))
        
        # Save report to file
        report_file = f"report_{final_state['call_sid']}.json"
        with open(report_file, 'w') as f:
            json.dump(final_state["final_report"], f, indent=2)
        print(f"\n✓ Report saved to: {report_file}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Starting Text-Based Telecalling Agent...")
    print("Make sure Ollama is running with llama3.2 model\n")
    
    run_text_telecalling_agent("CUST001")