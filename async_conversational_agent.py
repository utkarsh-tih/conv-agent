"""
Low-latency conversational async telephonic agent with streaming and optimistic acknowledgments.
Uses interrupt() for production-ready human-in-the-loop.
"""

import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, List
from pydantic import BaseModel, Field
from langgraph.types import Command, interrupt
from langgraph.types import RetryPolicy
import os
import random

# --- Constants ---
MAX_RETRIES = 2
SYSTEM_RETRIES = 3

# --- LLM Initialization ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEBlfBBLJhRZ47mGtmwSwXmnFNnJmVzPM"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# --- Conversational Prompts ---
class QuickAcknowledgments:
    """Immediate responses - no LLM needed, fire instantly."""
    
    MINIMAL = ["Mm-hmm.", "Got it.", "Okay.", "Right.", "Alright.", "Sure."]
    POSITIVE = ["Perfect.", "Great.", "Excellent.", "Wonderful.", "Fantastic."]
    

class ConversationalPrompts:
    """Natural prompts for voice calls."""
    
    CONSENT_OPENERS = [
        "Hi! Before we get started, I just need to let you know that this call will be recorded for quality and compliance. Is that okay with you?",
        "Hello! Quick heads up - we'll be recording this conversation for our records. Are you comfortable with that?",
    ]
    
    CONSENT_RETRY = [
        "Sorry, I didn't quite catch that. Just to confirm - are you okay with us recording this call?",
        "Let me ask again - do I have your permission to record this call?",
    ]
    
    CONSENT_GRANTED = [
        "Perfect! Let's get you verified.",
        "Great! Alright, for security purposes...",
    ]
    
    CONSENT_DENIED = [
        "No problem at all. Have a great day!",
        "I understand. Thanks for your time!",
    ]
    
    AUTH_OPENERS = [
        "I need your date of birth and the last 4 digits of your Aadhaar card.",
        "Could you share your date of birth and Aadhaar's last 4 digits?",
    ]
    
    AUTH_RETRY = [
        "I didn't catch all of that. Date of birth and last 4 digits of Aadhaar again?",
        "Sorry, could you repeat your date of birth and Aadhaar's last 4 digits?",
    ]
    
    AUTH_SUCCESS = [
        "Perfect! Now I have a few quick questions.",
        "Great! Let me ask you some verification questions.",
    ]
    
    AUTH_FAILED = [
        "I'm having trouble verifying those details. Let's try this another time.",
        "I couldn't verify that information. Please call back when you have your details ready.",
    ]
    
    TRANSITIONS = ["Next -", "Alright -", "Okay -", ""]
    
    COMPLETION = [
        "That's everything! Thanks for your time.",
        "Perfect! We're all done. Have a great day!",
    ]


# --- Pydantic Models ---
class ConsentResult(BaseModel):
    consent_given: bool
    confidence: int = Field(ge=0, le=100)
    reasoning: str
    should_retry: bool


class AuthenticationResult(BaseModel):
    date_of_birth: str
    aadhaar_last4: str
    confidence_date_of_birth: int = Field(ge=0, le=100)
    confidence_aadhaar: int = Field(ge=0, le=100)
    reasoning_date_of_birth: str
    reasoning_aadhaar: str
    should_retry: bool


class QuestionAnalysis(BaseModel):
    """Analysis of answer completeness."""
    answer_extracted: str
    is_complete: bool
    confidence: int = Field(ge=0, le=100)
    missing_parts: List[str] = Field(default_factory=list)
    follow_up_needed: bool
    reasoning: str


class FollowUpQuestion(BaseModel):
    """Dynamic follow-up question."""
    question_text: str
    additional_context_questions: List[str] = Field(default_factory=list, max_items=3)


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_stage: str
    
    # Consent
    consent_given: bool
    consent_retry_count: int
    
    # Auth
    authenticated: bool
    auth_attempts: int
    user_dob: str
    user_aadhar_digits: str
    
    # Questions
    current_question_index: int
    question_retry_count: int
    questions_answers: dict
    pending_follow_up: bool
    follow_up_context: dict


# --- Question Definitions ---
VERIFICATION_QUESTIONS = [
    {
        "id": "q1_address",
        "question": "What's your current residential address?",
        "required_parts": ["street/house", "area/locality", "city", "pincode"],
        "context_questions": [
            "How long have you been living there?",
            "Is this a rented or owned property?",
        ]
    },
    {
        "id": "q2_income",
        "question": "Could you tell me about your monthly income?",
        "required_parts": ["amount", "source/employment"],
        "context_questions": [
            "What do you do for work?",
            "Is this your only source of income?",
        ]
    },
    {
        "id": "q3_purpose",
        "question": "What brings you here today? What's this verification for?",
        "required_parts": ["purpose/reason"],
        "context_questions": [
            "When do you need this completed by?",
            "Have you done this kind of verification before?",
        ]
    }
]


# ============================================================================
# CONSENT FLOW
# ============================================================================

def ask_consent_node(state: AgentState) -> dict:
    """Ask for consent."""
    print("\n==== Ask Consent ====")
    
    retry_count = state.get("consent_retry_count", 0)
    
    if retry_count > 0:
        message = random.choice(ConversationalPrompts.CONSENT_RETRY)
    else:
        message = random.choice(ConversationalPrompts.CONSENT_OPENERS)
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": "asking_consent"
    }


async def process_consent_node(state: AgentState) -> Command:
    """Process consent with instant acknowledgment then validation."""
    print("\n==== Process Consent ====")
    
    # Get user input via interrupt
    user_input = interrupt("consent_input")
    
    # INSTANT acknowledgment
    ack = random.choice(QuickAcknowledgments.MINIMAL)
    print(f"[INSTANT] Agent: {ack}")
    
    # Now validate in background (from user's perspective)
    print(f"[VALIDATING] User said: {user_input}")
    
    system_prompt = """Analyze voice call response about recording consent.

Be lenient:
- "yes", "yeah", "sure", "okay", "yep", "go ahead", "fine" = consent given
- "no", "nope", "not comfortable" = consent denied
- Unclear = should_retry

Handle natural speech and filler words."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    
    llm_output = await llm.with_structured_output(ConsentResult).ainvoke(messages)
    print(f"[VALIDATED] Consent: {llm_output.consent_given}, Confidence: {llm_output.confidence}")
    
    retry_count = state.get("consent_retry_count", 0)
    should_retry = llm_output.should_retry and not llm_output.consent_given
    
    return Command(
        update={
            "messages": [HumanMessage(content=user_input), AIMessage(content=ack)],
            "consent_given": llm_output.consent_given,
            "consent_retry_count": retry_count + 1 if should_retry and retry_count < MAX_RETRIES else retry_count,
            "current_stage": "consent_processed"
        }
    )


def respond_to_consent_node(state: AgentState) -> dict:
    """Final consent response."""
    print("\n==== Respond to Consent ====")
    
    if state.get("consent_given"):
        message = random.choice(ConversationalPrompts.CONSENT_GRANTED)
    else:
        message = random.choice(ConversationalPrompts.CONSENT_DENIED)
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": "consent_responded"
    }


# ============================================================================
# AUTHENTICATION FLOW
# ============================================================================

def ask_authentication_node(state: AgentState) -> dict:
    """Ask for authentication."""
    print("\n==== Ask Authentication ====")
    
    attempts = state.get("auth_attempts", 0)
    
    if attempts > 0:
        message = random.choice(ConversationalPrompts.AUTH_RETRY)
    else:
        message = random.choice(ConversationalPrompts.AUTH_OPENERS)
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": "asking_authentication"
    }


async def process_authentication_node(state: AgentState) -> Command:
    """Process authentication with instant acknowledgment."""
    print("\n==== Process Authentication ====")
    
    user_input = interrupt("auth_input")
    
    # INSTANT acknowledgment
    ack = random.choice(QuickAcknowledgments.MINIMAL)
    print(f"[INSTANT] Agent: {ack}")
    
    # Validate
    print(f"[VALIDATING] User said: {user_input}")
    
    system_prompt = """Extract date of birth and Aadhaar last 4 digits from voice response.

Flexible date formats:
- "15th March 1990", "March 15, 1990", "15/3/1990" → 15/03/1990
- Handle spoken numbers

Be lenient with confidence if reasonably clear.
Set should_retry only if genuinely missing or ambiguous."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    
    llm_output = await llm.with_structured_output(AuthenticationResult).ainvoke(messages)
    
    authenticated = (
        llm_output.confidence_date_of_birth > 50 and 
        llm_output.confidence_aadhaar > 50 and
        not llm_output.should_retry
    )
    
    print(f"[VALIDATED] Authenticated: {authenticated}")
    
    attempts = state.get("auth_attempts", 0)
    
    return Command(
        update={
            "messages": [HumanMessage(content=user_input), AIMessage(content=ack)],
            "authenticated": authenticated,
            "user_dob": llm_output.date_of_birth,
            "user_aadhar_digits": llm_output.aadhaar_last4,
            "auth_attempts": attempts + 1 if not authenticated and attempts < MAX_RETRIES else attempts,
            "current_stage": "authentication_processed"
        }
    )


def respond_to_authentication_node(state: AgentState) -> dict:
    """Final auth response."""
    print("\n==== Respond to Authentication ====")
    
    if state.get("authenticated"):
        message = random.choice(ConversationalPrompts.AUTH_SUCCESS)
    else:
        message = random.choice(ConversationalPrompts.AUTH_FAILED)
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": "authentication_responded"
    }


# ============================================================================
# QUESTIONS FLOW
# ============================================================================

def ask_question_node(state: AgentState) -> dict:
    """Ask question or follow-up."""
    print("\n==== Ask Question ====")
    
    current_idx = state.get("current_question_index", 0)
    pending_follow_up = state.get("pending_follow_up", False)
    follow_up_context = state.get("follow_up_context", {})
    
    if pending_follow_up and follow_up_context:
        message = follow_up_context.get("follow_up_question", "")
        print(f"Asking follow-up: {message}")
    else:
        question_data = VERIFICATION_QUESTIONS[current_idx]
        intro = random.choice(ConversationalPrompts.TRANSITIONS) if current_idx > 0 else ""
        message = f"{intro}{question_data['question']}"
        print(f"Asking Q{current_idx + 1}: {message}")
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": f"asking_question_{current_idx}"
    }


async def process_question_node(state: AgentState) -> Command:
    """Process answer with instant acknowledgment and validation."""
    print("\n==== Process Question ====")
    
    current_idx = state.get("current_question_index", 0)
    pending_follow_up = state.get("pending_follow_up", False)
    follow_up_context = state.get("follow_up_context", {})
    
    user_input = interrupt("question_input")
    
    # INSTANT acknowledgment
    ack = random.choice(QuickAcknowledgments.MINIMAL)
    print(f"[INSTANT] Agent: {ack}")
    
    # Validate
    print(f"[VALIDATING] User said: {user_input}")
    
    question_data = VERIFICATION_QUESTIONS[current_idx]
    
    # Combine with previous partial answer if follow-up
    if pending_follow_up and follow_up_context:
        combined_answer = f"{follow_up_context.get('partial_answer', '')} {user_input}"
        print(f"[COMBINED] {combined_answer}")
    else:
        combined_answer = user_input
    
    # Analyze completeness
    analysis_prompt = f"""Analyze answer to: "{question_data['question']}"

User: "{combined_answer}"

Required: {', '.join(question_data['required_parts'])}

Be reasonably lenient. If most key info is there, mark complete.
If missing, specify exactly what for follow-up."""
    
    messages = [
        SystemMessage(content=analysis_prompt),
        HumanMessage(content=combined_answer),
    ]
    
    analysis = await llm.with_structured_output(QuestionAnalysis).ainvoke(messages)
    print(f"[ANALYSIS] Complete: {analysis.is_complete}, Confidence: {analysis.confidence}")
    
    is_acceptable = analysis.is_complete and analysis.confidence > 60
    
    answers = state.get("questions_answers", {}).copy()
    update_dict = {
        "messages": [HumanMessage(content=user_input), AIMessage(content=ack)]
    }
    
    if is_acceptable:
        # Answer complete
        answers[question_data["id"]] = {
            "question": question_data["question"],
            "answer": analysis.answer_extracted,
            "confidence": analysis.confidence
        }
        
        update_dict.update({
            "questions_answers": answers,
            "current_question_index": current_idx + 1,
            "question_retry_count": 0,
            "pending_follow_up": False,
            "follow_up_context": {},
            "current_stage": f"question_{current_idx}_complete"
        })
        print(f"[COMPLETE] Moving to Q{current_idx + 2}")
        
    elif analysis.follow_up_needed and analysis.missing_parts:
        # Generate follow-up
        print(f"[GENERATING FOLLOW-UP] Missing: {analysis.missing_parts}")
        
        follow_up_prompt = f"""Generate natural follow-up question.

Original: {question_data['question']}
They said: {combined_answer}
Missing: {', '.join(analysis.missing_parts)}

Create ONE concise follow-up + 1-2 brief context questions.

Example: "Got it. And what's the pincode?"
"""
        
        messages = [
            SystemMessage(content=follow_up_prompt),
            HumanMessage(content="Generate follow-up"),
        ]
        
        follow_up = await llm.with_structured_output(FollowUpQuestion).ainvoke(messages)
        
        context_q = random.choice(follow_up.additional_context_questions) if follow_up.additional_context_questions else ""
        full_follow_up = f"{follow_up.question_text} {context_q}".strip()
        
        update_dict.update({
            "questions_answers": answers,
            "pending_follow_up": True,
            "follow_up_context": {
                "original_question": question_data["question"],
                "partial_answer": combined_answer,
                "missing_parts": analysis.missing_parts,
                "follow_up_question": full_follow_up
            },
            "current_stage": f"question_{current_idx}_follow_up"
        })
        
    else:
        # Retry or skip
        retry_count = state.get("question_retry_count", 0)
        if retry_count < MAX_RETRIES:
            update_dict.update({
                "question_retry_count": retry_count + 1,
                "pending_follow_up": False,
                "current_stage": f"question_{current_idx}_retry"
            })
            print(f"[RETRY] Attempt {retry_count + 1}/{MAX_RETRIES}")
        else:
            update_dict.update({
                "current_question_index": current_idx + 1,
                "question_retry_count": 0,
                "pending_follow_up": False,
                "current_stage": f"question_{current_idx}_skipped"
            })
            print(f"[SKIPPED] Max retries exceeded")
    
    return Command(update=update_dict)


def respond_to_question_node(state: AgentState) -> dict:
    """Final question response."""
    print("\n==== Respond to Question ====")
    
    current_idx = state.get("current_question_index", 0)
    
    if current_idx >= len(VERIFICATION_QUESTIONS):
        message = random.choice(ConversationalPrompts.COMPLETION)
    else:
        message = ""  # Minimal, acknowledgment already happened
    
    return {
        "messages": [AIMessage(content=message)] if message else {},
        "current_stage": f"question_response"
    }


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_consent(state: AgentState) -> str:
    consent_given = state.get("consent_given", False)
    retry_count = state.get("consent_retry_count", 0)
    
    if not consent_given and 0 < retry_count <= MAX_RETRIES:
        return "retry"
    return "success" if consent_given else "failure"


def route_after_authentication(state: AgentState) -> str:
    authenticated = state.get("authenticated", False)
    attempts = state.get("auth_attempts", 0)
    
    if not authenticated and 0 < attempts <= MAX_RETRIES:
        return "retry"
    return "success" if authenticated else "failure"


def route_after_question(state: AgentState) -> str:
    current_idx = state.get("current_question_index", 0)
    pending_follow_up = state.get("pending_follow_up", False)
    retry_count = state.get("question_retry_count", 0)
    
    if pending_follow_up:
        return "follow_up"
    
    answers = state.get("questions_answers", {})
    prev_idx = current_idx - 1 if current_idx > 0 else 0
    
    if prev_idx < len(VERIFICATION_QUESTIONS):
        prev_question_id = VERIFICATION_QUESTIONS[prev_idx]["id"]
        if prev_question_id in answers:
            if current_idx >= len(VERIFICATION_QUESTIONS):
                return "complete"
            return "next_question"
    
    if 0 < retry_count <= MAX_RETRIES:
        return "retry"
    
    return "complete"


def route_after_question_response(state: AgentState) -> str:
    current_idx = state.get("current_question_index", 0)
    return "end" if current_idx >= len(VERIFICATION_QUESTIONS) else "continue"


def route_after_consent_response(state: AgentState) -> str:
    return "end" if not state.get("consent_given") else "continue"


def route_after_auth_response(state: AgentState) -> str:
    return "end" if not state.get("authenticated") else "continue"


# ============================================================================
# BUILD GRAPH
# ============================================================================

workflow = StateGraph(AgentState)

system_retry_policy = RetryPolicy(max_attempts=SYSTEM_RETRIES, backoff_factor=1.0)

# Add nodes
workflow.add_node("ask_consent", ask_consent_node)
workflow.add_node("process_consent", process_consent_node, retry_policy=system_retry_policy)
workflow.add_node("respond_to_consent", respond_to_consent_node)

workflow.add_node("ask_authentication", ask_authentication_node)
workflow.add_node("process_authentication", process_authentication_node, retry_policy=system_retry_policy)
workflow.add_node("respond_to_authentication", respond_to_authentication_node)

workflow.add_node("ask_question", ask_question_node)
workflow.add_node("process_question", process_question_node, retry_policy=system_retry_policy)
workflow.add_node("respond_to_question", respond_to_question_node)

# Edges - Consent
workflow.add_edge(START, "ask_consent")
workflow.add_edge("ask_consent", "process_consent")
workflow.add_conditional_edges(
    "process_consent",
    route_after_consent,
    {"retry": "ask_consent", "success": "respond_to_consent", "failure": "respond_to_consent"},
)
workflow.add_conditional_edges(
    "respond_to_consent",
    route_after_consent_response,
    {"continue": "ask_authentication", "end": END},
)

# Edges - Auth
workflow.add_edge("ask_authentication", "process_authentication")
workflow.add_conditional_edges(
    "process_authentication",
    route_after_authentication,
    {"retry": "ask_authentication", "success": "respond_to_authentication", "failure": "respond_to_authentication"},
)
workflow.add_conditional_edges(
    "respond_to_authentication",
    route_after_auth_response,
    {"continue": "ask_question", "end": END},
)

# Edges - Questions
workflow.add_edge("ask_question", "process_question")
workflow.add_conditional_edges(
    "process_question",
    route_after_question,
    {
        "follow_up": "ask_question",
        "retry": "ask_question",
        "next_question": "respond_to_question",
        "complete": "respond_to_question",
    },
)
workflow.add_conditional_edges(
    "respond_to_question",
    route_after_question_response,
    {"continue": "ask_question", "end": END},
)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# ============================================================================
# ASYNC EXECUTION
# ============================================================================

async def run_conversation():
    """Run conversation with production-ready interrupt()."""
    
    initial_state = {
        "messages": [],
        "current_stage": "init",
        "consent_given": False,
        "consent_retry_count": 0,
        "authenticated": False,
        "auth_attempts": 0,
        "user_dob": "",
        "user_aadhar_digits": "",
        "current_question_index": 0,
        "question_retry_count": 0,
        "questions_answers": {},
        "pending_follow_up": False,
        "follow_up_context": {},
    }
    
    config = {"configurable": {"thread_id": "voice-call-1"}}
    
    print("=" * 70)
    print("🎙️  TELEPHONIC VERIFICATION CALL")
    print("=" * 70)
    
    last_message_count = 0
    
    def print_new_messages(event):
        nonlocal last_message_count
        if "messages" in event and event["messages"]:
            current_count = len(event["messages"])
            for i in range(last_message_count, current_count):
                msg = event["messages"][i]
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 Agent: {msg.content}")
            last_message_count = current_count
    
    # Initial stream
    async for event in app.astream(initial_state, config, stream_mode="values"):
        print_new_messages(event)
    
    # Main loop
    while True:
        state = await app.aget_state(config)
        
        if state.next == ():
            print("\n" + "=" * 70)
            print("✅ CALL COMPLETE")
            print("=" * 70)
            
            answers = state.values.get("questions_answers", {})
            if answers:
                print("\n📋 Collected Information:")
                for qid, data in answers.items():
                    print(f"\n  Q: {data['question']}")
                    print(f"  A: {data['answer']} (confidence: {data['confidence']}%)")
            break
        
        # Get user input
        user_input = await asyncio.get_event_loop().run_in_executor(
            None, input, "\n👤 You: "
        )
        
        if not user_input.strip():
            continue
        
        # Resume with user input
        async for event in app.astream(Command(resume=user_input), config, stream_mode="values"):
            print_new_messages(event)


if __name__ == "__main__":
    asyncio.run(run_conversation())
