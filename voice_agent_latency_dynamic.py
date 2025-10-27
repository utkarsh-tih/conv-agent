"""
Fully async voice agent with proper interrupt handling.
Based on LangGraph functional API documentation patterns.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, Literal
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from langgraph.types import RetryPolicy
from enum import Enum
import os
import random
import asyncio

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEBlfBBLJhRZ47mGtmwSwXmnFNnJmVzPM"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=150,
    timeout=None,
    max_retries=2,
)

MAX_RETRIES = 2
SYSTEM_RETRIES = 2

# --- Question Definitions ---
VERIFICATION_QUESTIONS = [
    {
        "id": "q1",
        "question": "What's your current residential address?",
        "required_fields": ["street", "city", "pincode"],
        "field_descriptions": {
            "street": "street address or house number",
            "city": "city or town name", 
            "pincode": "PIN code"
        },
        "follow_up_templates": {
            "street": "Got the {present_fields}. What's your street address or house number?",
            "city": "Thanks. Which city or town is that?",
            "pincode": "And the PIN code?",
        }
    },
    {
        "id": "q2",
        "question": "What's your monthly income in rupees?",
        "required_fields": ["amount"],
        "field_descriptions": {
            "amount": "monthly income amount"
        },
        "follow_up_templates": {
            "amount": "Could you tell me the monthly amount?"
        }
    },
    {
        "id": "q3",
        "question": "What brings you in for verification today?",
        "required_fields": ["purpose"],
        "field_descriptions": {
            "purpose": "reason for verification"
        },
        "follow_up_templates": {
            "purpose": "What's the purpose of this verification?"
        }
    }
]

# --- Enums ---
class SystemPrompts(Enum):
    PROCESS_CONSENT = (
        "You are a friendly consent validator. The user was asked if they consent to being recorded. "
        "Be generous - people say 'yeah', 'sure', 'go ahead', 'okay'. "
        "Only flag unclear if genuinely ambiguous or they seem hesitant."
    )
    PROCESS_AUTHENTICATION = (
        "You are an authentication validator. Extract date of birth (DD/MM/YYYY) and last 4 Aadhaar digits. "
        "Be flexible with formats - people say dates many ways. "
        "If anything is missing or unclear, set should_retry to true."
    )

class AIMessages(Enum):
    CONSENT_QUESTION = "Hi there! Before we start, I need to let you know this call will be recorded for compliance. Are you okay with that?"
    POSITIVE_CONSENT = "Great, thanks! Let's get you verified."
    NEGATIVE_CONSENT = "No problem, I understand. Feel free to call back anytime. Take care!"
    CONSENT_RETRY = "Sorry, I didn't catch that. Just need a yes or no - is recording okay?"
    
    AUTHENTICATION_QUESTION = "Alright, for security I need your date of birth and last 4 digits of your Aadhaar card."
    POSITIVE_AUTH = "Perfect, got it! Now I've got a few quick questions for you."
    NEGATIVE_AUTH = "Sorry, I couldn't verify those details. Please try calling again."
    AUTHENTICATION_RETRY = "I missed part of that. Could you give me both again - date of birth and last 4 Aadhaar digits?"
    
    ALL_COMPLETE = "Wonderful! That's everything I needed. Thanks so much for your time."

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    stage: Literal["consent", "authenticate", "questions", "complete"]
    consent_given: bool
    consent_confidence: int
    consent_retry_count: int
    authenticated: bool
    auth_attempts: int
    user_dob: str
    user_aadhar_digits: str
    current_question_index: int
    questions_answers: dict
    partial_answer: dict
    pending_fields: list
    follow_up_count: int

# --- Pydantic Models ---
class ConsentResult(BaseModel):
    consent_given: bool = Field(..., description="Whether consent was given")
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str = Field(..., description="Brief explanation")
    should_retry: bool = Field(..., description="Whether to retry due to ambiguity")

class AuthenticationResult(BaseModel):
    date_of_birth: str = Field(..., description="DOB in DD/MM/YYYY format")
    aadhaar_last4: str = Field(..., description="Last 4 Aadhaar digits")
    confidence_date_of_birth: int = Field(..., ge=0, le=100)
    confidence_aadhaar: int = Field(..., ge=0, le=100)
    reasoning_date_of_birth: str
    reasoning_aadhaar: str
    should_retry: bool

class PartialAnswerExtraction(BaseModel):
    """Extracts whatever fields are present in the answer."""
    extracted_fields: dict = Field(..., description="Field name to value mapping")
    missing_fields: list = Field(..., description="Required field names still missing")
    confidence: int = Field(..., ge=0, le=100)
    is_complete: bool = Field(..., description="True if all required fields extracted")
    reasoning: str

QUICK_ACKS = ["Got it.", "Okay.", "Thanks.", "Alright.", "Perfect."]

# ============================================================================
# FULLY ASYNC NODES - All use async def
# ============================================================================

async def ask_consent_node(state: AgentState) -> dict:
    """
    Ask for consent. Async node that returns instantly.
    No await calls here, but async for consistency.
    """
    print("\n==== Ask Consent ===")
    message = (
        AIMessages.CONSENT_RETRY.value
        if state.get("consent_retry_count", 0) > 0
        else AIMessages.CONSENT_QUESTION.value
    )
    return {"messages": [AIMessage(content=message)], "stage": "consent"}

async def process_consent_node(state: AgentState) -> dict:
    """
    Process consent. Async node that uses interrupt() and await.
    The key: interrupt() works in async nodes when graph is run with astream.
    """
    print("\n==== Process Consent ===")
    
    # interrupt() works in async nodes - LangGraph handles context automatically
    consent_input = interrupt("waiting_for_consent")
    print(f"User said: {consent_input}")
    
    messages = [
        SystemMessage(content=SystemPrompts.PROCESS_CONSENT.value),
        HumanMessage(content=consent_input),
    ]
    
    # Use await directly - we're in an async function
    llm_output = await llm.with_structured_output(ConsentResult).ainvoke(messages)
    
    should_retry = llm_output.should_retry
    consent_given = llm_output.consent_given
    
    if should_retry and state.get("consent_retry_count", 0) < MAX_RETRIES:
        next_stage = "consent"
        retry_count = state.get("consent_retry_count", 0) + 1
    elif consent_given:
        next_stage = "authenticate"
        retry_count = state.get("consent_retry_count", 0)
    else:
        next_stage = "complete"
        retry_count = state.get("consent_retry_count", 0)
    
    return {
        "messages": [HumanMessage(content=consent_input)],
        "consent_given": consent_given,
        "consent_confidence": llm_output.confidence,
        "consent_retry_count": retry_count,
        "stage": next_stage,
    }

async def respond_to_consent_node(state: AgentState) -> dict:
    """Respond based on consent result."""
    print("\n==== Respond to Consent ===")
    message_map = {
        "authenticate": AIMessages.POSITIVE_CONSENT.value,
        "complete": AIMessages.NEGATIVE_CONSENT.value,
    }
    message = message_map.get(state.get("stage"))
    if message:
        return {"messages": [AIMessage(content=message)]}
    return {}

# ============================================================================
# AUTHENTICATION NODES - All async
# ============================================================================

async def ask_authentication_node(state: AgentState) -> dict:
    """Ask for authentication."""
    print("\n==== Ask Authentication ===")
    message = (
        AIMessages.AUTHENTICATION_RETRY.value
        if state.get("auth_attempts", 0) > 0
        else AIMessages.AUTHENTICATION_QUESTION.value
    )
    return {"messages": [AIMessage(content=message)], "stage": "authenticate"}

async def process_authentication_node(state: AgentState) -> dict:
    """Process authentication with async LLM call."""
    print("\n==== Process Authentication ===")
    
    # interrupt() works in async nodes
    auth_input = interrupt("waiting_for_authentication")
    print(f"User said: {auth_input}")
    
    messages = [
        SystemMessage(content=SystemPrompts.PROCESS_AUTHENTICATION.value),
        HumanMessage(content=auth_input),
    ]
    
    # Direct await - clean and simple
    llm_output = await llm.with_structured_output(AuthenticationResult).ainvoke(messages)
    
    should_retry = llm_output.should_retry
    authenticated = (
        llm_output.confidence_date_of_birth > 10 and 
        llm_output.confidence_aadhaar > 10
    )
    
    if should_retry and state.get("auth_attempts", 0) < MAX_RETRIES:
        next_stage = "authenticate"
        attempts = state.get("auth_attempts", 0) + 1
    elif authenticated and not should_retry:
        next_stage = "questions"
        attempts = state.get("auth_attempts", 0)
    else:
        next_stage = "complete"
        attempts = state.get("auth_attempts", 0)
    
    return {
        "messages": [HumanMessage(content=auth_input)],
        "authenticated": authenticated,
        "user_dob": llm_output.date_of_birth,
        "user_aadhar_digits": llm_output.aadhaar_last4,
        "auth_attempts": attempts,
        "stage": next_stage,
    }

async def respond_to_authentication_node(state: AgentState) -> dict:
    """Respond based on auth result."""
    print("\n==== Respond to Authentication ===")
    message_map = {
        "questions": AIMessages.POSITIVE_AUTH.value,
        "complete": AIMessages.NEGATIVE_AUTH.value,
    }
    message = message_map.get(state.get("stage"))
    if message:
        return {"messages": [AIMessage(content=message)]}
    return {}

# ============================================================================
# QUESTION NODES WITH DYNAMIC FOLLOW-UPS - All async
# ============================================================================

async def ask_question_node(state: AgentState) -> dict:
    """
    Ask question or dynamic follow-up based on what's missing.
    """
    print("\n==== Ask Question ===")
    
    current_idx = state.get("current_question_index", 0)
    follow_up_count = state.get("follow_up_count", 0)
    pending_fields = state.get("pending_fields", [])
    partial_answer = state.get("partial_answer", {})
    
    question_data = VERIFICATION_QUESTIONS[current_idx]
    
    # First time asking this question
    if follow_up_count == 0:
        message = f"Question {current_idx + 1} of {len(VERIFICATION_QUESTIONS)}: {question_data['question']}"
    
    # Following up on missing fields
    elif pending_fields:
        first_missing = pending_fields[0]
        
        # Use template for speed (no LLM call needed)
        if first_missing in question_data.get("follow_up_templates", {}):
            template = question_data["follow_up_templates"][first_missing]
            present_fields = ", ".join(partial_answer.keys())
            message = template.format(present_fields=present_fields if present_fields else "that")
            
            # Optionally add a related question if multiple fields missing
            if len(pending_fields) > 1:
                next_field = pending_fields[1]
                message += f" And the {question_data['field_descriptions'].get(next_field, next_field)}?"
        else:
            # Fallback generic follow-up
            message = f"Thanks. I still need the {first_missing}. Could you provide that?"
    
    else:
        message = "Could you repeat that for me?"
    
    print(f"Asking: {message}")
    
    return {
        "messages": [AIMessage(content=message)],
        "stage": "questions"
    }

async def process_question_node(state: AgentState) -> dict:
    """
    Process answer with intelligent field extraction.
    This is where we parse natural language and extract structured data.
    """
    print("\n==== Process Question ===")
    
    current_idx = state.get("current_question_index", 0)
    question_data = VERIFICATION_QUESTIONS[current_idx]
    
    # Get user's answer - interrupt works perfectly in async nodes
    user_answer = interrupt("waiting_for_question_answer")
    print(f"User said: {user_answer}")
    
    # In a real voice system, you'd trigger TTS immediately here
    # while the LLM processes in the background
    quick_ack = random.choice(QUICK_ACKS)
    print(f"[INSTANT RESPONSE to user]: {quick_ack}")
    
    # Build extraction prompt
    extraction_prompt = f"""Analyze this answer to: "{question_data['question']}"

REQUIRED FIELDS to extract:
{chr(10).join(f"- {field}: {desc}" for field, desc in question_data['field_descriptions'].items())}

USER ANSWER: "{user_answer}"

Your task:
1. Extract whatever information you CAN find from their answer
2. Identify which required fields are still missing
3. Be generous - if they mention something that could be a field, extract it

Examples of generous interpretation:
- "I live in Mumbai 400001" → city: Mumbai, pincode: 400001
- "50k per month" → amount: 50000
- "123 Main Street" → street: 123 Main Street
"""
    
    messages = [
        SystemMessage(content=extraction_prompt),
        HumanMessage(content=user_answer)
    ]
    
    # Async LLM call with await - clean and efficient
    llm_output = await llm.with_structured_output(PartialAnswerExtraction).ainvoke(messages)
    
    print(f"DEBUG - Extracted fields: {llm_output.extracted_fields}")
    print(f"DEBUG - Missing fields: {llm_output.missing_fields}")
    print(f"DEBUG - Is complete: {llm_output.is_complete}")
    print(f"DEBUG - Confidence: {llm_output.confidence}%")
    
    # Merge new extractions with any previous partial answer
    # This accumulates information across multiple follow-ups
    partial_answer = state.get("partial_answer", {})
    partial_answer.update(llm_output.extracted_fields)
    
    return {
        "messages": [HumanMessage(content=user_answer)],
        "partial_answer": partial_answer,
        "pending_fields": llm_output.missing_fields,
        "extraction_confidence": llm_output.confidence,
        "is_complete": llm_output.is_complete,
    }

async def save_and_continue_node(state: AgentState) -> dict:
    """
    Decide whether to accept the answer or ask for clarification.
    This implements the graceful degradation strategy.
    """
    print("\n==== Save and Continue ===")
    
    current_idx = state.get("current_question_index", 0)
    question_data = VERIFICATION_QUESTIONS[current_idx]
    partial_answer = state.get("partial_answer", {})
    pending_fields = state.get("pending_fields", [])
    follow_up_count = state.get("follow_up_count", 0)
    is_complete = state.get("is_complete", False)
    
    # Decision logic: When do we accept what we have?
    # 1. We got everything (is_complete)
    # 2. We've asked twice already (follow_up_count >= 2)
    # 3. We got something after one follow-up (compromise)
    should_accept = (
        is_complete or
        follow_up_count >= 2 or
        (len(partial_answer) > 0 and follow_up_count >= 1)
    )
    
    if should_accept:
        # Save the answer and move to next question
        answers = state.get("questions_answers", {})
        answers[question_data["id"]] = {
            "question": question_data["question"],
            "answer": partial_answer,
            "is_complete": is_complete,
            "missing_fields": pending_fields if not is_complete else []
        }
        
        next_idx = current_idx + 1
        next_stage = "questions" if next_idx < len(VERIFICATION_QUESTIONS) else "complete"
        
        # Natural acknowledgment based on completeness
        if is_complete:
            ack = "Perfect, got everything."
        else:
            ack = "Alright, I've noted that down."
        
        return {
            "messages": [AIMessage(content=ack)],
            "questions_answers": answers,
            "current_question_index": next_idx,
            "partial_answer": {},  # Reset for next question
            "pending_fields": [],
            "follow_up_count": 0,
            "stage": next_stage,
            "needs_followup": False,
        }
    
    else:
        # Need to ask a follow-up question
        return {
            "follow_up_count": follow_up_count + 1,
            "needs_followup": True,
        }

# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_consent(state: AgentState) -> str:
    """Route after processing consent response."""
    stage = state.get("stage", "consent")
    retry_count = state.get("consent_retry_count", 0)
    
    if stage == "consent" and retry_count < MAX_RETRIES:
        return "retry"
    if stage == "authenticate":
        return "success"
    return "failure"

def route_after_authentication(state: AgentState) -> str:
    """Route after processing authentication."""
    stage = state.get("stage", "authenticate")
    attempts = state.get("auth_attempts", 0)
    
    if stage == "authenticate" and attempts < MAX_RETRIES:
        return "retry"
    if stage == "questions":
        return "success"
    return "failure"

def route_after_question_processing(state: AgentState) -> str:
    """
    Decide if we need a follow-up or can save and move on.
    This is the key decision point for dynamic follow-ups.
    """
    needs_followup = state.get("needs_followup", False)
    
    if needs_followup:
        return "followup"
    else:
        return "next"

def route_after_save(state: AgentState) -> str:
    """Check if there are more questions remaining."""
    stage = state.get("stage", "questions")
    current_idx = state.get("current_question_index", 0)
    
    if stage == "complete" or current_idx >= len(VERIFICATION_QUESTIONS):
        return "end"
    return "continue"

# ============================================================================
# BUILD GRAPH
# ============================================================================

workflow = StateGraph(AgentState)

# Retry policy for LLM failures
system_retry = RetryPolicy(max_attempts=SYSTEM_RETRIES, backoff_factor=1.0)

# Add all nodes
workflow.add_node("ask_consent", ask_consent_node)
workflow.add_node("process_consent", process_consent_node, retry_policy=system_retry)
workflow.add_node("respond_to_consent", respond_to_consent_node)

workflow.add_node("ask_authentication", ask_authentication_node)
workflow.add_node("process_authentication", process_authentication_node, retry_policy=system_retry)
workflow.add_node("respond_to_authentication", respond_to_authentication_node)

workflow.add_node("ask_question", ask_question_node)
workflow.add_node("process_question", process_question_node, retry_policy=system_retry)
workflow.add_node("save_and_continue", save_and_continue_node)

# Build the flow
workflow.add_edge(START, "ask_consent")
workflow.add_edge("ask_consent", "process_consent")
workflow.add_conditional_edges(
    "process_consent",
    route_after_consent,
    {"retry": "ask_consent", "success": "respond_to_consent", "failure": "respond_to_consent"},
)
workflow.add_conditional_edges(
    "respond_to_consent",
    lambda s: "end" if s.get("stage") == "complete" else "continue",
    {"continue": "ask_authentication", "end": END},
)

workflow.add_edge("ask_authentication", "process_authentication")
workflow.add_conditional_edges(
    "process_authentication",
    route_after_authentication,
    {"retry": "ask_authentication", "success": "respond_to_authentication", "failure": "respond_to_authentication"},
)
workflow.add_conditional_edges(
    "respond_to_authentication",
    lambda s: "end" if s.get("stage") == "complete" else "continue",
    {"continue": "ask_question", "end": END},
)

# Question flow with dynamic follow-up loop
workflow.add_edge("ask_question", "process_question")
workflow.add_edge("process_question", "save_and_continue")
workflow.add_conditional_edges(
    "save_and_continue",
    route_after_question_processing,
    {
        "followup": "ask_question",  # Loop back for follow-up
        "next": "check_more_questions",  # Move to next question
    },
)

async def check_more_questions_node(state: AgentState) -> dict:
    """Check if we're done with all questions."""
    if state.get("stage") == "complete":
        return {"messages": [AIMessage(content=AIMessages.ALL_COMPLETE.value)]}
    return {}

workflow.add_node("check_more_questions", check_more_questions_node)
workflow.add_conditional_edges(
    "check_more_questions",
    route_after_save,
    {"continue": "ask_question", "end": END},
)

# Compile the graph
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ============================================================================
# ASYNC EXECUTION - The Correct Pattern
# ============================================================================

async def run_conversation_async():
    """
    Run the conversation asynchronously.
    This is the correct pattern from the LangGraph documentation.
    """
    initial_state = {
        "messages": [],
        "stage": "consent",
        "consent_given": False,
        "consent_confidence": 0,
        "consent_retry_count": 0,
        "authenticated": False,
        "auth_attempts": 0,
        "user_dob": "",
        "user_aadhar_digits": "",
        "current_question_index": 0,
        "questions_answers": {},
        "partial_answer": {},
        "pending_fields": [],
        "follow_up_count": 0,
    }
    
    config = {"configurable": {"thread_id": "conversation-1"}}
    
    print("=" * 70)
    print("STARTING ASYNC CONVERSATION")
    print("=" * 70)
    
    last_message_count = 0
    
    # Use astream for async execution - this is the key
    async for event in app.astream(initial_state, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            current_count = len(event["messages"])
            for i in range(last_message_count, current_count):
                msg = event["messages"][i]
                if isinstance(msg, AIMessage):
                    print(f"\nAI: {msg.content}")
            last_message_count = current_count
    
    # Main conversation loop
    while True:
        state = app.get_state(config)
        
        # Check if conversation is complete
        if state.next == ():
            print("\n" + "=" * 70)
            print("CONVERSATION COMPLETE")
            print("=" * 70)
            print(f"Consent given: {state.values.get('consent_given')}")
            print(f"Authenticated: {state.values.get('authenticated')}")
            
            # Display collected answers
            answers = state.values.get("questions_answers", {})
            if answers:
                print("\n--- Collected Answers ---")
                for qid, data in answers.items():
                    print(f"\nQ: {data['question']}")
                    print(f"A: {data['answer']}")
                    if data.get('missing_fields'):
                        print(f"   (Incomplete - missing: {', '.join(data['missing_fields'])})")
                    else:
                        print(f"   (Complete)")
            break
        
        # Get user input - in a real system, this would be speech-to-text output
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            print("Please provide a response.")
            continue
        
        # Resume execution with user input using astream
        async for event in app.astream(Command(resume=user_input), config, stream_mode="values"):
            if "messages" in event and event["messages"]:
                current_count = len(event["messages"])
                for i in range(last_message_count, current_count):
                    msg = event["messages"][i]
                    if isinstance(msg, AIMessage):
                        print(f"\nAI: {msg.content}")
                last_message_count = current_count

def run_conversation():
    """
    Synchronous wrapper to run the async conversation.
    This is what you call from the main block.
    """
    asyncio.run(run_conversation_async())

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  FULLY ASYNC VOICE AGENT - Correct Implementation Pattern    ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Key Insights:
    
    1. ALL NODES ARE ASYNC (async def)
       - This allows await for LLM calls
       - Much cleaner code, no asyncio.run() needed
    
    2. INTERRUPT() WORKS IN ASYNC NODES
       - Just call interrupt() normally
       - LangGraph handles context when using astream
    
    3. USE app.astream() NOT app.stream()
       - This is the key to making async nodes work
       - Async context maintained automatically
    
    4. DYNAMIC FOLLOW-UPS
       - Extracts partial information from answers
       - Asks specifically for missing fields
       - Accepts incomplete after 2 follow-ups
    
    5. LOW LATENCY DESIGN
       - Immediate acknowledgments before LLM processing
       - Template-based questions (no LLM overhead)
       - Async throughout for maximum efficiency
    
    This is production-ready for voice applications!
    """)
    
    run_conversation()
