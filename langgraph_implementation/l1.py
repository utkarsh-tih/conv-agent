"""
Fixed version with proper retry_policy and fallback_to implementation.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, Literal
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError
from langgraph.types import RetryPolicy
from enum import Enum
import os

# --- Configuration ---
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_RETRIES = 3  # For user ambiguity retries

# --- LLM Initialization ---
llm = ChatOllama(model=MODEL, temperature=0, base_url=BASE_URL)


# --- Enums ---
class SystemPrompts(Enum):
    PROCESS_CONSENT = (
        "You are a consent validator. The user was asked if they consent to being recorded. "
        "Analyze their response and return a JSON object.\n"
        "Consent is given for responses like: 'yes', 'yeah', 'sure', 'ok', 'I agree', 'I consent'. "
        "Consent is NOT given for: 'no', 'nope', unclear responses, or anything ambiguous."
    )
    PROCESS_AUTHENTICATION = (
        "You are an authentication validator. Extract the user's date of birth and last 4 digits of Aadhaar. "
        "Date of birth should be in DD/MM/YYYY format. "
        "Aadhaar last 4 digits should be a 4-digit number. "
        "If information is unclear or missing, set should_retry to true."
    )


class AIMessages(Enum):
    CONSENT_QUESTION = (
        "Hello! Before we begin the verification process, I need your consent. "
        "This conversation will be recorded for compliance purposes. "
        "Do you agree to proceed? (Please say yes or no)"
    )
    POSITIVE_CONSENT_RESPONSE = (
        "Thank you for your consent. Let's proceed with the verification."
    )
    NEGATIVE_CONSENT_RESPONSE = "I understand you did not consent. Goodbye."
    CONSENT_RETRY = (
        "I didn't quite understand. Could you please clearly say 'yes' or 'no'?"
    )
    AUTHENTICATION_QUESTION = (
        "For security purposes, please provide:\n"
        "1. Your date of birth (DD/MM/YYYY)\n"
        "2. Last 4 digits of your Aadhar card"
    )
    POSITIVE_AUTHENTICATION_RESPONSE = (
        "Thank you for your authentication. Let's proceed with the verification."
    )
    NEGATIVE_AUTHENTICATION_RESPONSE = "I am sorry we could not authenticate. Goodbye."
    AUTHENTICATION_RETRY = "I couldn't capture all the details. Please provide both your date of birth and last 4 digits of Aadhaar again."
    SYSTEM_ERROR_FINAL = "I'm experiencing technical difficulties. Please try again later."


# --- State and Pydantic Models ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    stage: Literal["consent", "authenticate", "questions", "complete"]
    consent_given: bool
    consent_confidence: int
    consent_reasoning: str
    consent_retry_count: int  # Tracks user ambiguity retries only
    authenticated: bool
    auth_attempts: int  # Tracks user ambiguity retries only
    user_dob: str
    user_aadhar_digits: str


class ConsentResult(BaseModel):
    consent_given: bool = Field(..., description="Whether consent was given")
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str = Field(...)
    should_retry: bool = Field(...)


class AuthenticationResult(BaseModel):
    date_of_birth: str = Field(...)
    aadhaar_last4: str = Field(...)
    confidence_date_of_birth: int = Field(..., ge=0, le=100)
    confidence_aadhaar: int = Field(..., ge=0, le=100)
    reasoning_date_of_birth: str = Field(...)
    reasoning_aadhaar: str = Field(...)
    should_retry: bool = Field(...)


# ============================================================================
# NODES - CLEAN AND SIMPLE
# ============================================================================


def ask_consent_node(state: AgentState) -> dict:
    """Ask for consent."""
    print("\n==== Ask Consent Node ===")
    
    message = (
        AIMessages.CONSENT_RETRY.value
        if state.get("consent_retry_count", 0) > 0
        else AIMessages.CONSENT_QUESTION.value
    )
    
    return {"messages": [AIMessage(content=message)], "stage": "consent"}


def process_consent_node(state: AgentState) -> dict:
    """Process consent response - only handles happy path."""
    print("\n==== Process Consent Node ===")
    consent_human_input = interrupt("waiting_for_consent")
    print(f"Received input: {consent_human_input}")

    consent_check_messages = [
        SystemMessage(content=SystemPrompts.PROCESS_CONSENT.value),
        HumanMessage(content=consent_human_input),
    ]
    
    # This will raise ValidationError if LLM fails to produce valid output
    # RetryPolicy will handle those errors automatically
    llm_output = llm.with_structured_output(ConsentResult).invoke(
        consent_check_messages
    )
    
    # Only handle application logic here
    should_retry = llm_output.should_retry
    consent_given = llm_output.consent_given

    # Determine next stage based on user response quality
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
        "messages": [HumanMessage(content=consent_human_input)],
        "consent_given": consent_given,
        "consent_confidence": llm_output.confidence,
        "consent_reasoning": llm_output.reasoning,
        "consent_retry_count": retry_count,
        "stage": next_stage,
    }


def respond_to_consent_node(state: AgentState) -> dict:
    """Generate response based on consent result."""
    print("\n==== Respond to Consent Node ===")
    
    message_map = {
        "authenticate": AIMessages.POSITIVE_CONSENT_RESPONSE.value,
        "complete": AIMessages.NEGATIVE_CONSENT_RESPONSE.value,
    }
    
    message = message_map.get(state.get("stage"))
    if message:
        return {"messages": [AIMessage(content=message)]}
    return {}


def ask_authentication_node(state: AgentState) -> dict:
    """Ask for authentication details."""
    print("\n==== Ask Authentication Node ===")
    
    message = (
        AIMessages.AUTHENTICATION_RETRY.value
        if state.get("auth_attempts", 0) > 0
        else AIMessages.AUTHENTICATION_QUESTION.value
    )
    
    return {"messages": [AIMessage(content=message)], "stage": "authenticate"}


def process_authentication_node(state: AgentState) -> dict:
    """Process authentication response - only handles happy path."""
    print("\n==== Process Authentication Node ===")
    auth_human_input = interrupt("waiting_for_authentication")
    print(f"Received input: {auth_human_input}")

    auth_check_messages = [
        SystemMessage(content=SystemPrompts.PROCESS_AUTHENTICATION.value),
        HumanMessage(content=auth_human_input),
    ]
    
    # This will raise ValidationError if LLM fails to produce valid output
    # RetryPolicy will handle those errors automatically
    llm_output = llm.with_structured_output(AuthenticationResult).invoke(
        auth_check_messages
    )
    
    # Only handle application logic here
    should_retry = llm_output.should_retry
    authenticated = (
        llm_output.confidence_date_of_birth > 70
        and llm_output.confidence_aadhaar > 70
    )

    # Determine next stage based on user response quality
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
        "messages": [HumanMessage(content=auth_human_input)],
        "authenticated": authenticated,
        "user_dob": llm_output.date_of_birth,
        "user_aadhar_digits": llm_output.aadhaar_last4,
        "auth_attempts": attempts,
        "stage": next_stage,
    }


def respond_to_authentication_node(state: AgentState) -> dict:
    """Generate response based on authentication result."""
    print("\n==== Respond to Authentication Node ===")
    
    message_map = {
        "questions": AIMessages.POSITIVE_AUTHENTICATION_RESPONSE.value,
        "complete": AIMessages.NEGATIVE_AUTHENTICATION_RESPONSE.value,
    }
    
    message = message_map.get(state.get("stage"))
    if message:
        return {"messages": [AIMessage(content=message)]}
    return {}


def handle_system_error_node(state: AgentState) -> dict:
    """Handle permanent system errors after all retries exhausted."""
    print("\n==== System Error Handler ===")
    return {
        "messages": [AIMessage(content=AIMessages.SYSTEM_ERROR_FINAL.value)],
        "stage": "complete",
    }


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================


def route_after_consent(state: AgentState) -> str:
    """Single routing decision after processing consent."""
    stage = state.get("stage", "consent")
    retry_count = state.get("consent_retry_count", 0)
    
    if stage == "consent" and retry_count < MAX_RETRIES:
        return "retry"
    
    if stage == "authenticate":
        return "success"
    
    return "failure"


def route_after_authentication(state: AgentState) -> str:
    """Single routing decision after processing authentication."""
    stage = state.get("stage", "authenticate")
    attempts = state.get("auth_attempts", 0)
    
    if stage == "authenticate" and attempts < MAX_RETRIES:
        return "retry"
    
    if stage == "questions":
        return "success"
    
    return "failure"


# ============================================================================
# BUILD GRAPH WITH RETRY POLICIES - FIXED VERSION
# ============================================================================

workflow = StateGraph(AgentState)

# Define retry policy for system-level errors
# This will silently retry ValidationErrors and other exceptions
system_retry_policy = RetryPolicy(
    max_attempts=3,  # Retry up to 3 times
    backoff_factor=1.0,  # 1 second, 2 seconds, 4 seconds
    retry_on=ValidationError,  # Retry on validation errors from structured output
)

# Add all nodes with retry policies where needed
workflow.add_node("ask_consent", ask_consent_node)
workflow.add_node(
    "process_consent", 
    process_consent_node,
    retry_policy=system_retry_policy,  # ✅ Fixed: correct parameter name
    fallback_to="handle_system_error"  # ✅ Fixed: added fallback
)
workflow.add_node("respond_to_consent", respond_to_consent_node)
workflow.add_node("ask_authentication", ask_authentication_node)
workflow.add_node(
    "process_authentication", 
    process_authentication_node,
    retry_policy=system_retry_policy,  # ✅ Fixed: correct parameter name
    fallback_to="handle_system_error"  # ✅ Fixed: added fallback
)
workflow.add_node("respond_to_authentication", respond_to_authentication_node)
workflow.add_node("handle_system_error", handle_system_error_node)

# Consent flow
workflow.add_edge(START, "ask_consent")
workflow.add_edge("ask_consent", "process_consent")

# The conditional edges ONLY run if process_consent SUCCEEDS.
# The failure case is now handled automatically by `fallback_to`.
workflow.add_conditional_edges(
    "process_consent",
    route_after_consent,
    {
        "retry": "ask_consent",
        "success": "respond_to_consent",
        "failure": "respond_to_consent",
    },
)

workflow.add_conditional_edges(
    "respond_to_consent",
    lambda state: "end" if state.get("stage") == "complete" else "continue",
    {
        "continue": "ask_authentication",
        "end": END,
    },
)

# Authentication flow
workflow.add_edge("ask_authentication", "process_authentication")

# The conditional edges ONLY run if process_authentication SUCCEEDS.
# The failure case is now handled automatically by `fallback_to`.
workflow.add_conditional_edges(
    "process_authentication",
    route_after_authentication,
    {
        "retry": "ask_authentication",
        "success": "respond_to_authentication",
        "failure": "respond_to_authentication",
    },
)

workflow.add_conditional_edges(
    "respond_to_authentication",
    lambda state: "end" if state.get("stage") == "complete" else "continue",
    {
        "continue": END,
        "end": END,
    },
)

# Error handling - this node is reached via fallback_to when retries are exhausted
workflow.add_edge("handle_system_error", END)

# Compile
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# ============================================================================
# EXECUTION
# ============================================================================


def run_conversation():
    """Run the conversation with human-in-the-loop."""

    initial_state = {
        "messages": [],
        "stage": "consent",
        "consent_given": False,
        "consent_confidence": 0,
        "consent_reasoning": "",
        "consent_retry_count": 0,
        "authenticated": False,
        "auth_attempts": 0,
        "user_dob": "",
        "user_aadhar_digits": "",
    }

    config = {"configurable": {"thread_id": "conversation-1"}}

    print("=" * 70)
    print("STARTING CONVERSATION")
    print("=" * 70)

    # Initial run until first interrupt
    for event in app.stream(initial_state, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print(f"\nAI: {last_msg.content}")

    # Main conversation loop
    while True:
        state = app.get_state(config)

        # Check if conversation ended
        if state.next == ():
            print("\n" + "=" * 70)
            print("CONVERSATION COMPLETE")
            print("=" * 70)
            print(f"Consent given: {state.values.get('consent_given')}")
            print(f"Authenticated: {state.values.get('authenticated')}")
            if state.values.get("authenticated"):
                print(f"DOB: {state.values.get('user_dob')}")
                print(f"Aadhaar last 4: {state.values.get('user_aadhar_digits')}")
            break

        # Get user input
        user_input = input("\nYou: ").strip()

        if not user_input:
            print("Please enter a response.")
            continue

        # Resume with user input
        for event in app.stream(
            Command(resume=user_input), config, stream_mode="values"
        ):
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"\nAI: {last_msg.content}")


if __name__ == "__main__":
    run_conversation()
