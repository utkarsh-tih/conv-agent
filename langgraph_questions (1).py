"""
Enhanced version with questions flow after authentication.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, Literal, List
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError
from langgraph.types import RetryPolicy
from enum import Enum
import os

# --- Constants ---
MAX_RETRIES = 3  # For user ambiguity retries
SYSTEM_RETRIES = 3  # For LLM/system errors

# # --- Configuration ---
# MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
# BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# # --- LLM Initialization ---
# llm = ChatOllama(model=MODEL, temperature=0, base_url=BASE_URL)

from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEBlfBBLJhRZ47mGtmwSwXmnFNnJmVzPM"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

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
    PROCESS_QUESTION = (
        "You are a question response validator. "
        "The user was asked: '{question}'\n"
        "Analyze their response for clarity and completeness. "
        "Extract the answer and determine if it's clear enough or needs clarification."
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
        "I didn't quite understand. "
        "To begin the verification process, I need your consent. "
        "This conversation will be recorded for compliance purposes. "
        "Could you please clearly say 'yes' or 'no'?"
    )
    AUTHENTICATION_QUESTION = (
        "For security purposes, please provide:\n"
        "1. Your date of birth (DD/MM/YYYY)\n"
        "2. Last 4 digits of your Aadhar card"
    )
    POSITIVE_AUTHENTICATION_RESPONSE = (
        "Thank you for your authentication. Now I'll ask you a few verification questions."
    )
    NEGATIVE_AUTHENTICATION_RESPONSE = "I am sorry we could not authenticate. Goodbye."
    AUTHENTICATION_RETRY = "I couldn't capture all the details. Please provide both your date of birth and last 4 digits of Aadhaar again."
    QUESTION_RETRY = "I didn't quite catch that. Could you please answer the question again?"
    ALL_QUESTIONS_COMPLETE = "Thank you! We've completed all verification questions."
    SYSTEM_ERROR_FINAL = (
        "I'm experiencing technical difficulties. Please try again later."
    )


# --- Question Definitions ---
VERIFICATION_QUESTIONS = [
    {
        "id": "q1",
        "question": "What is your current residential address?",
        "validation_hint": "Should include street, city, and pincode"
    },
    {
        "id": "q2", 
        "question": "What is your monthly income?",
        "validation_hint": "Should be a numeric value in rupees"
    },
    {
        "id": "q3",
        "question": "What is the purpose of this verification?",
        "validation_hint": "Should be a clear statement of purpose"
    }
]


# --- State and Pydantic Models ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    stage: Literal["consent", "authenticate", "questions", "complete"]
    consent_given: bool
    consent_confidence: int
    consent_reasoning: str
    consent_retry_count: int
    authenticated: bool
    auth_attempts: int
    user_dob: str
    user_aadhar_digits: str
    # Question-related state
    current_question_index: int
    question_retry_count: int  # Retries for current question
    questions_answers: dict  # Stores {question_id: answer}


class ConsentResult(BaseModel):
    """Analysis of user consent from text."""

    consent_given: bool = Field(..., description="Whether consent was given")
    confidence: int = Field(
        ...,
        description="Confidence level of consent interpretation (0-100)",
        ge=0,
        le=100,
    )
    reasoning: str = Field(
        ..., description="Brief explanation of the consent interpretation"
    )
    should_retry: bool = Field(
        ..., description="Whether to retry asking for consent because of ambiguity"
    )


class AuthenticationResult(BaseModel):
    """Analysis of user authentication from text."""

    date_of_birth: str = Field(
        ..., description="User's date of birth in DD/MM/YYYY format"
    )
    aadhaar_last4: str = Field(..., description="Last 4 digits of user's Aadhaar card")
    confidence_date_of_birth: int = Field(
        ...,
        description="Confidence level of date of birth interpretation (0-100)",
        ge=0,
        le=100,
    )
    confidence_aadhaar: int = Field(
        ...,
        description="Confidence level of Aadhaar interpretation (0-100)",
        ge=0,
        le=100,
    )
    reasoning_date_of_birth: str = Field(
        ..., description="Brief explanation of the date of birth interpretation"
    )
    reasoning_aadhaar: str = Field(
        ..., description="Brief explanation of the Aadhaar interpretation"
    )
    should_retry: bool = Field(
        ...,
        description="Whether to retry asking for authentication because of ambiguity",
    )


class QuestionResult(BaseModel):
    """Analysis of user's answer to a question."""
    answer: str = Field(..., description="Extracted answer from user response")
    confidence: int = Field(..., description="Confidence level (0-100)", ge=0, le=100)
    reasoning: str = Field(..., description="Why this confidence level")
    should_retry: bool = Field(..., description="Whether answer is too ambiguous")
    is_complete: bool = Field(..., description="Whether answer addresses the question")


# ============================================================================
# CONSENT & AUTH NODES (unchanged)
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
    """Process consent response."""
    print("\n==== Process Consent Node ===")
    consent_human_input = interrupt("waiting_for_consent")
    print(f"Received input: {consent_human_input}")

    consent_check_messages = [
        SystemMessage(content=SystemPrompts.PROCESS_CONSENT.value),
        HumanMessage(content=consent_human_input),
    ]

    llm_output = llm.with_structured_output(ConsentResult).invoke(consent_check_messages)

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
    """Process authentication response."""
    print("\n==== Process Authentication Node ===")
    auth_human_input = interrupt("waiting_for_authentication")
    print(f"Received input: {auth_human_input}")

    auth_check_messages = [
        SystemMessage(content=SystemPrompts.PROCESS_AUTHENTICATION.value),
        HumanMessage(content=auth_human_input),
    ]

    llm_output = llm.with_structured_output(AuthenticationResult).invoke(auth_check_messages)

    print(f"DEBUG - DOB: {llm_output.date_of_birth}, Confidence: {llm_output.confidence_date_of_birth}")
    print(f"DEBUG - Aadhaar: {llm_output.aadhaar_last4}, Confidence: {llm_output.confidence_aadhaar}")
    print(f"DEBUG - Reasoning DOB: {llm_output.reasoning_date_of_birth}")
    print(f"DEBUG - Reasoning Aadhaar: {llm_output.reasoning_aadhaar}")
    print(f"DEBUG - Should retry: {llm_output.should_retry}")


    should_retry = llm_output.should_retry
    authenticated = (
        llm_output.confidence_date_of_birth > 10 and llm_output.confidence_aadhaar > 10
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


# ============================================================================
# QUESTION NODES - NEW
# ============================================================================

def ask_question_node(state: AgentState) -> dict:
    """Ask the current question."""
    print("\n==== Ask Question Node ===")
    
    current_idx = state.get("current_question_index", 0)
    retry_count = state.get("question_retry_count", 0)
    
    # Get current question
    question_data = VERIFICATION_QUESTIONS[current_idx]
    question_text = question_data["question"]
    
    # Add retry message if this is a retry
    if retry_count > 0:
        message = f"{AIMessages.QUESTION_RETRY.value}\n\n{question_text}"
    else:
        message = f"Question {current_idx + 1} of {len(VERIFICATION_QUESTIONS)}: {question_text}"
    
    print(f"Asking question {current_idx + 1}: {question_text}")
    
    return {
        "messages": [AIMessage(content=message)],
        "stage": "questions"
    }


def process_question_node(state: AgentState) -> dict:
    """Process answer to current question."""
    print("\n==== Process Question Node ===")
    
    current_idx = state.get("current_question_index", 0)
    question_data = VERIFICATION_QUESTIONS[current_idx]
    
    # Get user input
    user_answer = interrupt("waiting_for_question_answer")
    print(f"Received answer: {user_answer}")
    
    # Build validation prompt with question context
    question_prompt = SystemPrompts.PROCESS_QUESTION.value.format(
        question=question_data["question"]
    )
    question_prompt += f"\n\nValidation hint: {question_data['validation_hint']}"
    
    validation_messages = [
        SystemMessage(content=question_prompt),
        HumanMessage(content=user_answer),
    ]
    
    # Get LLM validation (will auto-retry on ValidationError)
    llm_output = llm.with_structured_output(QuestionResult).invoke(validation_messages)

    print(f"DEBUG - {llm_output}")

    # Application logic
    should_retry = llm_output.should_retry or not llm_output.is_complete
    retry_count = state.get("question_retry_count", 0)
    
    # Determine next action
    if should_retry and retry_count < MAX_RETRIES:
        # Retry same question
        next_stage = "questions"
        new_retry_count = retry_count + 1
        new_question_index = current_idx  # Stay on same question
        
    elif llm_output.confidence > 60:  # Accept answer even if not perfect
        # Save answer and move to next question
        answers = state.get("questions_answers", {})
        answers[question_data["id"]] = {
            "question": question_data["question"],
            "answer": llm_output.answer,
            "confidence": llm_output.confidence
        }
        
        next_stage = "questions"
        new_retry_count = 0  # Reset for next question
        new_question_index = current_idx + 1
        
        # Check if all questions done
        if new_question_index >= len(VERIFICATION_QUESTIONS):
            next_stage = "complete"
            
    else:
        # Failed after max retries or low confidence
        next_stage = "complete"
        new_retry_count = retry_count
        new_question_index = current_idx
    
    return {
        "messages": [HumanMessage(content=user_answer)],
        "questions_answers": state.get("questions_answers", {}),
        "current_question_index": new_question_index,
        "question_retry_count": new_retry_count,
        "stage": next_stage,
    }


def respond_to_question_node(state: AgentState) -> dict:
    """Generate response based on question processing result."""
    print("\n==== Respond to Question Node ===")
    
    current_idx = state.get("current_question_index", 0)
    
    # If we've completed all questions
    if current_idx >= len(VERIFICATION_QUESTIONS):
        return {
            "messages": [AIMessage(content=AIMessages.ALL_QUESTIONS_COMPLETE.value)],
            "stage": "complete"
        }
    
    # If we're continuing to next question, just acknowledge
    if state.get("question_retry_count", 0) == 0:
        return {"messages": [AIMessage(content="Thank you. Next question:")]}
    
    # If we're retrying, the ask_question_node will handle the retry message
    return {}


# ============================================================================
# ERROR HANDLING
# ============================================================================

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
    """Route after processing consent."""
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


def route_after_question(state: AgentState) -> str:
    """Route after processing a question answer."""
    stage = state.get("stage", "questions")
    current_idx = state.get("current_question_index", 0)
    retry_count = state.get("question_retry_count", 0)
    
    # If we need to retry the same question
    if retry_count > 0 and retry_count <= MAX_RETRIES:
        return "retry"
    
    # If we're moving to next question
    if stage == "questions" and current_idx < len(VERIFICATION_QUESTIONS):
        return "next_question"
    
    # If all questions complete or max retries exceeded
    return "complete"


def route_after_question_response(state: AgentState) -> str:
    """Route after responding to question."""
    stage = state.get("stage", "questions")
    current_idx = state.get("current_question_index", 0)
    
    if stage == "complete" or current_idx >= len(VERIFICATION_QUESTIONS):
        return "end"
    
    return "continue"


# ============================================================================
# BUILD GRAPH
# ============================================================================

workflow = StateGraph(AgentState)

# Define retry policy for system-level errors
system_retry_policy = RetryPolicy(
    max_attempts=SYSTEM_RETRIES,
    backoff_factor=1.0,
    retry_on=ValidationError,
)

# Add all nodes
workflow.add_node("ask_consent", ask_consent_node)
workflow.add_node("process_consent", process_consent_node, 
                  retry_policy=system_retry_policy, fallback_to="handle_system_error")
workflow.add_node("respond_to_consent", respond_to_consent_node)

workflow.add_node("ask_authentication", ask_authentication_node)
workflow.add_node("process_authentication", process_authentication_node,
                  retry_policy=system_retry_policy, fallback_to="handle_system_error")
workflow.add_node("respond_to_authentication", respond_to_authentication_node)

# Question nodes - NEW
workflow.add_node("ask_question", ask_question_node)
workflow.add_node("process_question", process_question_node,
                  retry_policy=system_retry_policy, fallback_to="handle_system_error")
workflow.add_node("respond_to_question", respond_to_question_node)

workflow.add_node("handle_system_error", handle_system_error_node)

# Consent flow (unchanged)
workflow.add_edge(START, "ask_consent")
workflow.add_edge("ask_consent", "process_consent")
workflow.add_conditional_edges(
    "process_consent",
    route_after_consent,
    {"retry": "ask_consent", "success": "respond_to_consent", "failure": "respond_to_consent"},
)
workflow.add_conditional_edges(
    "respond_to_consent",
    lambda state: "end" if state.get("stage") == "complete" else "continue",
    {"continue": "ask_authentication", "end": END},
)

# Authentication flow (unchanged)
workflow.add_edge("ask_authentication", "process_authentication")
workflow.add_conditional_edges(
    "process_authentication",
    route_after_authentication,
    {"retry": "ask_authentication", "success": "respond_to_authentication", "failure": "respond_to_authentication"},
)
workflow.add_conditional_edges(
    "respond_to_authentication",
    lambda state: "end" if state.get("stage") == "complete" else "continue",
    {"continue": "ask_question", "end": END},
)

# Questions flow - NEW
workflow.add_edge("ask_question", "process_question")
workflow.add_conditional_edges(
    "process_question",
    route_after_question,
    {
        "retry": "ask_question",  # Retry same question
        "next_question": "respond_to_question",  # Move to next
        "complete": "respond_to_question",  # All done or failed
    },
)
workflow.add_conditional_edges(
    "respond_to_question",
    route_after_question_response,
    {
        "continue": "ask_question",  # Ask next question
        "end": END,  # All questions complete
    },
)

# Error handling
workflow.add_edge("handle_system_error", END)

# Compile
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

print(app.get_graph().draw_mermaid())
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
        "current_question_index": 0,
        "question_retry_count": 0,
        "questions_answers": {},
    }

    config = {"configurable": {"thread_id": "conversation-1"}}

    print("=" * 70)
    print("STARTING CONVERSATION")
    print("=" * 70)

    # Track the last message count to avoid reprinting
    last_message_count = 0

    # Initial run until first interrupt
    for event in app.stream(initial_state, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            # Only print new messages
            current_count = len(event["messages"])
            for i in range(last_message_count, current_count):
                msg = event["messages"][i]
                if isinstance(msg, AIMessage):
                    print(f"\nAI: {msg.content}")
            last_message_count = current_count

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
            
            # Print collected answers
            answers = state.values.get("questions_answers", {})
            if answers:
                print("\n--- Collected Answers ---")
                for qid, data in answers.items():
                    print(f"\nQ: {data['question']}")
                    print(f"A: {data['answer']} (confidence: {data['confidence']}%)")
            break

        # Get user input
        user_input = input("\nYou: ").strip()

        if not user_input:
            print("Please enter a response.")
            continue

        # Resume with user input
        for event in app.stream(Command(resume=user_input), config, stream_mode="values"):
            if "messages" in event and event["messages"]:
                # Only print new messages
                current_count = len(event["messages"])
                for i in range(last_message_count, current_count):
                    msg = event["messages"][i]
                    if isinstance(msg, AIMessage):
                        print(f"\nAI: {msg.content}")
                last_message_count = current_count


if __name__ == "__main__":
    run_conversation()
