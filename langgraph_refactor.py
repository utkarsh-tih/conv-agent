"""
Refactored version with routing logic in conditional edges.
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
MAX_RETRIES = 3
SYSTEM_RETRIES = 3

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
    current_stage: str  # For debugging only
    
    # Consent data
    consent_given: bool
    consent_confidence: int
    consent_reasoning: str
    consent_retry_count: int
    
    # Authentication data
    authenticated: bool
    auth_attempts: int
    user_dob: str
    user_aadhar_digits: str
    
    # Question data
    current_question_index: int
    question_retry_count: int
    questions_answers: dict


class ConsentResult(BaseModel):
    consent_given: bool = Field(..., description="Whether consent was given")
    confidence: int = Field(..., description="Confidence level (0-100)", ge=0, le=100)
    reasoning: str = Field(..., description="Brief explanation")
    should_retry: bool = Field(..., description="Whether to retry due to ambiguity")


class AuthenticationResult(BaseModel):
    date_of_birth: str = Field(..., description="Date of birth in DD/MM/YYYY format")
    aadhaar_last4: str = Field(..., description="Last 4 digits of Aadhaar")
    confidence_date_of_birth: int = Field(..., description="Confidence level (0-100)", ge=0, le=100)
    confidence_aadhaar: int = Field(..., description="Confidence level (0-100)", ge=0, le=100)
    reasoning_date_of_birth: str = Field(..., description="DOB reasoning")
    reasoning_aadhaar: str = Field(..., description="Aadhaar reasoning")
    should_retry: bool = Field(..., description="Whether to retry due to ambiguity")


class QuestionResult(BaseModel):
    answer: str = Field(..., description="Extracted answer")
    confidence: int = Field(..., description="Confidence level (0-100)", ge=0, le=100)
    reasoning: str = Field(..., description="Reasoning")
    should_retry: bool = Field(..., description="Whether answer is ambiguous")
    is_complete: bool = Field(..., description="Whether answer addresses the question")


# ============================================================================
# CONSENT NODES
# ============================================================================

def ask_consent_node(state: AgentState) -> dict:
    """Ask for consent."""
    print("\n==== Ask Consent Node ===")
    message = (
        AIMessages.CONSENT_RETRY.value
        if state.get("consent_retry_count", 0) > 0
        else AIMessages.CONSENT_QUESTION.value
    )
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": "asking_consent"
    }


def process_consent_node(state: AgentState) -> dict:
    """Process consent response - validation and state updates."""
    print("\n==== Process Consent Node ===")
    consent_human_input = interrupt("waiting_for_consent")
    print(f"Received input: {consent_human_input}")

    consent_check_messages = [
        SystemMessage(content=SystemPrompts.PROCESS_CONSENT.value),
        HumanMessage(content=consent_human_input),
    ]

    llm_output = llm.with_structured_output(ConsentResult).invoke(consent_check_messages)

    # Determine if we should increment retry count
    retry_count = state.get("consent_retry_count", 0)
    should_retry = llm_output.should_retry and not llm_output.consent_given
    
    if should_retry and retry_count < MAX_RETRIES:
        new_retry_count = retry_count + 1
    else:
        new_retry_count = retry_count

    return {
        "messages": [HumanMessage(content=consent_human_input)],
        "consent_given": llm_output.consent_given,
        "consent_confidence": llm_output.confidence,
        "consent_reasoning": llm_output.reasoning,
        "consent_retry_count": new_retry_count,
        "current_stage": "consent_processed"
    }


def respond_to_consent_node(state: AgentState) -> dict:
    """Respond based on consent."""
    print("\n==== Respond to Consent Node ===")
    
    if state.get("consent_given"):
        message = AIMessages.POSITIVE_CONSENT_RESPONSE.value
    else:
        message = AIMessages.NEGATIVE_CONSENT_RESPONSE.value
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": "consent_responded"
    }


# ============================================================================
# AUTHENTICATION NODES
# ============================================================================

def ask_authentication_node(state: AgentState) -> dict:
    """Ask for authentication."""
    print("\n==== Ask Authentication Node ===")
    message = (
        AIMessages.AUTHENTICATION_RETRY.value
        if state.get("auth_attempts", 0) > 0
        else AIMessages.AUTHENTICATION_QUESTION.value
    )
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": "asking_authentication"
    }


def process_authentication_node(state: AgentState) -> dict:
    """Process authentication - validation and state updates."""
    print("\n==== Process Authentication Node ===")
    auth_human_input = interrupt("waiting_for_authentication")
    print(f"Received input: {auth_human_input}")

    auth_check_messages = [
        SystemMessage(content=SystemPrompts.PROCESS_AUTHENTICATION.value),
        HumanMessage(content=auth_human_input),
    ]

    llm_output = llm.with_structured_output(AuthenticationResult).invoke(auth_check_messages)

    authenticated = (
        llm_output.confidence_date_of_birth > 10 and 
        llm_output.confidence_aadhaar > 10 and
        not llm_output.should_retry
    )

    # Update retry count
    attempts = state.get("auth_attempts", 0)
    should_retry = not authenticated and llm_output.should_retry
    
    if should_retry and attempts < MAX_RETRIES:
        new_attempts = attempts + 1
    else:
        new_attempts = attempts

    return {
        "messages": [HumanMessage(content=auth_human_input)],
        "authenticated": authenticated,
        "user_dob": llm_output.date_of_birth,
        "user_aadhar_digits": llm_output.aadhaar_last4,
        "auth_attempts": new_attempts,
        "current_stage": "authentication_processed"
    }


def respond_to_authentication_node(state: AgentState) -> dict:
    """Respond based on authentication."""
    print("\n==== Respond to Authentication Node ===")
    
    if state.get("authenticated"):
        message = AIMessages.POSITIVE_AUTHENTICATION_RESPONSE.value
    else:
        message = AIMessages.NEGATIVE_AUTHENTICATION_RESPONSE.value
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": "authentication_responded"
    }


# ============================================================================
# QUESTION NODES
# ============================================================================

def ask_question_node(state: AgentState) -> dict:
    """Ask the current question."""
    print("\n==== Ask Question Node ===")
    
    current_idx = state.get("current_question_index", 0)
    retry_count = state.get("question_retry_count", 0)
    
    question_data = VERIFICATION_QUESTIONS[current_idx]
    question_text = question_data["question"]
    
    if retry_count > 0:
        message = f"{AIMessages.QUESTION_RETRY.value}\n\n{question_text}"
    else:
        message = f"Question {current_idx + 1} of {len(VERIFICATION_QUESTIONS)}: {question_text}"
    
    print(f"Asking question {current_idx + 1}: {question_text}")
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": f"asking_question_{current_idx}"
    }


def process_question_node(state: AgentState) -> dict:
    """Process question answer - validation AND state updates."""
    print("\n==== Process Question Node ===")
    
    current_idx = state.get("current_question_index", 0)
    retry_count = state.get("question_retry_count", 0)
    question_data = VERIFICATION_QUESTIONS[current_idx]
    
    user_answer = interrupt("waiting_for_question_answer")
    print(f"Received answer: {user_answer}")
    
    question_prompt = SystemPrompts.PROCESS_QUESTION.value.format(
        question=question_data["question"]
    )
    question_prompt += f"\n\nValidation hint: {question_data['validation_hint']}"
    
    validation_messages = [
        SystemMessage(content=question_prompt),
        HumanMessage(content=user_answer),
    ]
    
    llm_output = llm.with_structured_output(QuestionResult).invoke(validation_messages)
    print(f"DEBUG - {llm_output}")

    # Determine if answer is acceptable
    answer_acceptable = (
        llm_output.confidence > 60 and 
        not llm_output.should_retry and 
        llm_output.is_complete
    )
    
    # Prepare state updates
    answers = state.get("questions_answers", {}).copy()
    new_question_index = current_idx
    new_retry_count = retry_count
    
    if answer_acceptable:
        # Save answer and move to next question
        answers[question_data["id"]] = {
            "question": question_data["question"],
            "answer": llm_output.answer,
            "confidence": llm_output.confidence
        }
        new_question_index = current_idx + 1
        new_retry_count = 0
        print(f"✓ Answer accepted, moving to question {new_question_index + 1}")
    else:
        # Need to retry
        new_retry_count = retry_count + 1
        print(f"✗ Answer needs retry ({new_retry_count}/{MAX_RETRIES})")

    return {
        "messages": [HumanMessage(content=user_answer)],
        "questions_answers": answers,
        "current_question_index": new_question_index,
        "question_retry_count": new_retry_count,
        "current_stage": f"question_{current_idx}_processed"
    }


def respond_to_question_node(state: AgentState) -> dict:
    """Acknowledge question answer."""
    print("\n==== Respond to Question Node ===")
    
    current_idx = state.get("current_question_index", 0)
    
    if current_idx >= len(VERIFICATION_QUESTIONS):
        message = AIMessages.ALL_QUESTIONS_COMPLETE.value
    else:
        message = "Thank you. Next question:"
    
    return {
        "messages": [AIMessage(content=message)],
        "current_stage": f"question_{current_idx}_responded"
    }


# ============================================================================
# ERROR HANDLING
# ============================================================================

def handle_system_error_node(state: AgentState) -> dict:
    """Handle system errors."""
    print("\n==== System Error Handler ===")
    return {
        "messages": [AIMessage(content=AIMessages.SYSTEM_ERROR_FINAL.value)],
        "current_stage": "system_error"
    }


# ============================================================================
# ROUTING FUNCTIONS - NOW CONTAIN THE LOGIC
# ============================================================================

def route_after_consent(state: AgentState) -> str:
    """Route after processing consent - just reads state."""
    consent_given = state.get("consent_given", False)
    retry_count = state.get("consent_retry_count", 0)
    
    if not consent_given and retry_count > 0 and retry_count <= MAX_RETRIES:
        print(f"→ Retrying consent (attempt {retry_count}/{MAX_RETRIES})")
        return "retry"
    elif consent_given:
        print("→ Consent granted, proceeding to authentication")
        return "success"
    else:
        print("→ Consent denied or max retries exceeded")
        return "failure"


def route_after_authentication(state: AgentState) -> str:
    """Route after processing authentication - just reads state."""
    authenticated = state.get("authenticated", False)
    attempts = state.get("auth_attempts", 0)
    
    if not authenticated and attempts > 0 and attempts <= MAX_RETRIES:
        print(f"→ Retrying authentication (attempt {attempts}/{MAX_RETRIES})")
        return "retry"
    elif authenticated:
        print("→ Authentication successful, proceeding to questions")
        return "success"
    else:
        print("→ Authentication failed or max retries exceeded")
        return "failure"


def route_after_question(state: AgentState) -> str:
    """Route after processing question - just reads state to decide."""
    current_idx = state.get("current_question_index", 0)
    retry_count = state.get("question_retry_count", 0)
    answers = state.get("questions_answers", {})
    
    # Get the previous question index (before any update)
    prev_idx = current_idx - 1 if current_idx > 0 else current_idx
    question_data = VERIFICATION_QUESTIONS[prev_idx] if prev_idx < len(VERIFICATION_QUESTIONS) else None
    
    # Check if we just answered a question successfully
    if question_data and question_data["id"] in answers:
        # Question was answered, check if there are more questions
        if current_idx >= len(VERIFICATION_QUESTIONS):
            print(f"→ All questions complete")
            return "complete"
        print(f"→ Moving to next question ({current_idx + 1}/{len(VERIFICATION_QUESTIONS)})")
        return "next_question"
    
    # Check if we need to retry
    if retry_count > 0 and retry_count <= MAX_RETRIES:
        print(f"→ Retrying question (attempt {retry_count}/{MAX_RETRIES})")
        return "retry"
    
    # Max retries exceeded
    print(f"→ Max retries exceeded, ending")
    return "complete"


def route_after_question_response(state: AgentState) -> str:
    """Route after responding to question."""
    current_idx = state.get("current_question_index", 0)
    
    if current_idx >= len(VERIFICATION_QUESTIONS):
        return "end"
    
    return "continue"


def route_after_consent_response(state: AgentState) -> str:
    """Route after consent response."""
    return "end" if not state.get("consent_given") else "continue"


def route_after_auth_response(state: AgentState) -> str:
    """Route after authentication response."""
    return "end" if not state.get("authenticated") else "continue"


# ============================================================================
# BUILD GRAPH
# ============================================================================

workflow = StateGraph(AgentState)

system_retry_policy = RetryPolicy(
    max_attempts=SYSTEM_RETRIES,
    backoff_factor=1.0,
    retry_on=ValidationError,
)

# Add nodes
workflow.add_node("ask_consent", ask_consent_node)
workflow.add_node("process_consent", process_consent_node, 
                  retry_policy=system_retry_policy, fallback_to="handle_system_error")
workflow.add_node("respond_to_consent", respond_to_consent_node)

workflow.add_node("ask_authentication", ask_authentication_node)
workflow.add_node("process_authentication", process_authentication_node,
                  retry_policy=system_retry_policy, fallback_to="handle_system_error")
workflow.add_node("respond_to_authentication", respond_to_authentication_node)

workflow.add_node("ask_question", ask_question_node)
workflow.add_node("process_question", process_question_node,
                  retry_policy=system_retry_policy, fallback_to="handle_system_error")
workflow.add_node("respond_to_question", respond_to_question_node)

workflow.add_node("handle_system_error", handle_system_error_node)

# Consent flow
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

# Authentication flow
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

# Questions flow
workflow.add_edge("ask_question", "process_question")
workflow.add_conditional_edges(
    "process_question",
    route_after_question,
    {
        "retry": "ask_question",
        "next_question": "respond_to_question",
        "complete": "respond_to_question",
    },
)
workflow.add_conditional_edges(
    "respond_to_question",
    route_after_question_response,
    {
        "continue": "ask_question",
        "end": END,
    },
)

workflow.add_edge("handle_system_error", END)

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
        "current_stage": "init",
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

    last_message_count = 0

    for event in app.stream(initial_state, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            current_count = len(event["messages"])
            for i in range(last_message_count, current_count):
                msg = event["messages"][i]
                if isinstance(msg, AIMessage):
                    print(f"\nAI: {msg.content}")
            last_message_count = current_count

    while True:
        state = app.get_state(config)

        if state.next == ():
            print("\n" + "=" * 70)
            print("CONVERSATION COMPLETE")
            print("=" * 70)
            print(f"Current Stage: {state.values.get('current_stage')}")
            print(f"Consent given: {state.values.get('consent_given')}")
            print(f"Authenticated: {state.values.get('authenticated')}")
            if state.values.get("authenticated"):
                print(f"DOB: {state.values.get('user_dob')}")
                print(f"Aadhaar last 4: {state.values.get('user_aadhar_digits')}")
            
            answers = state.values.get("questions_answers", {})
            if answers:
                print("\n--- Collected Answers ---")
                for qid, data in answers.items():
                    print(f"\nQ: {data['question']}")
                    print(f"A: {data['answer']} (confidence: {data['confidence']}%)")
            break

        user_input = input("\nYou: ").strip()

        if not user_input:
            print("Please enter a response.")
            continue

        for event in app.stream(Command(resume=user_input), config, stream_mode="values"):
            if "messages" in event and event["messages"]:
                current_count = len(event["messages"])
                for i in range(last_message_count, current_count):
                    msg = event["messages"][i]
                    if isinstance(msg, AIMessage):
                        print(f"\nAI: {msg.content}")
                last_message_count = current_count


if __name__ == "__main__":
    run_conversation()
