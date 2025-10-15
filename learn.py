from typing import TypedDict, Annotated
from langgraph.graph import MessagesState

class VerificationState(MessagesState):  # Inherits messages list
    # Authentication & Consent
    consent_given: bool
    authenticated: bool
    auth_attempts: int
    
    # Question Flow
    current_question: str  # e.g., "6"
    questions_asked: list  # e.g., ["1", "2", "6"]
    
    # Extraction & Retries
    extracted_data: dict  # e.g., {"6": {"owner": "Mr. Sharma", ...}}
    retry_count: dict  # e.g., {"6": 2, "7": 0}
    extraction_status: str  # "complete" or "incomplete"
    
    # Compliance
    conversation_transcript: list  # Raw conversation history


def ask_question_node(state):
    # How do you determine next_question_id?
    # Hint: Look at current_question, questions_asked, route decision
    
    next_question_id = state["current_question"]
    question_text = VERIFICATION_QUESTIONS[next_question_id]
    
    # Update state...
    return state