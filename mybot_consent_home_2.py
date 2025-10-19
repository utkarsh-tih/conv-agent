from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, Literal
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from enum import Enum

NODES = Literal[
    "get_consent_node",
    "process_consent_node",
    "get_authentication_node",
    "process_authentication_node",
]
MODEL = "llama3.2:1b"
CHROMADB_DIRECTORY = "./chroma_langchain_db"

"""
Initialisation of LLM
"""
llm = ChatOllama(
    model=MODEL, 
    temperature=0, 
    base_url="http://localhost:11434"
)


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
    POSITIVE_CONSENT_RESPONSE = "Thank you for your consent. Let's proceed with the verification."
    NEGATIVE_CONSENT_RESPONSE = "I understand you did not consent. Goodbye."
    CONSENT_RETRY = "I didn't quite understand. Could you please clearly say 'yes' or 'no'?"
    
    AUTHENTICATION_QUESTION = (
        "For security purposes, please provide:\n"
        "1. Your date of birth (DD/MM/YYYY)\n"
        "2. Last 4 digits of your Aadhar card"
    )
    POSITIVE_AUTHENTICATION_RESPONSE = "Thank you for your authentication. Let's proceed with the verification."
    NEGATIVE_AUTHENTICATION_RESPONSE = "I am sorry we could not authenticate. Goodbye."
    AUTHENTICATION_RETRY = "I couldn't capture all the details. Please provide both your date of birth and last 4 digits of Aadhaar again."


SYSTEM_PROMPT = "You are a calling agent on behalf of credit underwriter, please interpret the answers given by loan applicant and collect relevant data"


class Questions(Enum):
    VERIFICATION_QUESTIONS = {
        1: "Could you please confirm the name of the person I'm speaking with?",
        2: "What documents do you have as proof of your current address? For example, utility bills, rent agreement, etc.",
        3: "What documents do you have for your permanent address proof?",
        4: "For your current address, do you have any ownership proof documents?",
        5: "For your permanent address, do you have ownership proof documents?",
        6: "Tell me about your current residence - who is the owner, how long have you been staying there, and please provide the complete detailed address.",
        7: "What is your office address? Please provide the complete address with landmarks.",
        8: "Regarding your permanent address as per official documents - what is the ownership status, how stable is this address, and please provide the full detailed address.",
        9: "Do you own any other property? If yes, please provide details - what kind of property, proof of ownership, and how it's related to you. Is it a home loan property or a house in a different city?",
        10: "Let me know about your family details - your spouse, father, mother - who is working, their income sources, and any asset details.",
        11: "What is your highest educational qualification? If you're a professional, what is your membership status and which year did you complete your education?",
        12: "What is your current working mode - work from office, work from home, or hybrid?",
        14: "What is your official office email ID?",
        15: "Tell me about your current job - employer name, how long you've been there, your designation, department, are you on third party payroll, working from client location, and any other relevant details.",
        16: "What is your total work experience? Please mention your previous company names and total years of experience.",
        17: "Let's discuss your CIBIL details - your score, credit vintage, existing loans with financier names, rate of interest, EMI amounts, tenure, any recent enquiries, any DPD or overdue amounts, and home loan details if any.",
        18: "Tell me about your banking relationships - all loan and credit card details, your repayment history, any high credits or debits, bounce charges, or return charges.",
        19: "What was the end use for any loans you took in the last 12 months, and what is the current end use for this loan you're applying for?",
        20: "Finally, what is the exact loan amount you require?",
    }


class AgentState(TypedDict):
    """
    State schema using TypedDict for explicit type definitions.
    """
    # Messages with proper reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Flow control - tracks which stage we're at
    stage: Literal["consent", "authenticate", "questions", "complete"]

    # Authentication & Consent
    consent_asked: bool
    consent_given: bool
    consent_confidence: int
    consent_reasoning: str
    consent_retry_count: int
    
    auth_question_asked: bool
    authenticated: bool
    auth_attempts: int
    user_dob: str
    user_aadhar_digits: str

    # Question Flow
    current_question: int
    questions_asked: list[int]

    # Extraction & Retries
    extracted_data: dict
    repeat_count: dict
    extraction_status: str
    should_retry: bool


class ConsentResult(BaseModel):
    """Analysis of user consent from text."""
    consent_given: bool = Field(..., description="Whether consent was given")
    confidence: int = Field(..., description="Confidence level of consent interpretation (0-100)", ge=0, le=100)
    reasoning: str = Field(..., description="Brief explanation of the consent interpretation")
    should_retry: bool = Field(..., description="Whether to retry asking for consent because of ambiguity")


class AuthenticationResult(BaseModel):
    """Analysis of user authentication from text."""
    date_of_birth: str = Field(..., description="User's date of birth in DD/MM/YYYY format")
    aadhaar_last4: str = Field(..., description="Last 4 digits of user's Aadhaar card")
    confidence_date_of_birth: int = Field(..., description="Confidence level of date of birth interpretation (0-100)", ge=0, le=100)
    confidence_aadhaar: int = Field(..., description="Confidence level of Aadhaar interpretation (0-100)", ge=0, le=100)
    reasoning_date_of_birth: str = Field(..., description="Brief explanation of the date of birth interpretation")
    reasoning_aadhaar: str = Field(..., description="Brief explanation of the Aadhaar interpretation")
    should_retry: bool = Field(..., description="Whether to retry asking for authentication because of ambiguity")


# ============================================================================
# NODES
# ============================================================================

def get_consent_node(state: AgentState) -> dict:
    """Ask for consent if not already asked."""
    print("\n==== Get Consent Node ===")
    
    if not state.get("consent_asked", False):
        return {
            "messages": [AIMessage(content=AIMessages.CONSENT_QUESTION.value)],
            "consent_asked": True,
            "stage": "consent"
        }
    
    # If consent already asked, check if retry is needed
    if state.get("should_retry", False) and state.get("consent_retry_count", 0) < 3:
        return {
            "messages": [AIMessage(content=AIMessages.CONSENT_RETRY.value)],
            "consent_retry_count": state.get("consent_retry_count", 0) + 1
        }
    
    return {}


def process_consent_node(state: AgentState) -> dict:
    """Process user's consent response."""
    print("\n==== Process Consent Node ===")

    # Wait for human input (this pauses execution)
    consent_human_input = interrupt("waiting_for_consent")
    print(f"Received input: {consent_human_input}")

    # Create messages for LLM to interpret consent
    consent_check_messages = [
        SystemMessage(content=SystemPrompts.PROCESS_CONSENT.value),
        HumanMessage(content=consent_human_input),
    ]

    # Get structured output (returns ConsentResult object)
    llm_output = llm.with_structured_output(ConsentResult).invoke(consent_check_messages)

    # Access fields directly (it's already a Pydantic object)
    consent_given = llm_output.consent_given
    confidence = llm_output.confidence
    reasoning = llm_output.reasoning
    should_retry = llm_output.should_retry

    print(f"Consent Analysis: given={consent_given}, confidence={confidence}, retry={should_retry}")

    # Determine response message
    if should_retry and state.get("consent_retry_count", 0) < 3:
        ai_response = AIMessage(content=AIMessages.CONSENT_RETRY.value)
        next_stage = "consent"
    elif consent_given:
        ai_response = AIMessage(content=AIMessages.POSITIVE_CONSENT_RESPONSE.value)
        next_stage = "authenticate"
    else:
        ai_response = AIMessage(content=AIMessages.NEGATIVE_CONSENT_RESPONSE.value)
        next_stage = "complete"

    # Return updates (don't mutate state)
    return {
        "messages": [HumanMessage(content=consent_human_input), ai_response],
        "consent_given": consent_given,
        "consent_confidence": confidence,
        "consent_reasoning": reasoning,
        "should_retry": should_retry,
        "stage": next_stage
    }


def get_authentication_node(state: AgentState) -> dict:
    """Ask for authentication details."""
    print("\n==== Get Authentication Node ===")
    
    if not state.get("auth_question_asked", False):
        return {
            "messages": [AIMessage(content=AIMessages.AUTHENTICATION_QUESTION.value)],
            "auth_question_asked": True,
            "stage": "authenticate"
        }
    
    # If authentication already asked, check if retry is needed
    if state.get("should_retry", False) and state.get("auth_attempts", 0) < 3:
        return {
            "messages": [AIMessage(content=AIMessages.AUTHENTICATION_RETRY.value)],
            "auth_attempts": state.get("auth_attempts", 0) + 1
        }
    
    return {}


def process_authentication_node(state: AgentState) -> dict:
    """Process user's authentication response."""
    print("\n==== Process Authentication Node ===")

    # Wait for human input
    auth_human_input = interrupt("waiting_for_authentication")
    print(f"Received input: {auth_human_input}")

    # Create messages for LLM to extract authentication details
    auth_check_messages = [
        SystemMessage(content=SystemPrompts.PROCESS_AUTHENTICATION.value),
        HumanMessage(content=auth_human_input),
    ]

    # Get structured output
    llm_output = llm.with_structured_output(AuthenticationResult).invoke(auth_check_messages)

    # Access fields directly
    date_of_birth = llm_output.date_of_birth
    aadhaar_last4 = llm_output.aadhaar_last4
    confidence_dob = llm_output.confidence_date_of_birth
    confidence_aadhaar = llm_output.confidence_aadhaar
    should_retry = llm_output.should_retry

    print(f"Auth Analysis: DOB={date_of_birth}, Aadhaar={aadhaar_last4}, retry={should_retry}")

    # Determine response
    if should_retry and state.get("auth_attempts", 0) < 3:
        ai_response = AIMessage(content=AIMessages.AUTHENTICATION_RETRY.value)
        authenticated = False
        next_stage = "authenticate"
    elif confidence_dob > 70 and confidence_aadhaar > 70:
        ai_response = AIMessage(content=AIMessages.POSITIVE_AUTHENTICATION_RESPONSE.value)
        authenticated = True
        next_stage = "questions"
    else:
        ai_response = AIMessage(content=AIMessages.NEGATIVE_AUTHENTICATION_RESPONSE.value)
        authenticated = False
        next_stage = "complete"

    return {
        "messages": [HumanMessage(content=auth_human_input), ai_response],
        "authenticated": authenticated,
        "user_dob": date_of_birth,
        "user_aadhar_digits": aadhaar_last4,
        "should_retry": should_retry,
        "stage": next_stage
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def route_after_consent(state: AgentState) -> str:
    """Route based on consent result."""
    if state.get("should_retry", False) and state.get("consent_retry_count", 0) < 3:
        return "get_consent"
    elif state.get("consent_given", False):
        return "get_authentication"
    else:
        return "end"


def route_after_authentication(state: AgentState) -> str:
    """Route based on authentication result."""
    if state.get("should_retry", False) and state.get("auth_attempts", 0) < 3:
        return "get_authentication"
    elif state.get("authenticated", False):
        return "end"  # TODO: Change to questions node when implemented
    else:
        return "end"


# ============================================================================
# BUILD GRAPH
# ============================================================================

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("get_consent", get_consent_node)
workflow.add_node("process_consent", process_consent_node)
workflow.add_node("get_authentication", get_authentication_node)
workflow.add_node("process_authentication", process_authentication_node)

# Add edges
workflow.add_edge(START, "get_consent")
workflow.add_edge("get_consent", "process_consent")
workflow.add_conditional_edges(
    "process_consent",
    route_after_consent,
    {
        "get_consent": "get_consent",
        "get_authentication": "get_authentication",
        "end": END
    }
)
workflow.add_edge("get_authentication", "process_authentication")
workflow.add_conditional_edges(
    "process_authentication",
    route_after_authentication,
    {
        "get_authentication": "get_authentication",
        "end": END
    }
)

# Compile with checkpointer
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# ============================================================================
# EXECUTION
# ============================================================================

def run_conversation():
    """Run the conversation with human-in-the-loop."""
    
    # Initial state
    initial_state = {
        "messages": [],
        "stage": "consent",
        "consent_asked": False,
        "consent_given": False,
        "consent_confidence": 0,
        "consent_reasoning": "",
        "consent_retry_count": 0,
        "auth_question_asked": False,
        "authenticated": False,
        "auth_attempts": 0,
        "user_dob": "",
        "user_aadhar_digits": "",
        "current_question": 0,
        "questions_asked": [],
        "extracted_data": {},
        "repeat_count": {},
        "extraction_status": "",
        "should_retry": False
    }
    
    config = {"configurable": {"thread_id": "conversation-1"}}
    
    print("=" * 70)
    print("STARTING CONVERSATION")
    print("=" * 70)
    
    # Start conversation
    for chunk in app.stream(initial_state, config):
        if "__interrupt__" in chunk:
            # Get current state to see AI's last message
            state = app.get_state(config)
            last_message = state.values["messages"][-1]
            print(f"\nAI: {last_message.content}")
            print(f"[Interrupt: {chunk['__interrupt__'][0].value}]")
            break
    
    # Main conversation loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            print("Please enter a response.")
            continue
        
        # Resume with user input
        result = None
        for chunk in app.stream(Command(resume=user_input), config):
            if "__interrupt__" in chunk:
                # Another interrupt - get state and show AI message
                state = app.get_state(config)
                last_message = state.values["messages"][-1]
                print(f"\nAI: {last_message.content}")
                print(f"[Interrupt: {chunk['__interrupt__'][0].value}]")
                break
            else:
                result = chunk
        
        # Check if conversation is complete
        if result:
            state = app.get_state(config)
            if state.values["stage"] == "complete":
                print("\n" + "=" * 70)
                print("CONVERSATION COMPLETE")
                print("=" * 70)
                print(f"Consent given: {state.values['consent_given']}")
                print(f"Authenticated: {state.values['authenticated']}")
                if state.values['authenticated']:
                    print(f"DOB: {state.values['user_dob']}")
                    print(f"Aadhaar last 4: {state.values['user_aadhar_digits']}")
                break


if __name__ == "__main__":
    run_conversation()
