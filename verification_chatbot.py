"""
LangGraph Verification Chatbot - Educational Implementation
This chatbot asks verification questions sequentially with retry logic
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import operator

# Your questions dictionary
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


# =============================================================================
# STATE DEFINITION - This is what persists across the conversation
# =============================================================================
class VerificationState(MessagesState):
    """
    State schema for the verification chatbot.
    MessagesState provides a 'messages' list automatically.

    LANGGRAPH CONCEPT: State is a typed dictionary that flows through nodes.
    Each node can read from and write to the state.
    """

    # Authentication & Consent
    consent_asked: bool
    consent_given: bool
    auth_question_asked: bool
    authenticated: bool
    auth_attempts: int
    user_dob: str
    user_aadhar_digits: str

    # Question Flow
    current_question: int  # Using int as you suggested!
    questions_asked: list[int]

    # Extraction & Retries
    extracted_data: dict
    retry_count: dict
    extraction_status: str

    # Compliance
    conversation_transcript: Annotated[list, operator.add]  # Append-only list


# =============================================================================
# INITIALIZE LLM
# =============================================================================
# Replace with your LLM - could be ChatAnthropic, ChatOpenAI, etc.

MODEL = "llama3-chatqa:8b"
llm = ChatOllama(model=MODEL, temperature=0.3, base_url="http://localhost:11434")


# =============================================================================
# NODE FUNCTIONS - Each node performs a specific task
# =============================================================================


def consent_node(state: VerificationState) -> VerificationState:
    """
    This node handles BOTH asking and processing
    """
    print("=== CONSENT NODE CALLED ===")
    messages = state["messages"]
    print(f"Message count: {len(messages)}")
    print(f"consent_asked: {state.get('consent_asked')}")
    messages = state["messages"]

    # Check if we've already asked the consent question
    if not state.get("consent_asked", False):
        # FIRST TIME: Ask the question
        message = AIMessage(
            content=(
                "Hello! Before we begin the verification process, I need your consent. "
                "This conversation will be recorded for compliance purposes. "
                "Do you agree to proceed? (Please say yes or no)"
            )
        )
        state["messages"].append(message)
        state["consent_asked"] = True
        state["conversation_transcript"] = [
            {"role": "assistant", "content": message.content}
        ]

    else:
        # SECOND TIME: Process the user's response
        last_message = messages[-1]

        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            state["conversation_transcript"].append(
                {"role": "user", "content": content}
            )

            if "yes" in content or "agree" in content or "consent" in content:
                state["consent_given"] = True
                msg = AIMessage(
                    content="Thank you for your consent. Let's proceed with authentication."
                )
                state["messages"].append(msg)
                state["conversation_transcript"].append(
                    {"role": "assistant", "content": msg.content}
                )
            else:
                state["consent_given"] = False
    print(f"=== CONSENT NODE ENDING, consent_given={state.get('consent_given')} ===")
    return state


def authenticate_node(state: VerificationState) -> VerificationState:
    if not state.get("auth_question_asked", False):
        # FIRST TIME: Ask the authentication question
        message = AIMessage(
            content=(
                "For security purposes, please provide:\n"
                "1. Your date of birth (DD/MM/YYYY)\n"
                "2. Last 4 digits of your Aadhar card"
            )
        )
        state["messages"].append(message)
        state["conversation_transcript"].append(
            {"role": "assistant", "content": message.content}
        )
        state["auth_question_asked"] = True
        state["auth_attempts"] = 0  # Initialize

    else:
        # SECOND+ TIME: Process the user's response
        last_message = state["messages"][-1].content
        state["conversation_transcript"].append(
            {"role": "user", "content": last_message}
        )

        # Extract DOB and Aadhar using LLM
        extraction_prompt = f"""
        Extract the date of birth and last 4 digits of Aadhar from this message:
        "{last_message}"
        
        Respond in this exact format:
        DOB: DD/MM/YYYY
        AADHAR: XXXX
        
        If information is missing, say "INCOMPLETE"
        """

        response = llm.invoke([SystemMessage(content=extraction_prompt)])
        extracted = response.content

        # Validate
        if (
            "INCOMPLETE" not in extracted
            and "DOB:" in extracted
            and "AADHAR:" in extracted
        ):
            state["authenticated"] = True
            state["user_dob"] = extracted.split("DOB:")[1].split("\n")[0].strip()
            state["user_aadhar_digits"] = extracted.split("AADHAR:")[1].strip()

            msg = AIMessage(
                content="Authentication successful! Let's begin the verification questions."
            )
            state["messages"].append(msg)
            state["conversation_transcript"].append(
                {"role": "assistant", "content": msg.content}
            )

            # Initialize question tracking
            state["current_question"] = 1
            state["questions_asked"] = []
            state["extracted_data"] = {}
            state["retry_count"] = {i: 0 for i in VERIFICATION_QUESTIONS.keys()}

        else:
            # Authentication failed
            state["auth_attempts"] += 1

            if state["auth_attempts"] < 3:
                msg = AIMessage(
                    content=f"I couldn't verify your details. Please try again (Attempt {state['auth_attempts']}/3).\nProvide your DOB and last 4 Aadhar digits."
                )
                state["messages"].append(msg)
                state["conversation_transcript"].append(
                    {"role": "assistant", "content": msg.content}
                )

    return state


def ask_question_node(state: VerificationState) -> VerificationState:
    """
    NODE 3: Ask the current question (first time or retry)

    LANGGRAPH CONCEPT: This node uses state to determine behavior.
    It checks if current question was already asked to decide next action.
    """
    current_q = state["current_question"]
    questions_asked = state["questions_asked"]

    # Determine which question to ask
    if current_q in questions_asked:
        # Question was answered, move to next
        next_q = current_q + 1
    else:
        # Ask current question (first time or retry)
        next_q = current_q

    state["current_question"] = next_q

    # Get question text
    if next_q in VERIFICATION_QUESTIONS:
        question_text = VERIFICATION_QUESTIONS[next_q]
        retry_num = state["retry_count"][next_q]

        if retry_num > 0:
            # This is a retry
            message = AIMessage(
                content=(
                    f"Let me ask that again. {question_text}\n"
                    f"Please provide complete details."
                )
            )
        else:
            message = AIMessage(content=f"Question {next_q}: {question_text}")

        state["messages"].append(message)
        state["conversation_transcript"].append(
            {"role": "assistant", "content": message.content}
        )

    return state


def extract_answer_node(state: VerificationState) -> VerificationState:
    """
    NODE 4: Extract structured information from user's answer using LLM

    LANGGRAPH CONCEPT: This is where AI does the heavy lifting.
    The LLM interprets natural language and extracts structured data.
    """
    current_q = state["current_question"]
    last_message = state["messages"][-1].content
    state["conversation_transcript"].append({"role": "user", "content": last_message})

    question_text = VERIFICATION_QUESTIONS[current_q]

    # Use LLM to extract relevant information
    extraction_prompt = f"""
    Question asked: {question_text}
    User's response: {last_message}
    
    Extract all relevant information from the user's response that answers the question.
    Be thorough - extract names, numbers, addresses, dates, documents mentioned, etc.
    
    Respond in this format:
    EXTRACTED_DATA: [detailed information]
    COMPLETENESS: [COMPLETE or INCOMPLETE]
    MISSING: [what's missing if incomplete]
    """

    response = llm.invoke([SystemMessage(content=extraction_prompt)])
    extraction_result = response.content

    # Parse extraction result
    if "COMPLETE" in extraction_result and "INCOMPLETE" not in extraction_result:
        state["extraction_status"] = "complete"
        # Store extracted data
        data_part = (
            extraction_result.split("EXTRACTED_DATA:")[1]
            .split("COMPLETENESS:")[0]
            .strip()
        )
        state["extracted_data"][current_q] = {
            "question": question_text,
            "raw_answer": last_message,
            "extracted": data_part,
        }
    else:
        state["extraction_status"] = "incomplete"
        state["extracted_data"][current_q] = {
            "question": question_text,
            "raw_answer": last_message,
            "extracted": "INCOMPLETE",
            "missing": (
                extraction_result.split("MISSING:")[1].strip()
                if "MISSING:" in extraction_result
                else "Details unclear"
            ),
        }

    return state


def check_retry_node(state: VerificationState) -> VerificationState:
    """
    NODE 5: Check if answer was complete and update retry count

    LANGGRAPH CONCEPT: This node updates state based on extraction results.
    It prepares state for conditional routing.
    """
    current_q = state["current_question"]
    extraction_status = state["extraction_status"]

    if extraction_status == "complete":
        # Answer was good!
        state["questions_asked"].append(current_q)
        state["retry_count"][current_q] = 0  # Reset
    else:
        # Answer was unclear, increment retry
        state["retry_count"][current_q] += 1

    return state


def exit_node(state: VerificationState) -> VerificationState:
    """
    NODE 6: Exit gracefully when consent denied or auth fails
    """
    if not state.get("consent_given", False):
        message = AIMessage(content="I understand. Thank you for your time. Goodbye!")
    elif not state.get("authenticated", False):
        message = AIMessage(
            content=(
                "I'm sorry, but we couldn't verify your identity after 3 attempts. "
                "Please contact support for assistance. Thank you."
            )
        )

    state["messages"].append(message)
    state["conversation_transcript"].append(
        {"role": "assistant", "content": message.content}
    )
    return state


def final_node(state: VerificationState) -> VerificationState:
    """
    NODE 7: Wrap up after all questions are asked
    """
    message = AIMessage(
        content=(
            "Thank you for completing the verification process! "
            "All your information has been recorded. We'll review your application shortly."
        )
    )
    state["messages"].append(message)
    state["conversation_transcript"].append(
        {"role": "assistant", "content": message.content}
    )

    # Here you could save the transcript and extracted_data to database
    print("\n=== VERIFICATION COMPLETE ===")
    print(f"Questions answered: {state['questions_asked']}")
    print(f"Total questions: {len(VERIFICATION_QUESTIONS)}")

    return state


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS - These determine routing
# =============================================================================

def should_continue_after_consent(
    state: VerificationState,
) -> Literal["authenticate", "exit", "__awaiting_human_input__"]:
    consent_given = state.get("consent_given")

    print(f"=== should_continue_after_CONSENT CALLED ===")  # Fixed message
    print(f"DEBUG: consent_given = {consent_given}")

    if consent_given is None:
        print("DEBUG: Returning __awaiting_human_input__")
        return "__awaiting_human_input__"
    elif consent_given == True:
        print("DEBUG: Returning authenticate")
        return "authenticate"
    else:
        print("DEBUG: Returning exit")
        return "exit"


def should_continue_after_auth(
    state: VerificationState,
) -> Literal["ask_question", "exit", "authenticate", "__awaiting_human_input__"]:
    # Has the authentication been processed yet?
    authenticated = state.get("authenticated")
    auth_attempts = state.get("auth_attempts", 0)

    if authenticated is None:
        # Question was asked but no response processed yet
        return "__awaiting_human_input__"

    elif authenticated == True:
        # Success! Move to questions
        return "ask_question"

    elif auth_attempts >= 3:
        # Failed 3 times, exit
        return "exit"

    else:
        # Failed but can retry (attempts 1-2)
        return "authenticate"  # Loop back to ask again


def route_after_retry_check(
    state: VerificationState,
) -> Literal["ask_question", "final"]:
    """
    LANGGRAPH CONCEPT: This conditional edge implements your retry logic.
    It checks state and decides whether to retry, move forward, or end.
    """
    current_q = state["current_question"]
    extraction_status = state["extraction_status"]
    retries = state["retry_count"][current_q]

    # Check if we're done with all questions
    if (
        current_q >= max(VERIFICATION_QUESTIONS.keys())
        and extraction_status == "complete"
    ):
        return "final"

    # If answer complete OR max retries reached, move to next question
    if extraction_status == "complete" or retries >= 3:
        return "ask_question"  # Will move to next question
    else:
        return "ask_question"  # Will retry same question


# =============================================================================
# BUILD THE GRAPH
# =============================================================================


def create_verification_graph():
    """
    LANGGRAPH CONCEPT: This is where you define the entire flow.
    - StateGraph: Container for your nodes and edges
    - add_node: Register node functions
    - add_edge: Unconditional transition
    - add_conditional_edges: Routing based on state
    """

    # Create the graph
    workflow = StateGraph(VerificationState)

    # Add all nodes
    workflow.add_node("consent", consent_node)
    workflow.add_node("authenticate", authenticate_node)
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("extract_answer", extract_answer_node)
    workflow.add_node("check_retry", check_retry_node)
    workflow.add_node("exit", exit_node)
    workflow.add_node("final", final_node)

    # Define the flow with edges
    workflow.add_edge(START, "consent")

    # After consent, conditionally route
    workflow.add_conditional_edges(
        "consent",
        should_continue_after_consent,
        {
            "authenticate": "authenticate",
            "exit": "exit",
            "__awaiting_human_input__": END,  # Pause here!
        },
    )

    # After authentication, conditionally route
    workflow.add_conditional_edges(
        "authenticate",
        should_continue_after_auth,
        {
            "ask_question": "ask_question",
            "authenticate": "authenticate",
            "exit": "exit",
            "__awaiting_human_input__": END,
        },
    )

    # After asking question, wait for user response then extract
    workflow.add_edge("ask_question", "extract_answer")

    # After extraction, check retry logic
    workflow.add_edge("extract_answer", "check_retry")

    # After retry check, decide next step
    workflow.add_conditional_edges(
        "check_retry",
        route_after_retry_check,
        {"ask_question": "ask_question", "final": "final"},
    )

    # Terminal nodes
    workflow.add_edge("exit", END)
    workflow.add_edge("final", END)

    # Compile with memory
    # LANGGRAPH CONCEPT: MemorySaver enables conversation persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    LANGGRAPH CONCEPT: Conversations are threaded.
    Each thread_id represents a unique conversation session.
    """

    # Create the graph
    app = create_verification_graph()

    # Conversation configuration
    config = {"configurable": {"thread_id": "user_123"}}

    # Initialize state
    initial_state = {
        "messages": [],
        "consent_given": None,
        "authenticated": None,
        "auth_attempts": 0,
        "user_dob": "",
        "user_aadhar_digits": "",
        "current_question": 0,
        "questions_asked": [],
        "extracted_data": {},
        "retry_count": {},
        "extraction_status": "",
        "conversation_transcript": [],
    }

    print("=== Verification Chatbot Started ===\n")
    print("Instructions:")
    print("1. You'll be asked for consent")
    print("2. Provide DOB and Aadhar for authentication")
    print("3. Answer verification questions")
    print("4. Type 'quit' to exit\n")

    # Start conversation
    result = app.invoke(initial_state, config)
    print(f"Bot: {result['messages'][-1].content}\n")

    # Conversation loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # Add user message to state
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)

        # Print bot's response
        if result["messages"]:
            print(f"Bot: {result['messages'][-1].content}\n")

        # Check if conversation ended
        if result.get("authenticated") and len(
            result.get("questions_asked", [])
        ) >= len(VERIFICATION_QUESTIONS):
            break
