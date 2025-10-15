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
# REDUCER FUNCTIONS
# =============================================================================
def merge_dicts(left: dict, right: dict) -> dict:
    """Merges two dictionaries, with values from the right dictionary overwriting the left."""
    return {**left, **right}

def last_write_wins(old_value, new_value):
    """
    Explicitly defines that the new value should always overwrite the old value.
    This is the default behavior, but making it explicit improves clarity and robustness.
    """
    return new_value


# =============================================================================
# STATE DEFINITION - This is what persists across the conversation
# =============================================================================
class VerificationState(MessagesState):
    """
    State schema for the verification chatbot.
    Using Annotated with reducer functions ensures predictable state updates.
    """

    # Authentication & Consent
    consent_asked: Annotated[bool, last_write_wins]
    consent_given: Annotated[bool, last_write_wins]
    auth_question_asked: Annotated[bool, last_write_wins]
    authenticated: Annotated[bool, last_write_wins]
    auth_attempts: Annotated[int, last_write_wins]
    user_dob: Annotated[str, last_write_wins]
    user_aadhar_digits: Annotated[str, last_write_wins]

    # Question Flow
    current_question: Annotated[int, last_write_wins]
    questions_asked: Annotated[list[int], operator.add]

    # Extraction & Retries
    extracted_data: Annotated[dict, merge_dicts]
    retry_count: Annotated[dict, merge_dicts]
    extraction_status: Annotated[str, last_write_wins]

    # Compliance
    conversation_transcript: Annotated[list, operator.add]


# =============================================================================
# INITIALIZE LLM
# =============================================================================
MODEL = "llama3-chatqa:8b"
llm = ChatOllama(model=MODEL, temperature=0.3, base_url="http://localhost:11434")


# =============================================================================
# NODE FUNCTIONS - Each node now returns a dictionary of ONLY the state it changes
# =============================================================================


def consent_node(state: VerificationState) -> dict:
    print("=== CONSENT NODE CALLED ===")
    messages = state["messages"]

    if not state.get("consent_asked"):
        message = AIMessage(
            content=(
                "Hello! Before we begin the verification process, I need your consent. "
                "This conversation will be recorded for compliance purposes. "
                "Do you agree to proceed? (Please say yes or no)"
            )
        )
        return {
            "messages": [message],
            "consent_asked": True,
            "conversation_transcript": [{"role": "assistant", "content": message.content}],
        }
    else:
        last_message = messages[-1]
        updates = {
            "conversation_transcript": [
                {"role": "user", "content": last_message.content.lower()}
            ]
        }
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            if "yes" in content or "agree" in content or "consent" in content:
                updates["consent_given"] = True
                msg = AIMessage(
                    content="Thank you for your consent. Let's proceed with authentication."
                )
                updates["messages"] = [msg]
                updates["conversation_transcript"].append(
                    {"role": "assistant", "content": msg.content}
                )
            else:
                updates["consent_given"] = False
        print(f"=== CONSENT NODE ENDING, consent_given={updates.get('consent_given')} ===")
        return updates


def authenticate_node(state: VerificationState) -> dict:
    if not state.get("auth_question_asked"):
        message = AIMessage(
            content=(
                "For security purposes, please provide:\n"
                "1. Your date of birth (DD/MM/YYYY)\n"
                "2. Last 4 digits of your Aadhar card"
            )
        )
        return {
            "messages": [message],
            "conversation_transcript": [{"role": "assistant", "content": message.content}],
            "auth_question_asked": True,
            "auth_attempts": 0,
        }
    else:
        last_message = state["messages"][-1].content
        updates = {
            "conversation_transcript": [{"role": "user", "content": last_message}]
        }

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

        if (
            "INCOMPLETE" not in extracted
            and "DOB:" in extracted
            and "AADHAR:" in extracted
        ):
            updates["authenticated"] = True
            updates["user_dob"] = extracted.split("DOB:")[1].split("\n")[0].strip()
            updates["user_aadhar_digits"] = extracted.split("AADHAR:")[1].strip()

            msg = AIMessage(
                content="Authentication successful! Let's begin the verification questions."
            )
            updates["messages"] = [msg]
            updates["conversation_transcript"].append(
                {"role": "assistant", "content": msg.content}
            )

            updates["current_question"] = 1
            updates["questions_asked"] = []
            updates["extracted_data"] = {}
            updates["retry_count"] = {i: 0 for i in VERIFICATION_QUESTIONS.keys()}
        else:
            auth_attempts = state.get("auth_attempts", 0) + 1
            updates["auth_attempts"] = auth_attempts

            if auth_attempts < 3:
                msg = AIMessage(
                    content=f"I couldn't verify your details. Please try again (Attempt {auth_attempts}/3).\nProvide your DOB and last 4 Aadhar digits."
                )
                updates["messages"] = [msg]
                updates["conversation_transcript"].append(
                    {"role": "assistant", "content": msg.content}
                )
        return updates


def ask_question_node(state: VerificationState) -> dict:
    current_q = state["current_question"]
    questions_asked = state.get("questions_asked", [])

    if current_q in questions_asked:
        next_q = current_q + 1
    else:
        next_q = current_q

    updates = {"current_question": next_q}

    if next_q in VERIFICATION_QUESTIONS:
        question_text = VERIFICATION_QUESTIONS[next_q]
        retry_num = state.get("retry_count", {}).get(next_q, 0)

        if retry_num > 0:
            message = AIMessage(
                content=(
                    f"Let me ask that again. {question_text}\n"
                    f"Please provide complete details."
                )
            )
        else:
            message = AIMessage(content=f"Question {next_q}: {question_text}")

        updates["messages"] = [message]
        updates["conversation_transcript"] = [
            {"role": "assistant", "content": message.content}
        ]

    return updates


def extract_answer_node(state: VerificationState) -> dict:
    current_q = state["current_question"]
    last_message = state["messages"][-1].content
    question_text = VERIFICATION_QUESTIONS[current_q]

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

    updates = {
        "conversation_transcript": [{"role": "user", "content": last_message}]
    }

    if "COMPLETE" in extraction_result and "INCOMPLETE" not in extraction_result:
        updates["extraction_status"] = "complete"
        data_part = (
            extraction_result.split("EXTRACTED_DATA:")[1]
            .split("COMPLETENESS:")[0]
            .strip()
        )
        updates["extracted_data"] = {
            current_q: {
                "question": question_text,
                "raw_answer": last_message,
                "extracted": data_part,
            }
        }
    else:
        updates["extraction_status"] = "incomplete"
        updates["extracted_data"] = {
            current_q: {
                "question": question_text,
                "raw_answer": last_message,
                "extracted": "INCOMPLETE",
                "missing": (
                    extraction_result.split("MISSING:")[1].strip()
                    if "MISSING:" in extraction_result
                    else "Details unclear"
                ),
            }
        }
    return updates


def check_retry_node(state: VerificationState) -> dict:
    current_q = state["current_question"]
    extraction_status = state["extraction_status"]

    if extraction_status == "complete":
        return {
            "questions_asked": [current_q],
            "retry_count": {current_q: 0},
        }
    else:
        current_retries = state.get("retry_count", {}).get(current_q, 0)
        return {"retry_count": {current_q: current_retries + 1}}


def exit_node(state: VerificationState) -> dict:
    if not state.get("consent_given"):
        message = AIMessage(content="I understand. Thank you for your time. Goodbye!")
    elif not state.get("authenticated"):
        message = AIMessage(
            content=(
                "I'm sorry, but we couldn't verify your identity after 3 attempts. "
                "Please contact support for assistance. Thank you."
            )
        )
    return {
        "messages": [message],
        "conversation_transcript": [{"role": "assistant", "content": message.content}],
    }


def final_node(state: VerificationState) -> dict:
    message = AIMessage(
        content=(
            "Thank you for completing the verification process! "
            "All your information has been recorded. We'll review your application shortly."
        )
    )
    print("\n=== VERIFICATION COMPLETE ===")
    print(f"Questions answered: {state['questions_asked']}")
    print(f"Total questions: {len(VERIFICATION_QUESTIONS)}")

    return {
        "messages": [message],
        "conversation_transcript": [{"role": "assistant", "content": message.content}],
    }


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS - These determine routing
# =============================================================================


def should_continue_after_consent(
    state: VerificationState,
) -> Literal["authenticate", "exit", "__awaiting_human_input__"]:
    # The check for 'None' is no longer strictly necessary if we initialize properly,
    # but it's safe to keep for robustness.
    if "consent_given" not in state or state["consent_given"] is None:
        return "__awaiting_human_input__"
    elif state["consent_given"]:
        return "authenticate"
    else:
        return "exit"


def should_continue_after_auth(
    state: VerificationState,
) -> Literal["ask_question", "exit", "authenticate", "__awaiting_human_input__"]:
    if "authenticated" not in state or state["authenticated"] is None:
        return "__awaiting_human_input__"
    elif state["authenticated"]:
        return "ask_question"
    elif state.get("auth_attempts", 0) >= 3:
        return "exit"
    else:
        return "authenticate"


def route_after_retry_check(
    state: VerificationState,
) -> Literal["ask_question", "final"]:
    current_q = state["current_question"]
    extraction_status = state["extraction_status"]
    retries = state.get("retry_count", {}).get(current_q, 0)

    # Check if we're done with all questions
    if (
        current_q >= max(VERIFICATION_QUESTIONS.keys())
        and extraction_status == "complete"
    ):
        return "final"

    # If answer complete OR max retries reached, move to next question
    if extraction_status == "complete" or retries >= 3:
        return "ask_question"
    else:
        return "ask_question"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================


def create_verification_graph():
    workflow = StateGraph(VerificationState)

    workflow.add_node("consent", consent_node)
    workflow.add_node("authenticate", authenticate_node)
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("extract_answer", extract_answer_node)
    workflow.add_node("check_retry", check_retry_node)
    workflow.add_node("exit", exit_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "consent")

    workflow.add_conditional_edges(
        "consent",
        should_continue_after_consent,
        {
            "authenticate": "authenticate",
            "exit": "exit",
            "__awaiting_human_input__": END,
        },
    )

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

    workflow.add_edge("ask_question", "extract_answer")
    workflow.add_edge("extract_answer", "check_retry")

    workflow.add_conditional_edges(
        "check_retry",
        route_after_retry_check,
        {"ask_question": "ask_question", "final": "final"},
    )

    workflow.add_edge("exit", END)
    workflow.add_edge("final", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    app = create_verification_graph()
    config = {"configurable": {"thread_id": "user_123"}}

    # Initialize state with type-correct default values
    # Note: Using 'None' is still a valid way to represent an 'unset' state,
    # and the conditional logic handles it.
    initial_state = {
        "messages": [],
        "consent_asked": False,
        "consent_given": None,
        "auth_question_asked": False,
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

    # The first invoke can be empty, as the graph starts from the START node
    result = app.invoke({}, config)
    print(f"Bot: {result['messages'][-1].content}\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)

        if result["messages"]:
            print(f"Bot: {result['messages'][-1].content}\n")

        # A more robust check for completion
        if "final" in result.get("__end__", []):
             break