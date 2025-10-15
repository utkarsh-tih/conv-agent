from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import operator

# Your questions dictionary
VERIFICATION_QUESTIONS = {
    1: "Could you please confirm the name of the person I'm speaking with?",
    2: "What documents do you have as proof of your current address?",
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

class VerificationState(MessagesState):
    # Authentication & Consent
    consent_asked: bool
    consent_given: bool
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
    retry_count: dict
    extraction_status: str

    # Compliance
    conversation_transcript: Annotated[list, operator.add]

MODEL = "llama3-chatqa:8b"
llm = ChatOllama(model=MODEL, temperature=0.3, base_url="http://localhost:11434")

# =============================================================================
# NODES
# =============================================================================

def ask_consent_node(state: VerificationState) -> VerificationState:
    print("=== ASKING CONSENT ===")
    # Only ask if we haven't asked yet AND there are no messages yet
    if not state.get("consent_asked", False) and len(state["messages"]) == 0:
        message = AIMessage(
            content=(
                "Hello! Before we begin the verification process, I need your consent. "
                "This conversation will be recorded for compliance purposes. "
                "Do you agree to proceed? (Please say yes or no)"
            )
        )
        state["messages"].append(message)
        state["conversation_transcript"] = [{"role": "assistant", "content": message.content}]
        state["consent_asked"] = True
    return state

def process_consent_node(state: VerificationState) -> VerificationState:
    print("=== PROCESSING CONSENT ===")
    
    # Process only if we have a human response after asking consent
    if len(state["messages"]) >= 2:  # At least ask + response
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            state["conversation_transcript"].append({"role": "user", "content": content})
            
            if "yes" in content or "agree" in content or "consent" in content:
                state["consent_given"] = True
                msg = AIMessage(content="Thank you for your consent. Let's proceed with authentication.")
            else:
                state["consent_given"] = False
                msg = AIMessage(content="I understand. Thank you for your time. Goodbye!")
            
            state["messages"].append(msg)
            state["conversation_transcript"].append({"role": "assistant", "content": msg.content})
    
    return state

def ask_auth_node(state: VerificationState) -> VerificationState:
    """Only asks auth question"""
    print("=== ASKING AUTH ===")
    message = AIMessage(
        content=(
            "For security purposes, please provide:\n"
            "1. Your date of birth (DD/MM/YYYY)\n"
            "2. Last 4 digits of your Aadhar card"
        )
    )
    state["messages"].append(message)
    state["conversation_transcript"].append({"role": "assistant", "content": message.content})
    state["auth_question_asked"] = True
    state["auth_attempts"] = 0
    return state

def process_auth_node(state: VerificationState) -> VerificationState:
    """Processes auth response"""
    print("=== PROCESSING AUTH ===")
    last_message = state["messages"][-1].content
    state["conversation_transcript"].append({"role": "user", "content": last_message})

    # Simple extraction for demo
    if "1998" in last_message and ("1968" in last_message or "68" in last_message):
        state["authenticated"] = True
        state["user_dob"] = "11/12/1998"
        state["user_aadhar_digits"] = "1968"

        msg = AIMessage(content="Authentication successful! Let's begin the verification questions.")
        state["messages"].append(msg)
        state["conversation_transcript"].append({"role": "assistant", "content": msg.content})

        # Initialize question tracking
        state["current_question"] = 1
        state["questions_asked"] = []
        state["extracted_data"] = {}
        state["retry_count"] = {i: 0 for i in VERIFICATION_QUESTIONS.keys()}
    else:
        state["auth_attempts"] += 1
        if state["auth_attempts"] < 3:
            msg = AIMessage(
                content=f"I couldn't verify your details. Please try again (Attempt {state['auth_attempts']}/3).\nProvide your DOB and last 4 Aadhar digits."
            )
            state["messages"].append(msg)
            state["conversation_transcript"].append({"role": "assistant", "content": msg.content})
        else:
            state["authenticated"] = False

    return state

def ask_question_node(state: VerificationState) -> VerificationState:
    """Asks the current question"""
    print(f"=== ASKING QUESTION {state['current_question']} ===")
    current_q = state["current_question"]
    
    # Check if this question was already answered (completed)
    if current_q in state.get("questions_asked", []):
        # Move to next question
        next_q = current_q + 1
        state["current_question"] = next_q
    else:
        # Ask current question
        next_q = current_q

    if next_q in VERIFICATION_QUESTIONS:
        question_text = VERIFICATION_QUESTIONS[next_q]
        
        # Check if this is a retry
        retry_num = state["retry_count"].get(next_q, 0)
        if retry_num > 0:
            message = AIMessage(
                content=f"Let me ask that again. {question_text}\nPlease provide complete details."
            )
        else:
            message = AIMessage(content=f"Question {next_q}: {question_text}")
        
        state["messages"].append(message)
        state["conversation_transcript"].append({"role": "assistant", "content": message.content})
    
    return state

def process_answer_node(state: VerificationState) -> VerificationState:
    """Processes user's answer"""
    print("=== PROCESSING ANSWER ===")
    current_q = state["current_question"]
    last_message = state["messages"][-1].content
    state["conversation_transcript"].append({"role": "user", "content": last_message})

    # For demo, mark as complete
    state["extraction_status"] = "complete"
    state["extracted_data"][current_q] = {
        "question": VERIFICATION_QUESTIONS[current_q],
        "raw_answer": last_message,
        "extracted": last_message,
    }

    return state

def handle_retry_logic_node(state: VerificationState) -> VerificationState:
    """Updates retry count and question tracking"""
    print("=== HANDLING RETRY LOGIC ===")
    current_q = state["current_question"]
    
    if state["extraction_status"] == "complete":
        # Answer was good!
        state["questions_asked"].append(current_q)
        state["retry_count"][current_q] = 0  # Reset retry count
    else:
        # Answer was incomplete, increment retry
        state["retry_count"][current_q] = state["retry_count"].get(current_q, 0) + 1

    return state

def final_node(state: VerificationState) -> VerificationState:
    """Final success message"""
    print("=== FINAL NODE ===")
    message = AIMessage(
        content="Thank you for completing the verification process! All your information has been recorded."
    )
    state["messages"].append(message)
    state["conversation_transcript"].append({"role": "assistant", "content": message.content})
    return state

def exit_node(state: VerificationState) -> VerificationState:
    """Exit node"""
    print("=== EXIT NODE ===")
    message = AIMessage(content="Session ended.")
    state["messages"].append(message)
    state["conversation_transcript"].append({"role": "assistant", "content": message.content})
    return state

# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def route_after_consent(state: VerificationState) -> Literal["ask_auth", "exit"]:
    print(f"=== ROUTE AFTER CONSENT: consent_given={state.get('consent_given')} ===")
    if state.get("consent_given") == True:
        return "ask_auth"
    else:
        return "exit"

def route_after_auth(state: VerificationState) -> Literal["ask_question", "ask_auth", "exit"]:
    print(f"=== ROUTE AFTER AUTH: authenticated={state.get('authenticated')}, attempts={state.get('auth_attempts')} ===")
    if state.get("authenticated") == True:
        return "ask_question"
    elif state.get("auth_attempts", 0) >= 3:
        return "exit"
    else:
        return "ask_auth"  # Retry auth

def route_after_retry(state: VerificationState) -> Literal["ask_question", "final"]:
    print(f"=== ROUTE AFTER RETRY: current_q={state.get('current_question')}, max={max(VERIFICATION_QUESTIONS.keys())} ===")
    current_q = state["current_question"]
    
    # If all questions done
    if current_q > max(VERIFICATION_QUESTIONS.keys()):
        return "final"
    
    # If max retries reached or question completed, move to next
    retries = state["retry_count"].get(current_q, 0)
    if state["extraction_status"] == "complete" or retries >= 3:
        return "ask_question"
    else:
        return "ask_question"

# =============================================================================
# BUILD GRAPH
# =============================================================================

def create_verification_graph():
    workflow = StateGraph(VerificationState)

    workflow.add_node("ask_consent", ask_consent_node)
    workflow.add_node("process_consent", process_consent_node)
    workflow.add_node("ask_auth", ask_auth_node)
    workflow.add_node("process_auth", process_auth_node)
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("process_answer", process_answer_node)
    workflow.add_node("handle_retry", handle_retry_logic_node)
    workflow.add_node("final", final_node)
    workflow.add_node("exit", exit_node)

    # Simple linear flow with conditional routing
    workflow.add_edge(START, "ask_consent")
    workflow.add_edge("ask_consent", "process_consent")
    workflow.add_conditional_edges(
        "process_consent",
        route_after_consent,
        {"ask_auth": "ask_auth", "exit": "exit"}
    )
    workflow.add_edge("ask_auth", "process_auth")
    workflow.add_conditional_edges(
        "process_auth",
        route_after_auth,
        {"ask_question": "ask_question", "ask_auth": "ask_auth", "exit": "exit"}
    )
    workflow.add_edge("ask_question", "process_answer")
    workflow.add_edge("process_answer", "handle_retry")
    workflow.add_conditional_edges(
        "handle_retry",
        route_after_retry,
        {"ask_question": "ask_question", "final": "final"}
    )
    workflow.add_edge("exit", END)
    workflow.add_edge("final", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
# =============================================================================
# MAIN LOOP
# =============================================================================

if __name__ == "__main__":
    app = create_verification_graph()
    config = {"configurable": {"thread_id": "user_123"}}

    # Start conversation
    print("=== Verification Chatbot Started ===\n")
    
    initial_state = {
        "messages": [],
        "consent_asked": False,
        "consent_given": None,
        "auth_question_asked": False,
        "authenticated": None,
        "auth_attempts": 0,
        "user_dob": "",
        "user_aadhar_digits": "",
        "current_question": 1,
        "questions_asked": [],
        "extracted_data": {},
        "retry_count": {i: 0 for i in VERIFICATION_QUESTIONS.keys()},
        "extraction_status": "",
        "conversation_transcript": [],
    }

    result = app.invoke(initial_state, config)
    print(f"Bot: {result['messages'][-1].content}\n")

    # Conversation loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # This resumes the graph from where it paused!
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )

        # Check if there's a new bot message to display
        if len(result["messages"]) > 0:
            last_msg = result["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print(f"Bot: {last_msg.content}\n")

        # Check if conversation should end
        if (result.get("authenticated") and 
            len(result.get("questions_asked", [])) >= len(VERIFICATION_QUESTIONS)):
            break