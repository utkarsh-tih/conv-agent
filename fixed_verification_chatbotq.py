"""
LangGraph Verification Chatbot - Fixed Implementation
This chatbot asks verification questions sequentially with retry logic
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from typing import Annotated

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
# STATE DEFINITION - Using TypedDict for better type safety
# =============================================================================
class VerificationState(TypedDict):
    """
    State schema using TypedDict for explicit type definitions.
    """
    # Messages with proper reducer
    messages: Annotated[list, add_messages]
    
    # Flow control - tracks which stage we're at
    stage: str  # "consent", "authenticate", "questions", "awaiting_answer", "complete"
    
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
    question_just_asked: bool  # NEW: Track if we just asked a question
    questions_asked: list[int]

    # Extraction & Retries
    extracted_data: dict
    retry_count: dict
    extraction_status: str

    # Compliance
    conversation_transcript: list


# =============================================================================
# INITIALIZE LLM
# =============================================================================
MODEL = "llama3-chatqa:8b"
llm = ChatOllama(model=MODEL, temperature=0.3, base_url="http://localhost:11434")


# =============================================================================
# NODE FUNCTIONS - Each node performs a specific task
# =============================================================================


def consent_node(state: VerificationState) -> dict:
    """Handle consent asking and processing"""
    print("=== CONSENT NODE CALLED ===")
    messages = state["messages"]
    print(f"Message count: {len(messages)}")
    print(f"consent_asked: {state.get('consent_asked')}")

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
        
        return {
            "messages": [message],
            "consent_asked": True,
            "conversation_transcript": [{"role": "assistant", "content": message.content}]
        }

    else:
        # SECOND TIME: Process the user's response
        last_message = messages[-1]

        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            transcript_update = [{"role": "user", "content": content}]

            if "yes" in content or "agree" in content or "consent" in content:
                msg = AIMessage(
                    content="Thank you for your consent. Let's proceed with authentication."
                )
                transcript_update.append({"role": "assistant", "content": msg.content})
                
                return {
                    "messages": [msg],
                    "consent_given": True,
                    "stage": "authenticate",
                    "conversation_transcript": transcript_update
                }
            else:
                return {
                    "consent_given": False,
                    "stage": "exit",
                    "conversation_transcript": transcript_update
                }
    
    return {}


def authenticate_node(state: VerificationState) -> dict:
    """Handle authentication asking and processing"""
    print("=== AUTHENTICATE NODE CALLED ===")
    print(f"auth_question_asked: {state.get('auth_question_asked')}")
    
    if not state.get("auth_question_asked", False):
        # FIRST TIME: Ask the authentication question
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
            "auth_attempts": 0
        }

    else:
        # SECOND+ TIME: Process the user's response
        last_message = state["messages"][-1].content
        transcript_update = [{"role": "user", "content": last_message}]

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
            dob = extracted.split("DOB:")[1].split("\n")[0].strip()
            aadhar = extracted.split("AADHAR:")[1].strip()
            
            msg = AIMessage(
                content="Authentication successful! Let's begin the verification questions."
            )
            transcript_update.append({"role": "assistant", "content": msg.content})

            return {
                "messages": [msg],
                "authenticated": True,
                "stage": "questions",  # Move to questions stage
                "question_just_asked": False,  # Haven't asked a question yet
                "user_dob": dob,
                "user_aadhar_digits": aadhar,
                "conversation_transcript": transcript_update,
                "current_question": 1,
                "questions_asked": [],
                "extracted_data": {},
                "retry_count": {i: 0 for i in VERIFICATION_QUESTIONS.keys()}
            }
        else:
            # Authentication failed
            auth_attempts = state.get("auth_attempts", 0) + 1

            if auth_attempts < 3:
                msg = AIMessage(
                    content=f"I couldn't verify your details. Please try again (Attempt {auth_attempts}/3).\nProvide your DOB and last 4 Aadhar digits."
                )
                transcript_update.append({"role": "assistant", "content": msg.content})
                
                return {
                    "messages": [msg],
                    "conversation_transcript": transcript_update,
                    "auth_attempts": auth_attempts,
                    "auth_question_asked": False  # Reset to ask again
                }
            else:
                return {
                    "authenticated": False,
                    "stage": "exit",
                    "auth_attempts": auth_attempts,
                    "conversation_transcript": transcript_update
                }

    return {}


def ask_question_node(state: VerificationState) -> dict:
    """Ask the current question (first time or retry)"""
    print("=== ASK QUESTION NODE CALLED ===")
    current_q = state["current_question"]
    questions_asked = state["questions_asked"]
    question_just_asked = state.get("question_just_asked", False)

    print(f"Current question: {current_q}")
    print(f"Question just asked: {question_just_asked}")
    print(f"Questions answered: {questions_asked}")

    # If we just asked this question, don't ask it again
    if question_just_asked:
        print("Question was just asked, waiting for answer...")
        return {"stage": "awaiting_answer"}

    # Determine which question to ask
    if current_q in questions_asked:
        # Question was answered, move to next
        next_q = current_q + 1
    else:
        # Ask current question (first time or retry)
        next_q = current_q

    # Check if we're done with all questions
    if next_q > max(VERIFICATION_QUESTIONS.keys()):
        return {"stage": "complete"}

    # Get question text
    if next_q in VERIFICATION_QUESTIONS:
        question_text = VERIFICATION_QUESTIONS[next_q]
        retry_num = state["retry_count"].get(next_q, 0)

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

        return {
            "messages": [message],
            "conversation_transcript": [{"role": "assistant", "content": message.content}],
            "current_question": next_q,
            "question_just_asked": True,  # Mark that we just asked
            "stage": "awaiting_answer"
        }

    return {}


def extract_answer_node(state: VerificationState) -> dict:
    """Extract structured information from user's answer using LLM"""
    print("=== EXTRACT ANSWER NODE CALLED ===")
    current_q = state["current_question"]
    last_message = state["messages"][-1].content
    
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
        data_part = (
            extraction_result.split("EXTRACTED_DATA:")[1]
            .split("COMPLETENESS:")[0]
            .strip()
        )
        extracted_data = state["extracted_data"].copy()
        extracted_data[current_q] = {
            "question": question_text,
            "raw_answer": last_message,
            "extracted": data_part,
        }
        
        return {
            "extraction_status": "complete",
            "extracted_data": extracted_data,
            "conversation_transcript": [{"role": "user", "content": last_message}]
        }
    else:
        extracted_data = state["extracted_data"].copy()
        extracted_data[current_q] = {
            "question": question_text,
            "raw_answer": last_message,
            "extracted": "INCOMPLETE",
            "missing": (
                extraction_result.split("MISSING:")[1].strip()
                if "MISSING:" in extraction_result
                else "Details unclear"
            ),
        }
        
        return {
            "extraction_status": "incomplete",
            "extracted_data": extracted_data,
            "conversation_transcript": [{"role": "user", "content": last_message}]
        }


def check_retry_node(state: VerificationState) -> dict:
    """Check if answer was complete and update retry count"""
    print("=== CHECK RETRY NODE CALLED ===")
    current_q = state["current_question"]
    extraction_status = state["extraction_status"]

    if extraction_status == "complete":
        # Answer was good!
        questions_asked = state["questions_asked"].copy()
        questions_asked.append(current_q)
        retry_count = state["retry_count"].copy()
        retry_count[current_q] = 0
        
        return {
            "questions_asked": questions_asked,
            "retry_count": retry_count,
            "question_just_asked": False,  # Reset so we can ask next question
            "stage": "questions"
        }
    else:
        # Answer was unclear, increment retry
        retry_count = state["retry_count"].copy()
        retry_count[current_q] = retry_count.get(current_q, 0) + 1
        
        # If max retries reached, move on
        if retry_count[current_q] >= 3:
            questions_asked = state["questions_asked"].copy()
            questions_asked.append(current_q)
            return {
                "questions_asked": questions_asked,
                "retry_count": retry_count,
                "question_just_asked": False,
                "stage": "questions"
            }
        
        return {
            "retry_count": retry_count,
            "question_just_asked": False,  # Reset so we can retry
            "stage": "questions"
        }


def exit_node(state: VerificationState) -> dict:
    """Exit gracefully when consent denied or auth fails"""
    print("=== EXIT NODE CALLED ===")
    if not state.get("consent_given", False):
        message = AIMessage(content="I understand. Thank you for your time. Goodbye!")
    elif not state.get("authenticated", False):
        message = AIMessage(
            content=(
                "I'm sorry, but we couldn't verify your identity after 3 attempts. "
                "Please contact support for assistance. Thank you."
            )
        )
    else:
        message = AIMessage(content="Thank you for your time. Goodbye!")

    return {
        "messages": [message],
        "stage": "complete",
        "conversation_transcript": [{"role": "assistant", "content": message.content}]
    }


def final_node(state: VerificationState) -> dict:
    """Wrap up after all questions are asked"""
    print("=== FINAL NODE CALLED ===")
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
        "stage": "complete",
        "conversation_transcript": [{"role": "assistant", "content": message.content}]
    }


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS - These determine routing
# =============================================================================

def route_based_on_stage(
    state: VerificationState,
) -> Literal["consent", "authenticate", "ask_question", "extract_answer", "exit", "final"]:
    """
    Main router that determines next node based on current stage.
    """
    stage = state.get("stage", "consent")
    
    print(f"=== ROUTING BASED ON STAGE: {stage} ===")
    
    if stage == "consent":
        # Check if consent was given
        consent_given = state.get("consent_given")
        if consent_given is None:
            return "consent"
        elif consent_given:
            return "authenticate"
        else:
            return "exit"
    
    elif stage == "authenticate":
        # Check authentication status
        authenticated = state.get("authenticated")
        auth_attempts = state.get("auth_attempts", 0)
        
        if authenticated is None:
            return "authenticate"
        elif authenticated:
            return "ask_question"
        elif auth_attempts >= 3:
            return "exit"
        else:
            return "authenticate"
    
    elif stage == "questions":
        # We're in question flow - ask next question
        return "ask_question"
    
    elif stage == "awaiting_answer":
        # We just asked a question, now extract the answer
        messages = state["messages"]
        # Check if the last message is from the user
        if messages and isinstance(messages[-1], HumanMessage):
            return "extract_answer"
        else:
            # Still waiting for user response
            return "exit"  # This will end the turn
    
    elif stage == "exit":
        return "exit"
    
    elif stage == "complete":
        return "final"
    
    else:
        # Default to consent
        return "consent"


def route_after_retry_check(
    state: VerificationState,
) -> Literal["ask_question", "final"]:
    """Route after retry check"""
    current_q = state["current_question"]
    questions_asked = state["questions_asked"]

    print(f"=== ROUTING AFTER RETRY CHECK ===")
    print(f"Current Q: {current_q}, Questions answered: {questions_asked}")

    # Check if we're done with all questions
    if current_q in questions_asked and current_q >= max(VERIFICATION_QUESTIONS.keys()):
        print("DEBUG: All questions complete, going to final")
        return "final"

    # Continue asking questions
    print("DEBUG: Returning ask_question")
    return "ask_question"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================


def create_verification_graph():
    """Build the verification graph with proper flow"""
    
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

    # Start with stage-based routing
    workflow.add_conditional_edges(
        START,
        route_based_on_stage,
        {
            "consent": "consent",
            "authenticate": "authenticate",
            "ask_question": "ask_question",
            "extract_answer": "extract_answer",
            "exit": "exit",
            "final": "final",
        },
    )

    # After consent, go back to START for stage-based routing
    workflow.add_edge("consent", END)

    # After authentication, go back to START for stage-based routing
    workflow.add_edge("authenticate", END)

    # After asking question, go back to START (which will route to extract or wait)
    workflow.add_edge("ask_question", END)

    # After extraction, check retry logic
    workflow.add_edge("extract_answer", "check_retry")

    # After retry check, decide next step
    workflow.add_conditional_edges(
        "check_retry",
        route_after_retry_check,
        {
            "ask_question": "ask_question",
            "final": "final"
        },
    )

    # Terminal nodes
    workflow.add_edge("exit", END)
    workflow.add_edge("final", END)

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """Run the verification chatbot"""
    
    # Create the graph
    app = create_verification_graph()

    # Conversation configuration
    config = {"configurable": {"thread_id": "user_123"}}

    # Initialize state
    initial_state = {
        "messages": [],
        "stage": "consent",
        "consent_asked": False,
        "consent_given": None,
        "auth_question_asked": False,
        "authenticated": None,
        "auth_attempts": 0,
        "user_dob": "",
        "user_aadhar_digits": "",
        "current_question": 0,
        "question_just_asked": False,
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
            last_msg = result['messages'][-1]
            if isinstance(last_msg, AIMessage):
                print(f"Bot: {last_msg.content}\n")

        # Check if conversation ended naturally
        stage = result.get("stage", "")
        if stage == "complete":
            break