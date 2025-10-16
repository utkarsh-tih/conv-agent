from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel, Field
import json

NODES = Literal["get_consent_node", "process_consent_node", "get_authentication_node", "process_authentication_node"]
MODEL = "llama3.2:1b"
CHROMADB_DIRECTORY = "./chroma_langchain_db"

"""
Initialisation of LLM, Embeddings and Vector Store
"""
llm = ChatOllama(model=MODEL, temperature=0.3, base_url="http://localhost:11434")

SYSTEM_PROMPT = "You are a calling agent on behalf of credit underwriter, please interpret the answers given by loan applicant and collect relevant data"

VERIFICATION_QUESTIONS = {
    "1": "Could you please confirm the name of the person I'm speaking with?",
    "2": "What documents do you have as proof of your current address? For example, utility bills, rent agreement, etc.",
    "3": "What documents do you have for your permanent address proof?",
    "4": "For your current address, do you have any ownership proof documents?",
    "5": "For your permanent address, do you have ownership proof documents?",
    "6": "Tell me about your current residence - who is the owner, how long have you been staying there, and please provide the complete detailed address.",
    "7": "What is your office address? Please provide the complete address with landmarks.",
    "8": "Regarding your permanent address as per official documents - what is the ownership status, how stable is this address, and please provide the full detailed address.",
    "9": "Do you own any other property? If yes, please provide details - what kind of property, proof of ownership, and how it's related to you. Is it a home loan property or a house in a different city?",
    "10": "Let me know about your family details - your spouse, father, mother - who is working, their income sources, and any asset details.",
    "11": "What is your highest educational qualification? If you're a professional, what is your membership status and which year did you complete your education?",
    "12": "What is your current working mode - work from office, work from home, or hybrid?",
    "14": "What is your official office email ID?",
    "15": "Tell me about your current job - employer name, how long you've been there, your designation, department, are you on third party payroll, working from client location, and any other relevant details.",
    "16": "What is your total work experience? Please mention your previous company names and total years of experience.",
    "17": "Let's discuss your CIBIL details - your score, credit vintage, existing loans with financier names, rate of interest, EMI amounts, tenure, any recent enquiries, any DPD or overdue amounts, and home loan details if any.",
    "18": "Tell me about your banking relationships - all loan and credit card details, your repayment history, any high credits or debits, bounce charges, or return charges.",
    "19": "What was the end use for any loans you took in the last 12 months, and what is the current end use for this loan you're applying for?",
    "20": "Finally, what is the exact loan amount you require?",
}


# Pydantic Models for Structured Output
class ConsentResponse(BaseModel):
    """Model for parsing consent from user response"""
    consent_given: bool = Field(description="True if user agrees/consents, False if they decline")
    confidence: str = Field(description="high, medium, or low confidence in the interpretation")
    reasoning: str = Field(description="Brief explanation of why this interpretation was made")


class AgentState(TypedDict):
    """
    State schema using TypedDict for explicit type definitions.
    """
    # Messages with proper reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Flow control - tracks which stage we're at
    stage: str  # "consent", "authenticate", "questions", "complete"

    # Authentication & Consent
    consent_asked: bool
    consent_given: bool | None  # None = not determined yet, True = given, False = declined
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

    # Compliance
    conversation_transcript: list


def get_consent_node(state: AgentState) -> AgentState:
    """Ask for user consent if not already asked"""
    print("==== Get Consent Node ====")
    
    if not state.get("consent_asked", False):
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
            "stage": "consent",
            "conversation_transcript": [
                {"role": "assistant", "content": message.content}
            ]
        }
    
    return {}


def process_consent_node(state: AgentState) -> AgentState:
    """Process user's consent response using LLM with structured output"""
    print("==== Process Consent Node ====")
    
    # Get the last user message
    if not state["messages"]:
        return {}
    
    last_message = state["messages"][-1]
    
    # Only process if it's a human message and consent not yet determined
    if not isinstance(last_message, HumanMessage):
        return {}
    
    if state.get("consent_given") is not None:
        # Already processed
        return {}
    
    user_response = last_message.content
    
    # Create LLM with structured output
    structured_llm = llm.with_structured_output(ConsentResponse)
    
    # Prompt for consent interpretation
    consent_prompt = f"""You are analyzing a user's response to a consent request for a recorded verification call.

User's response: "{user_response}"

Determine if the user has given consent or declined. Look for:
- Affirmative responses: yes, yeah, sure, okay, ok, I agree, I consent, go ahead, proceed, etc.
- Negative responses: no, nope, I don't agree, I decline, I don't consent, etc.
- Unclear responses: maybe, I'm not sure, what does that mean, etc.

Be generous with interpretation but accurate. If unclear, mark as low confidence."""

    try:
        # Get structured response from LLM
        consent_analysis: ConsentResponse = structured_llm.invoke(consent_prompt)
        
        print(f"Consent Analysis: {consent_analysis.model_dump()}")
        
        # Update transcript
        new_transcript_entry = {"role": "user", "content": user_response}
        updated_transcript = state.get("conversation_transcript", []) + [new_transcript_entry]
        
        # If confidence is low, ask for clarification
        if consent_analysis.confidence == "low":
            clarify_message = AIMessage(
                content="I didn't quite understand your response. Could you please clearly say 'yes' if you agree to this recorded conversation, or 'no' if you don't?"
            )
            return {
                "messages": [clarify_message],
                "conversation_transcript": updated_transcript + [
                    {"role": "assistant", "content": clarify_message.content}
                ]
            }
        
        # Process based on consent decision
        if consent_analysis.consent_given:
            proceed_message = AIMessage(
                content="Thank you for your consent. Let's proceed with the verification process."
            )
            return {
                "consent_given": True,
                "stage": "authenticate",
                "messages": [proceed_message],
                "conversation_transcript": updated_transcript + [
                    {"role": "assistant", "content": proceed_message.content}
                ]
            }
        else:
            goodbye_message = AIMessage(
                content="I understand. Without your consent, we cannot proceed with the verification. Thank you for your time. Goodbye!"
            )
            return {
                "consent_given": False,
                "stage": "complete",
                "messages": [goodbye_message],
                "conversation_transcript": updated_transcript + [
                    {"role": "assistant", "content": goodbye_message.content}
                ]
            }
    
    except Exception as e:
        print(f"Error processing consent: {e}")
        # Fallback to simple string matching
        user_lower = user_response.lower().strip()
        
        if any(word in user_lower for word in ["yes", "yeah", "sure", "okay", "ok", "agree", "consent"]):
            proceed_message = AIMessage(
                content="Thank you for your consent. Let's proceed with the verification process."
            )
            return {
                "consent_given": True,
                "stage": "authenticate",
                "messages": [proceed_message],
                "conversation_transcript": state.get("conversation_transcript", []) + [
                    {"role": "user", "content": user_response},
                    {"role": "assistant", "content": proceed_message.content}
                ]
            }
        elif any(word in user_lower for word in ["no", "nope", "don't", "decline"]):
            goodbye_message = AIMessage(
                content="I understand. Without your consent, we cannot proceed. Thank you. Goodbye!"
            )
            return {
                "consent_given": False,
                "stage": "complete",
                "messages": [goodbye_message],
                "conversation_transcript": state.get("conversation_transcript", []) + [
                    {"role": "user", "content": user_response},
                    {"role": "assistant", "content": goodbye_message.content}
                ]
            }
        else:
            clarify_message = AIMessage(
                content="I couldn't understand your response. Please say 'yes' or 'no'."
            )
            return {
                "messages": [clarify_message],
                "conversation_transcript": state.get("conversation_transcript", []) + [
                    {"role": "user", "content": user_response},
                    {"role": "assistant", "content": clarify_message.content}
                ]
            }


def route_after_consent(state: AgentState) -> str:
    """Determine next step after processing consent"""
    consent_status = state.get("consent_given")
    
    if consent_status is None:
        # Need to ask again or wait for response
        return "get_consent"
    elif consent_status == True:
        return "authenticate"
    else:  # consent_status == False
        return "end"


# Build the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("get_consent", get_consent_node)
workflow.add_node("process_consent", process_consent_node)

# Set entry point
workflow.set_entry_point("get_consent")

# Add edges
workflow.add_edge("get_consent", "process_consent")

# Add conditional edge after processing consent
workflow.add_conditional_edges(
    "process_consent",
    route_after_consent,
    {
        "get_consent": "get_consent",
        "authenticate": END,  # Placeholder until authentication is implemented
        "end": END
    }
)

# Compile the graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# Test the workflow
if __name__ == "__main__":
    # Initial state
    initial_state = {
        "messages": [],
        "stage": "consent",
        "consent_asked": False,
        "consent_given": None,
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
        "conversation_transcript": []
    }
    
    # Configuration for thread
    config = {"configurable": {"thread_id": "test_conversation_1"}}
    
    # Step 1: Ask for consent
    print("\n=== Step 1: Asking for consent ===")
    result = app.invoke(initial_state, config)
    print(f"Assistant: {result['messages'][-1].content}")
    
    # Step 2: User provides consent (simulate user input)
    print("\n=== Step 2: User responds ===")
    user_message = HumanMessage(content="Yes, I agree")
    result = app.invoke({"messages": [user_message]}, config)
    print(f"User: {user_message.content}")
    print(f"Assistant: {result['messages'][-1].content}")
    print(f"Consent Given: {result.get('consent_given')}")
    print(f"Current Stage: {result.get('stage')}")
    
    # Print conversation transcript
    print("\n=== Conversation Transcript ===")
    for entry in result.get('conversation_transcript', []):
        print(f"{entry['role'].upper()}: {entry['content']}")
