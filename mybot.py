from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# from langchain.vectorstores import Chroma
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


NODES = Literal["get_consent_node", "process_consent_node", "get_authentication_node", "process_authetication_node", ]
MODEL = "llama3.2:1b"
CHROMADB_DIRECTORY = "./chroma_langchain_db"

"""
Initialisation of LLM, Embeddings and Vector Store
"""
llm = ChatOllama(model=MODEL, temperature=0.3, base_url="http://localhost:11434")
# embeddings = OllamaEmbeddings(model=MODEL)
# vector_store = Chroma(collection_name = "example_collection", embedding_function= embeddings, persist_directory= CHROMADB_DIRECTORY)

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

class AgentState(TypedDict):
    """
    State schema using TypedDict for explicit type definitions.

    Why TypedDict instead of MessagesState?
    - MessagesState is a convenience class that includes messages handling
    - TypedDict gives you full control and clarity over your state structureF
    - We use add_messages for proper message list management
    """

    # Messages with proper reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Flow control - tracks which stage we're at
    stage: str  # "consent", "authenticate", "questions", "complete"

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
    repeat_count: dict
    extraction_status: str

    # Compliance
    conversation_transcript: list


def model_call(state: AgentState)-> AgentState:
    system_prompt = SystemMessage(content= SYSTEM_PROMPT)
    response = llm.invoke([system_prompt])
    return {"messages": [response]}


def get_consent_node(state: AgentState) -> AgentState:
    print("====Get Consent Runnable ===")
    if state["consent_asked"] != True:
        message = AIMessage(
            content=(
                "Hello! Before we begin the verification process, I need your consent. "
                "This conversation will be recorded for compliance purposes. "
                "Do you agree to proceed? (Please say yes or no)"
            )
        )
        state["messages"] = message
        state["consent_asked"] = True
        state["conversation_transcript"] = [
            {"role": "assistant", "content": message.content}
        ]
    return state

def process_consent_node(state: AgentState)-> AgentState:
    if state["consent_given"] == True:
        return state
    elif state["consent_given"] == False:
        return state

def get_authentication_node(state: AgentState)-> AgentState:
    print("==== Get Authentication Runnable ===")
    message = AIMessage(
        content=(
            "For security purposes, please provide:\n"
            "1. Your date of birth (DD/MM/YYYY)\n"
            "2. Last 4 digits of your Aadhar card"
        )
    )
    state["conversation_transcript"].append(
        {"role": "assistant", "content": message.content}
    )
    state["auth_question_asked"] = True

    return state

def process_authentication_node(state: AgentState)-> AgentState:
    state["auth_attempts"] += 1 
    pass

def ask_question(state: AgentState)-> AgentState:
    pass

def extract_answer(state: AgentState)-> AgentState:
    pass


#conditional edge
def is_retry_required(state: AgentState)-> NODES:
    pass

def is_consent_given(state: AgentState):
    if state["consent_given"]==True:
        pass




workflow = StateGraph(AgentState)


app = workflow.compile()
initial_state = AgentState()
app.invoke(initial_state)