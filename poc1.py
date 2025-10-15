from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# from langchain.vectorstores import Chroma
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MODEL = "llama3.2:1b"
CHROMADB_DIRECTORY = "./chroma_langchain_db"

"""
Initialisation of LLM, Embeddings and Vector Store
"""
llm = ChatOllama(model=MODEL, temperature=0.3, base_url="http://localhost:11434")
# embeddings = OllamaEmbeddings(model=MODEL)
# vector_store = Chroma(collection_name = "example_collection", embedding_function= embeddings, persist_directory= CHROMADB_DIRECTORY)

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

response = llm.invoke([HumanMessage(content="Hi! I'm Bob")])


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = llm.invoke(prompt)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Bob."
language = "Spanish"

# Define a new graph
workflow = StateGraph(state_schema=State)

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)