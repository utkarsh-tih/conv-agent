from typing import Sequence, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

# State Definition
class AgentState(TypedDict):
    """
    State schema using TypedDict for explicit type definitions.

    Why TypedDict instead of MessagesState?
    - MessagesState is a convenience class that includes messages handling
    - TypedDict gives you full control and clarity over your state structure
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


# Fixed Nodes
def model_call(state: AgentState) -> AgentState:
    """
    Call the LLM with system prompt and conversation history.
    Returns only the new message to be added (reducer handles merging).
    """
    system_prompt = SystemMessage(content=SYSTEM_PROMPT)
    
    # Combine system prompt with existing messages
    messages_to_send = [system_prompt] + list(state["messages"])
    
    # Get LLM response
    response = llm.invoke(messages_to_send)
    
    # Return only the new message - add_messages reducer will append it
    return {"messages": [response]}


def get_consent_node(state: AgentState) -> AgentState:
    """
    Ask for user consent if not already asked.
    Returns partial state updates.
    """
    print("==== Get Consent Node ====")
    
    # Check if consent already asked
    if not state.get("consent_asked", False):
        message = AIMessage(
            content=(
                "Hello! Before we begin the verification process, I need your consent. "
                "This conversation will be recorded for compliance purposes. "
                "Do you agree to proceed? (Please say yes or no)"
            )
        )
        
        # Return partial updates - only changed fields
        return {
            "messages": [message],  # add_messages will append this
            "consent_asked": True,
            "stage": "consent",
            "conversation_transcript": [
                {"role": "assistant", "content": message.content}
            ]
        }
    
    # If already asked, return empty dict (no changes)
    return {}


def process_consent_node(state: AgentState) -> AgentState:
    """
    Process the user's consent response.
    Extracts consent from the last user message.
    """
    print("==== Process Consent Node ====")
    
    # Get the last message (should be user's response)
    if state["messages"]:
        last_message = state["messages"][-1]
        
        if isinstance(last_message, HumanMessage):
            user_response = last_message.content.lower().strip()
            
            # Simple consent detection (you might want to use LLM for this)
            if "yes" in user_response or "agree" in user_response or "consent" in user_response:
                return {
                    "consent_given": True,
                    "stage": "authenticate",
                    "conversation_transcript": state.get("conversation_transcript", []) + [
                        {"role": "user", "content": last_message.content}
                    ]
                }
            elif "no" in user_response or "don't" in user_response or "decline" in user_response:
                goodbye_message = AIMessage(
                    content="I understand. Without consent, we cannot proceed. Thank you for your time. Goodbye!"
                )
                return {
                    "consent_given": False,
                    "stage": "complete",
                    "messages": [goodbye_message],
                    "conversation_transcript": state.get("conversation_transcript", []) + [
                        {"role": "user", "content": last_message.content},
                        {"role": "assistant", "content": goodbye_message.content}
                    ]
                }
    
    # If unclear, ask again
    clarify_message = AIMessage(
        content="I didn't quite catch that. Do you consent to this recorded conversation? Please say 'yes' or 'no'."
    )
    return {
        "messages": [clarify_message],
        "conversation_transcript": state.get("conversation_transcript", []) + [
            {"role": "assistant", "content": clarify_message.content}
        ]
    }


# Routing function for conditional edges
def should_continue_after_consent(state: AgentState) -> str:
    """
    Determines next step after processing consent.
    """
    if state.get("consent_given") == True:
        return "authenticate"
    elif state.get("consent_given") == False:
        return "end"
    else:
        return "get_consent"  # Ask again if unclear


# Example graph construction
def create_graph():
    """
    Build the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("get_consent", get_consent_node)
    workflow.add_node("process_consent", process_consent_node)
    workflow.add_node("model_call", model_call)
    
    # Add edges
    workflow.set_entry_point("get_consent")
    workflow.add_edge("get_consent", "process_consent")
    
    # Conditional edge after processing consent
    workflow.add_conditional_edges(
        "process_consent",
        should_continue_after_consent,
        {
            "authenticate": "model_call",  # Next stage
            "get_consent": "get_consent",   # Ask again
            "end": END                       # Terminate
        }
    )
    
    return workflow.compile()


# Usage example (you need to define SYSTEM_PROMPT and llm)
if __name__ == "__main__":
    # Example initialization
    SYSTEM_PROMPT = "You are a helpful verification assistant."
    
    # Initialize your LLM here
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4")
    
    # Create graph
    graph = create_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "stage": "consent",
        "consent_asked": False,
        "consent_given": False,
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
    
    # Run graph
    # result = graph.invoke(initial_state)
