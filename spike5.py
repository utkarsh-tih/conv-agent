"""
Improved LangGraph Telecalling AI Agent with proper tool calling pattern
Key improvements:
- Proper LLM tool calling pattern
- ToolNode for automatic tool execution
- Better state management
- Cleaner graph flow
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
import operator
import json
from datetime import datetime
import os
import speech_recognition as sr
from TTS.api import TTS
import whisper
import wave
import time
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# ============= DATABASE SETUP =============
DATABASE_URL = "postgresql://telecall_user:lolxd%402025@localhost/telecalling_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CallTranscript(Base):
    __tablename__ = "call_transcripts"
    call_sid = Column(String, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    transcript = Column(JSON)
    audio_paths = Column(Text)
    customer_score = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ============= TTS/STT INITIALIZATION =============
print("Initializing engines...")
tts = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False
)
stt_model = whisper.load_model("base")
print("Engines initialized.")

# ============= TOOLS DEFINITION =============


@tool
def get_customer_data(customer_id: str) -> dict:
    """Fetch customer information from database."""
    print(f"[TOOL] Fetching customer data: {customer_id}")
    return {
        "customer_id": customer_id,
        "name": "Alex Doe",
        "phone": "N/A (microphone)",
        "language_preference": "english",
        "previous_interactions": 2,
        "risk_category": "medium",
    }


@tool
def speak_to_customer(text: str) -> dict:
    """
    Convert text to speech and play it to the customer.
    This is the ONLY way the agent should communicate verbally.

    Args:
        text: The message to speak to the customer
    Returns:
        Status and audio file path
    """
    print(f"[TOOL] Speaking: '{text}'")
    timestamp = datetime.now().timestamp()
    audio_path = f"audio_{timestamp}.wav"

    try:
        tts.tts_to_file(text=text, file_path=audio_path)
        # To actually play: from playsound import playsound; playsound(audio_path)
        return {"status": "success", "audio_path": audio_path, "text_spoken": text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def listen_to_customer() -> dict:
    """
    Capture audio from microphone and transcribe it.
    This is the ONLY way to get customer input.

    Returns:
        Transcribed text and confidence score
    """
    print("[TOOL] Listening to customer...")
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("  → Speak now...")
            r.pause_threshold = 1.5
            r.adjust_for_ambient_noise(source, duration=1)
            audio_data = r.listen(source, timeout=10)

        # Save and transcribe
        audio_path = f"customer_{datetime.now().timestamp()}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_data.get_wav_data())

        result = stt_model.transcribe(audio_path, language="english")
        text = result["text"].strip()
        confidence = 0.95 if len(text) > 5 else 0.6

        print(f"  → Heard: '{text}' (confidence: {confidence})")
        return {
            "text": text,
            "confidence": confidence,
            "audio_path": audio_path,
            "quality": "clear" if confidence > 0.7 else "unclear",
        }
    except Exception as e:
        print(f"  → Error: {e}")
        return {"text": "", "confidence": 0.0, "error": str(e), "quality": "failed"}


@tool
def save_conversation(
    call_sid: str, customer_id: str, transcript: list, score: dict
) -> dict:
    """Save the conversation transcript and score to database."""
    print(f"[TOOL] Saving conversation: {call_sid}")
    db = SessionLocal()
    try:
        existing = (
            db.query(CallTranscript).filter(CallTranscript.call_sid == call_sid).first()
        )
        if existing:
            existing.transcript = transcript
            existing.customer_score = score
        else:
            new_record = CallTranscript(
                call_sid=call_sid,
                customer_id=customer_id,
                transcript=transcript,
                customer_score=score,
                audio_paths=json.dumps([]),
            )
            db.add(new_record)
        db.commit()
        return {"status": "success", "call_sid": call_sid}
    except Exception as e:
        db.rollback()
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


@tool
def calculate_score(transcript: list) -> dict:
    """Calculate customer engagement and interaction score."""
    print("[TOOL] Calculating scores...")
    # Simple scoring logic - can be enhanced with LLM analysis
    customer_turns = [t for t in transcript if t.get("speaker") == "customer"]
    agent_turns = [t for t in transcript if t.get("speaker") == "agent"]

    engagement = min(100, len(customer_turns) * 20)
    clarity = 75  # Could analyze confidence scores
    overall = (engagement + clarity) / 2

    return {
        "overall_score": overall,
        "engagement_score": engagement,
        "clarity_score": clarity,
        "total_turns": len(transcript),
    }


# ============= TOOLS LIST =============
tools = [
    get_customer_data,
    speak_to_customer,
    listen_to_customer,
    save_conversation,
    calculate_score,
]


# ============= LLM SETUP =============
def create_llm():
    """Create LLM instance with tools bound."""
    llm = ChatOllama(
        model="llama3.1:8b", temperature=0.3, base_url="http://localhost:11434"
    )
    return llm.bind_tools(tools)


LOCAL_LLM = create_llm()


# ============= STATE DEFINITION =============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_data: dict
    call_sid: str
    transcript: list
    customer_score: dict
    turn_count: int
    stage: Literal["init", "greeting", "conversation", "closing", "end"]
    needs_retry: bool
    retry_count: int


# ============= NODE FUNCTIONS =============


def agent_node(state: AgentState) -> AgentState:
    """
    Main agent reasoning node - the LLM decides what to do next.
    This is where the LLM will call tools based on the conversation state.
    """
    print(f"\n[AGENT NODE] Stage: {state['stage']}, Turn: {state['turn_count']}")

    # Build system prompt based on stage
    if state["stage"] == "init":
        system_msg = """You are a professional telecalling agent. You have access to tools.
        
Your first task: Use get_customer_data to fetch customer info, then greet them using speak_to_customer."""

    elif state["stage"] == "greeting":
        system_msg = f"""Customer data: {json.dumps(state['customer_data'])}

You just greeted the customer. Now use listen_to_customer to hear their response."""

    elif state["stage"] == "conversation":
        system_msg = f"""You are in conversation with {state['customer_data']['name']}.
        
Based on the conversation history, decide what to do:
- If you need to ask a question: use speak_to_customer
- If you just spoke: use listen_to_customer to get their response
- If the conversation feels complete (3-4 exchanges): move to closing

Keep questions relevant and professional."""

    elif state["stage"] == "closing":
        system_msg = """Time to end the call professionally.
        
Tasks:
1. Use speak_to_customer to say goodbye
2. Use calculate_score to evaluate the conversation
3. Use save_conversation to store everything"""

    else:
        return state  # End stage

    # Invoke LLM with tools
    messages = [SystemMessage(content=system_msg)] + list(state["messages"])
    response = LOCAL_LLM.invoke(messages)

    # Add AI response to messages
    state["messages"].append(response)

    return state


def tool_execution_node(state: AgentState) -> AgentState:
    """
    Execute tools called by the LLM.
    This node is automatically called when the LLM generates tool calls.
    """
    print("[TOOL EXECUTION NODE]")

    last_message = state["messages"][-1]

    # Execute tools and create ToolMessages
    tool_messages = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Find and execute the tool
            tool_func = next((t for t in tools if t.name == tool_name), None)
            if tool_func:
                result = tool_func.invoke(tool_args)

                # Update transcript for speech/listen tools
                if tool_name == "speak_to_customer":
                    state["transcript"].append(
                        {
                            "speaker": "agent",
                            "text": tool_args.get("text", ""),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                elif tool_name == "listen_to_customer":
                    customer_text = result.get("text", "")
                    if customer_text:
                        state["transcript"].append(
                            {
                                "speaker": "customer",
                                "text": customer_text,
                                "confidence": result.get("confidence", 0),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        # Add customer message to conversation
                        state["messages"].append(HumanMessage(content=customer_text))

                elif tool_name == "calculate_score":
                    state["customer_score"] = result

                # Create tool message
                tool_msg = ToolMessage(
                    content=json.dumps(result), tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_msg)

    state["messages"].extend(tool_messages)
    state["turn_count"] += 1

    return state


def should_continue(state: AgentState) -> Literal["agent", "tools", "end"]:
    """Routing logic for the graph."""

    last_message = state["messages"][-1]

    # If LLM called tools, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Stage progression logic
    if state["stage"] == "init" and state["turn_count"] >= 1:
        state["stage"] = "greeting"
        return "agent"

    elif state["stage"] == "greeting" and state["turn_count"] >= 2:
        state["stage"] = "conversation"
        return "agent"

    elif state["stage"] == "conversation" and state["turn_count"] >= 8:
        state["stage"] = "closing"
        return "agent"

    elif state["stage"] == "closing" and state["turn_count"] >= 12:
        return "end"

    # Continue conversation
    return "agent"


# ============= GRAPH CONSTRUCTION =============


def create_graph():
    """Build the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent", should_continue, {"agent": "agent", "tools": "tools", "end": END}
    )

    # After tools, always return to agent
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============= MAIN EXECUTION =============


def run_call(customer_id: str):
    """Execute a complete telecalling session."""
    print("\n" + "=" * 60)
    print("STARTING TELECALLING SESSION")
    print("=" * 60)

    initial_state = {
        "messages": [],
        "customer_data": {},
        "call_sid": f"call_{int(time.time())}",
        "transcript": [],
        "customer_score": {},
        "turn_count": 0,
        "stage": "init",
        "needs_retry": False,
        "retry_count": 0,
    }

    # Add initial instruction
    initial_state["messages"].append(
        HumanMessage(content=f"Start a call with customer ID: {customer_id}")
    )

    app = create_graph()
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("CALL COMPLETED")
    print("=" * 60)
    print(f"Score: {final_state.get('customer_score', {})}")
    print(f"\nTranscript ({len(final_state['transcript'])} turns):")
    for turn in final_state["transcript"]:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        print(f"  [{speaker.upper()}]: {text}")

    return final_state


if __name__ == "__main__":
    run_call(customer_id="CUST456")
