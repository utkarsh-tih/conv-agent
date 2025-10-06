"""
LangGraph Telecalling AI Agent with Microphone Integration
Supports: TTS/STT, Multilingual (English + Indic), Local deployment, Scoring

Key: Uses LOCAL LLM (Llama 3.1, Mistral, or Gemma) for tool selection and reasoning
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

# from langchain_openai import ChatOpenAI
import operator
import json
from datetime import datetime
import os
import speech_recognition as sr
from TTS.api import TTS
import whisper
import wave
import time
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Text, insert
from sqlalchemy.orm import sessionmaker, declarative_base

# from sqlalchemy.ext.declarative import declarative_base

# ============= DATABASE SETUP (SQLAlchemy) =============
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


# Create table if it doesn't exist
Base.metadata.create_all(bind=engine)

# ============= LOCAL TTS/STT ENGINE INITIALIZATION =============
# Initialize TTS Engine (Coqui TTS) - will download model on first run
print("Initializing Text-to-Speech engine...")
tts = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False
)
print("TTS engine initialized.")

# Initialize STT Engine (Whisper) - will download model on first run
print("Initializing Speech-to-Text engine...")
stt_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
print("STT engine initialized.")


# ============= TOOLS DEFINITION =============


@tool
def get_customer_data(customer_id: str) -> dict:
    """
    Fetch customer information from a mock database.
    Args:
        customer_id: Unique customer identifier.
    Returns:
        Customer data as JSON.
    """
    # Mock implementation - in a real scenario, this would query a database
    print(f"--- TOOL: Fetching data for customer {customer_id} ---")
    return {
        "customer_id": customer_id,
        "name": "Alex Doe",
        "phone": "N/A (using microphone)",
        "language_preference": "english",
        "previous_interactions": 2,
        "risk_category": "medium",
    }


@tool
def text_to_speech_and_play(text: str, language: str) -> dict:
    """
    Convert text to speech using a local TTS engine and play it.
    Supports English and can be extended for Indic languages.
    Args:
        text: Text to convert to speech.
        language: Language code (e.g., 'en').
    Returns:
        Audio file path and metadata.
    """
    print(f"--- TOOL: Converting text to speech: '{text}' ---")
    timestamp = datetime.now().timestamp()
    audio_path = f"audio_{timestamp}.wav"

    # Use Coqui TTS to generate audio file
    # Note: The chosen model is English-only. For multilingual, a different model is needed.
    tts.tts_to_file(text=text, file_path=audio_path)

    # This part is a placeholder for playing audio. You'll need a library like `playsound` or `pydub`.
    print(f"--- ACTION: (Simulated) Playing audio from {audio_path} ---")
    # from playsound import playsound
    # playsound(audio_path)

    return {
        "audio_path": audio_path,
        "duration": 5.0,  # Placeholder duration
        "language": language,
        "text": text,
    }


@tool
def speech_to_text_from_mic(language: str) -> dict:
    """
    Capture audio from the microphone and convert it to text using a local STT engine.
    Args:
        language: Expected language (e.g., 'english').
    Returns:
        Transcribed text with confidence score.
    """
    print("--- TOOL: Listening for speech from microphone... ---")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak now...")
        r.pause_threshold = 1.5
        r.adjust_for_ambient_noise(source, duration=1)
        audio_data = r.listen(source)

    # Save the captured audio to a temporary file
    audio_path = f"mic_input_{datetime.now().timestamp()}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_data.get_wav_data())

    print(f"--- TOOL: Transcribing audio from {audio_path} ---")
    # Use Whisper to transcribe the audio file
    result = stt_model.transcribe(audio_path, language=language)
    transcribed_text = result["text"]

    # Whisper doesn't provide a direct confidence score, so we simulate it.
    # For a real confidence score, you might need a different STT engine or custom logic.
    confidence = 0.95 if len(transcribed_text) > 5 else 0.6

    print(f"--- TOOL: Transcription result: '{transcribed_text}' ---")
    return {
        "text": transcribed_text,
        "confidence": confidence,
        "language": language,
        "audio_quality": "clear" if confidence > 0.7 else "unclear",
    }


@tool
def store_transcript_in_db(
    call_sid: str, customer_id: str, transcript: list, customer_score: dict
) -> bool:
    """
    Store conversation transcript and scores to a local PostgreSQL database.
    Args:
        call_sid: Unique call identifier.
        customer_id: The customer's ID.
        transcript: List of conversation turns.
        customer_score: Final scoring metrics.
    Returns:
        Success status.
    """
    print(f"--- TOOL: Storing transcript for call {call_sid} to database ---")
    db = SessionLocal()
    try:
        # Check if a record with the same call_sid exists
        existing_record = (
            db.query(CallTranscript).filter(CallTranscript.call_sid == call_sid).first()
        )
        if existing_record:
            # Update the existing record
            existing_record.transcript = transcript
            existing_record.customer_score = customer_score
        else:
            # Insert a new record
            new_transcript = CallTranscript(
                call_sid=call_sid,
                customer_id=customer_id,
                transcript=transcript,
                customer_score=customer_score,
                audio_paths=json.dumps([]),  # Placeholder
            )
            db.add(new_transcript)
        db.commit()
        return True
    except Exception as e:
        print(f"Database Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()


# Keep other tools as they are, since they are internal logic or placeholders
@tool
def calculate_customer_score(transcript: list, customer_data: dict) -> dict:
    """Calculate customer scoring based on conversation"""
    print("--- TOOL: Calculating customer score ---")
    return {"overall_score": 75, "engagement_score": 80, "clarity_score": 70}


@tool
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text between languages"""
    print(f"--- TOOL: Translating '{text}' from {source_lang} to {target_lang} ---")
    return f"Translated: {text}"  # Placeholder


@tool
def validate_audio_quality(audio_metrics: dict) -> dict:
    """Validate audio quality and determine if retry needed"""
    confidence = audio_metrics.get("confidence", 0)
    if confidence < 0.5:
        return {"quality": "very_poor", "action": "retry"}
    elif confidence < 0.7:
        return {"quality": "unclear", "action": "clarify"}
    else:
        return {"quality": "clear", "action": "proceed"}


# ============= TOOLS LIST & LLM CONFIGURATION =============
# Create a list of all tools for the agent
tools = [
    get_customer_data,
    text_to_speech_and_play,
    speech_to_text_from_mic,
    store_transcript_in_db,
    calculate_customer_score,
    translate_text,
    validate_audio_quality,
]


def initialize_local_llm_with_tools(model_choice: str = "llama3.1"):
    """
    Initialize local LLM and BIND the tools to it for effective tool calling.
    """
    llm = ChatOllama(
        model=model_choice,
        temperature=0.3,
        base_url="http://localhost:11434",
        format="json",  # Essential for structured output/tool-calling
    )
    # Bind the tools to the LLM instance. This is crucial for the LLM
    # to know what tools it can use and how to format its output for them.
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools


# Create global LLM instance with tools bound
LOCAL_LLM = initialize_local_llm_with_tools("llama3.1:8b")


# ============= STATE DEFINITION =============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_data: dict
    call_sid: str
    language: str
    transcript: list
    customer_score: dict
    current_question: str
    retry_count: int
    audio_quality: str
    conversation_stage: str
    needs_clarification: bool
    error_type: Literal["unclear_audio", "wrong_answer", "no_response", None]


# ============= NODE FUNCTIONS =============


def initiate_call_node(state: AgentState) -> AgentState:
    print("\n>>> NODE: initiate_call_node")
    customer_data = state["customer_data"]
    language = customer_data.get("language_preference", "english")

    system_prompt = f"""You are a professional telecalling AI agent.
Customer Details: {json.dumps(customer_data)}
Language: {language}
Task: Generate a warm, professional greeting to start the call. Keep it brief."""

    response = LOCAL_LLM.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate the initial greeting."),
        ]
    )

    greeting = response.content
    text_to_speech_and_play.invoke({"text": greeting, "language": language})

    state["messages"].append(AIMessage(content=greeting))
    state["language"] = language
    state["conversation_stage"] = "greeting"
    state["current_question"] = greeting
    state["retry_count"] = 0
    state["transcript"].append({"speaker": "agent", "text": greeting})
    return state


def listen_and_transcribe_node(state: AgentState) -> AgentState:
    print("\n>>> NODE: listen_and_transcribe_node")
    # Use the new tool to get input from the microphone
    stt_result = speech_to_text_from_mic.invoke({"language": state["language"]})

    quality_check = validate_audio_quality.invoke(
        {"confidence": stt_result["confidence"]}
    )

    state["audio_quality"] = quality_check["quality"]
    state["transcript"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "speaker": "customer",
            "text": stt_result["text"],
            "confidence": stt_result["confidence"],
        }
    )

    if quality_check["quality"] in ["unclear", "very_poor"]:
        state["needs_clarification"] = True
        state["error_type"] = "unclear_audio"
    else:
        state["needs_clarification"] = False
        state["messages"].append(HumanMessage(content=stt_result["text"]))

    return state


def handle_unclear_audio_node(state: AgentState) -> AgentState:
    print("\n>>> NODE: handle_unclear_audio_node")
    state["retry_count"] += 1
    if state["retry_count"] > 2:
        print("--- Too many retries for unclear audio. Ending call. ---")
        state["conversation_stage"] = "closing"
        return state

    message = "I'm sorry, I couldn't hear you clearly. Could you please repeat that?"
    text_to_speech_and_play.invoke({"text": message, "language": state["language"]})

    state["messages"].append(AIMessage(content=message))
    state["needs_clarification"] = False  # Reset flag
    return state


def validate_answer_node(state: AgentState) -> AgentState:
    print("\n>>> NODE: validate_answer_node")
    last_customer_message = state["messages"][-1].content

    validation_prompt = f"""You are validating a customer's response.
Current Question: {state['current_question']}
Customer's Answer: {last_customer_message}
Task: Is the answer valid and relevant?
Respond with JSON: {{"is_valid": true/false, "reason": "brief explanation"}}"""

    response = LOCAL_LLM.invoke(
        [
            SystemMessage(
                content="You are a validation assistant. Respond with valid JSON."
            ),
            HumanMessage(content=validation_prompt),
        ]
    )

    try:
        validation_result = json.loads(response.content)
        if not validation_result.get("is_valid", False):
            state["error_type"] = "wrong_answer"
            state["needs_clarification"] = True
            state["retry_count"] += 1
        else:
            state["error_type"] = None
            state["retry_count"] = 0
    except json.JSONDecodeError:
        print("--- LLM validation failed to return valid JSON. Assuming valid. ---")
        state["error_type"] = None

    return state


def ask_question_node(state: AgentState) -> AgentState:
    print("\n>>> NODE: ask_question_node")
    conversation_history = "\n".join(
        [f"{type(msg).__name__}: {msg.content}" for msg in state["messages"][-4:]]
    )

    prompt_task = "Generate the NEXT question to ask the customer. Keep it concise."
    if state.get("error_type") == "wrong_answer" and state["retry_count"] < 3:
        prompt_task = "The customer gave an unclear answer. Politely rephrase the question or ask for clarification."

    system_prompt = f"""You are a telecalling AI agent.
Customer Data: {json.dumps(state['customer_data'])}
Previous Conversation:
{conversation_history}
Task: {prompt_task}
Respond with ONLY the question text."""

    response = LOCAL_LLM.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content="What should I ask next?"),
        ]
    )

    question = response.content.strip()
    text_to_speech_and_play.invoke({"text": question, "language": state["language"]})

    state["messages"].append(AIMessage(content=question))
    state["current_question"] = question
    state["transcript"].append({"speaker": "agent", "text": question})

    # Simple logic to move to closing stage
    if len(state["messages"]) > 8:
        state["conversation_stage"] = "closing"
    elif state["conversation_stage"] == "greeting":
        state["conversation_stage"] = "questioning"

    return state


def score_and_store_node(state: AgentState) -> AgentState:
    print("\n>>> NODE: score_and_store_node")
    closing_message = "Thank you for your time. Have a great day!"
    text_to_speech_and_play.invoke(
        {"text": closing_message, "language": state["language"]}
    )
    state["transcript"].append({"speaker": "agent", "text": closing_message})

    score = calculate_customer_score.invoke(
        {"transcript": state["transcript"], "customer_data": state["customer_data"]}
    )
    state["customer_score"] = score

    store_transcript_in_db.invoke(
        {
            "call_sid": state["call_sid"],
            "customer_id": state["customer_data"]["customer_id"],
            "transcript": state["transcript"],
            "customer_score": score,
        }
    )

    return state


def routing_function(
    state: AgentState,
) -> Literal["handle_unclear", "validate", "ask_question", "score_and_store"]:
    print(
        f"\n>>> ROUTER: Stage: {state['conversation_stage']}, Messages: {len(state['messages'])}, Needs Clarification: {state['needs_clarification']}"
    )

    if state["needs_clarification"] and state["error_type"] == "unclear_audio":
        return "handle_unclear"

    if state["retry_count"] >= 3:
        print("--- Max retries reached. Moving to end. ---")
        return "score_and_store"

    # If the conversation is marked for closing
    if state["conversation_stage"] == "closing":
        return "score_and_store"

    # After listening, if the last message was from a human, it needs validation.
    if isinstance(state["messages"][-1], HumanMessage):
        return "validate"

    # Default action is to ask the next question
    return "ask_question"


# ============= GRAPH CONSTRUCTION =============


def create_telecalling_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("initiate_call", initiate_call_node)
    workflow.add_node("listen", listen_and_transcribe_node)
    workflow.add_node("handle_unclear", handle_unclear_audio_node)
    workflow.add_node("validate", validate_answer_node)
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("score_and_store", score_and_store_node)

    workflow.set_entry_point("initiate_call")

    workflow.add_edge("initiate_call", "listen")
    workflow.add_edge(
        "handle_unclear", "listen"
    )  # After asking to repeat, listen again
    workflow.add_edge(
        "ask_question", "listen"
    )  # After asking a question, listen for an answer

    # Conditional routing after validation and listening
    workflow.add_conditional_edges(
        "validate",
        lambda s: "ask_question",  # Always ask the next question after validation
    )

    workflow.add_conditional_edges(
        "listen",
        routing_function,
        {
            "handle_unclear": "handle_unclear",
            "validate": "validate",
            "score_and_store": "score_and_store",
            "ask_question": "ask_question",  # Fallback
        },
    )

    workflow.add_edge("score_and_store", END)

    return workflow.compile()


# ============= USAGE EXAMPLE =============
def run_telecalling_agent(customer_id: str):
    print("--- Starting Telecalling Agent ---")
    customer_data = get_customer_data.invoke({"customer_id": customer_id})
    call_sid = f"mic_session_{int(time.time())}"  # Create a unique ID for this session

    initial_state = {
        "messages": [],
        "customer_data": customer_data,
        "call_sid": call_sid,
        "language": customer_data.get("language_preference", "english"),
        "transcript": [],
        "customer_score": {},
        "current_question": "",
        "retry_count": 0,
        "audio_quality": "clear",
        "conversation_stage": "greeting",
        "needs_clarification": False,
        "error_type": None,
    }

    app = create_telecalling_graph()

    # The graph will run until it hits the END state
    final_state = app.invoke(initial_state)

    print("\n--- Call Session Ended ---")
    print(f"Final Score: {final_state['customer_score']}")
    print("Full Transcript:")
    for turn in final_state["transcript"]:
        print(f"  - {turn.get('speaker', 'system')}: {turn.get('text')}")

    return final_state


if __name__ == "__main__":
    run_telecalling_agent(customer_id="CUST456")
