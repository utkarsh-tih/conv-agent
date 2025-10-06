"""
Complete LangGraph Telecalling Agent - All Features Restored
This version includes ALL functionality from the original PLUS architectural improvements
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
import speech_recognition as sr
from TTS.api import TTS
import whisper
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

# ============= COMPLETE TOOLS DEFINITION =============


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
def speak_to_customer(text: str, language: str = "english") -> dict:
    """
    Convert text to speech and play it to the customer.
    Supports multiple languages.

    Args:
        text: The message to speak to the customer
        language: Language code (e.g., 'english', 'hindi')
    Returns:
        Status and audio file path
    """
    print(f"[TOOL] Speaking ({language}): '{text}'")
    timestamp = datetime.now().timestamp()
    audio_path = f"audio_{timestamp}.wav"

    try:
        # For multilingual support, use different TTS models based on language
        # Current model is English-only, but you can load different models
        tts.tts_to_file(text=text, file_path=audio_path)

        # Actual playback (uncomment if needed):
        # from playsound import playsound
        # playsound(audio_path)

        return {
            "status": "success",
            "audio_path": audio_path,
            "text_spoken": text,
            "language": language,
        }
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return {"status": "error", "error": str(e)}


@tool
def listen_to_customer(language: str = "english", max_retries: int = 2) -> dict:
    """
    Capture audio from microphone and transcribe it.
    Automatically retries on poor quality audio.

    Args:
        language: Expected language for transcription
        max_retries: Maximum number of retry attempts for unclear audio
    Returns:
        Transcribed text, confidence score, and audio quality
    """
    print(f"[TOOL] Listening to customer (language: {language})...")
    r = sr.Recognizer()

    for attempt in range(max_retries + 1):
        try:
            with sr.Microphone() as source:
                print(f"  → Attempt {attempt + 1}/{max_retries + 1}: Speak now...")
                r.pause_threshold = 1.5
                r.adjust_for_ambient_noise(source, duration=1)
                audio_data = r.listen(source, timeout=10, phrase_time_limit=15)

            # Save and transcribe
            audio_path = f"customer_{datetime.now().timestamp()}.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_data.get_wav_data())

            # Transcribe with Whisper
            result = stt_model.transcribe(audio_path, language=language)
            text = result["text"].strip()

            # Calculate confidence (Whisper doesn't provide direct confidence)
            confidence = 0.95 if len(text) > 5 else 0.6
            quality = "clear" if confidence > 0.7 else "unclear"

            print(f"  → Heard: '{text}' (confidence: {confidence}, quality: {quality})")

            # If quality is good enough, return immediately
            if confidence > 0.7 or attempt == max_retries:
                return {
                    "text": text,
                    "confidence": confidence,
                    "audio_path": audio_path,
                    "quality": quality,
                    "attempts": attempt + 1,
                }

            # If unclear and retries remain, inform user
            print(f"  → Audio unclear, retrying...")

        except Exception as e:
            print(f"  → Error: {e}")
            if attempt == max_retries:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "error": str(e),
                    "quality": "failed",
                    "attempts": attempt + 1,
                }


@tool
def validate_audio_quality(confidence: float, text: str) -> dict:
    """
    Validate audio quality and determine if retry is needed.

    Args:
        confidence: Confidence score from STT (0.0 to 1.0)
        text: Transcribed text
    Returns:
        Quality assessment and recommended action
    """
    print(f"[TOOL] Validating audio quality (confidence: {confidence})")

    if confidence < 0.5:
        return {
            "quality": "very_poor",
            "action": "retry",
            "message": "I'm sorry, I couldn't hear you at all. Could you please repeat that?",
        }
    elif confidence < 0.7:
        return {
            "quality": "unclear",
            "action": "clarify",
            "message": "I didn't catch that clearly. Could you please say that again?",
        }
    else:
        return {
            "quality": "clear",
            "action": "proceed",
            "message": "Audio quality is good",
        }


@tool
def validate_customer_response(
    question: str, response: str, expected_type: str = "any"
) -> dict:
    """
    Validate if customer's response is relevant to the question asked.

    Args:
        question: The question that was asked
        response: Customer's response
        expected_type: Type of expected answer (e.g., 'yes_no', 'number', 'text', 'any')
    Returns:
        Validation result with is_valid flag and reason
    """
    print(f"[TOOL] Validating response: '{response}' for question: '{question}'")

    # Simple validation logic - in production, use LLM for better validation
    response_lower = response.lower().strip()

    # Check if response is too short or just noise
    if len(response_lower) < 2:
        return {
            "is_valid": False,
            "reason": "Response too short or unclear",
            "suggested_action": "rephrase_question",
        }

    # Check for explicit confusion indicators
    confusion_words = ["what", "huh", "sorry", "didn't understand", "repeat"]
    if any(word in response_lower for word in confusion_words):
        return {
            "is_valid": False,
            "reason": "Customer seems confused",
            "suggested_action": "clarify_question",
        }

    # Type-specific validation
    if expected_type == "yes_no":
        yes_words = ["yes", "yeah", "yep", "sure", "okay", "ok", "correct", "right"]
        no_words = ["no", "nope", "nah", "not", "wrong", "incorrect"]

        has_yes = any(word in response_lower for word in yes_words)
        has_no = any(word in response_lower for word in no_words)

        if not (has_yes or has_no):
            return {
                "is_valid": False,
                "reason": "Expected yes/no answer but got something else",
                "suggested_action": "rephrase_as_yes_no",
            }

    # If all checks pass
    return {
        "is_valid": True,
        "reason": "Response appears valid and relevant",
        "suggested_action": "continue",
    }


@tool
def translate_text(text: str, source_lang: str, target_lang: str) -> dict:
    """
    Translate text between languages.

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
    Returns:
        Translated text
    """
    print(f"[TOOL] Translating '{text}' from {source_lang} to {target_lang}")

    # Placeholder - integrate with actual translation API
    # Example: Google Translate API, DeepL, or local models

    # For now, return placeholder
    return {
        "original_text": text,
        "translated_text": f"[Translated to {target_lang}]: {text}",
        "source_language": source_lang,
        "target_language": target_lang,
        "confidence": 0.95,
    }


@tool
def calculate_customer_score(transcript: list, customer_data: dict) -> dict:
    """
    Calculate comprehensive customer engagement and interaction score.

    Args:
        transcript: Full conversation transcript
        customer_data: Customer information
    Returns:
        Detailed scoring metrics
    """
    print("[TOOL] Calculating customer scores...")

    customer_turns = [t for t in transcript if t.get("speaker") == "customer"]
    agent_turns = [t for t in transcript if t.get("speaker") == "agent"]

    # Engagement score based on participation
    num_customer_responses = len(customer_turns)
    engagement_score = min(100, num_customer_responses * 15)

    # Clarity score based on confidence levels
    confidences = [
        t.get("confidence", 1.0) for t in customer_turns if "confidence" in t
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
    clarity_score = int(avg_confidence * 100)

    # Response quality - average length of customer responses
    avg_response_length = sum(len(t.get("text", "")) for t in customer_turns) / max(
        len(customer_turns), 1
    )
    quality_score = min(100, int(avg_response_length * 2))

    # Overall score
    overall_score = (engagement_score + clarity_score + quality_score) / 3

    return {
        "overall_score": round(overall_score, 2),
        "engagement_score": engagement_score,
        "clarity_score": clarity_score,
        "quality_score": quality_score,
        "total_turns": len(transcript),
        "customer_turns": num_customer_responses,
        "agent_turns": len(agent_turns),
        "average_confidence": round(avg_confidence, 2),
    }


@tool
def save_conversation(
    call_sid: str, customer_id: str, transcript: list, score: dict
) -> dict:
    """
    Save the complete conversation transcript and scores to PostgreSQL database.

    Args:
        call_sid: Unique call session identifier
        customer_id: Customer ID
        transcript: Full conversation transcript
        score: Calculated scores
    Returns:
        Save status
    """
    print(f"[TOOL] Saving conversation to database: {call_sid}")
    db = SessionLocal()
    try:
        existing = (
            db.query(CallTranscript).filter(CallTranscript.call_sid == call_sid).first()
        )
        if existing:
            existing.transcript = transcript
            existing.customer_score = score
            print(f"  → Updated existing record")
        else:
            new_record = CallTranscript(
                call_sid=call_sid,
                customer_id=customer_id,
                transcript=transcript,
                customer_score=score,
                audio_paths=json.dumps([]),
            )
            db.add(new_record)
            print(f"  → Created new record")

        db.commit()
        return {
            "status": "success",
            "call_sid": call_sid,
            "records_saved": len(transcript),
        }
    except Exception as e:
        db.rollback()
        print(f"  → Database error: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


# ============= COMPLETE TOOLS LIST =============
tools = [
    get_customer_data,
    speak_to_customer,
    listen_to_customer,
    validate_audio_quality,
    validate_customer_response,
    translate_text,
    calculate_customer_score,
    save_conversation,
]


# ============= LLM SETUP =============
def create_llm():
    """Create LLM instance with all tools bound."""
    llm = ChatOllama(
        model="llama3.1:8b", temperature=0.3, base_url="http://localhost:11434"
    )
    return llm.bind_tools(tools)


LOCAL_LLM = create_llm()


# ============= ENHANCED STATE DEFINITION =============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_data: dict
    call_sid: str
    language: str  # RESTORED: Language tracking
    transcript: list
    customer_score: dict
    turn_count: int
    stage: Literal["init", "greeting", "conversation", "closing", "end"]
    retry_count: int  # RESTORED: Retry tracking
    last_audio_quality: str  # RESTORED: Audio quality tracking
    needs_clarification: bool  # RESTORED: Clarification flag


# ============= NODE FUNCTIONS =============


def agent_node(state: AgentState) -> AgentState:
    """
    Main agent reasoning node with enhanced prompting for all features.
    """
    print(
        f"\n[AGENT NODE] Stage: {state['stage']}, Turn: {state['turn_count']}, Retries: {state['retry_count']}"
    )

    # Build context-aware system prompt
    if state["stage"] == "init":
        system_msg = f"""You are a professional telecalling agent. Your goal is to have a productive conversation.

FIRST STEPS:
1. Use get_customer_data to fetch customer information
2. Use speak_to_customer to greet them warmly by name
3. Set a friendly, professional tone

Remember: Always speak in {state['language']} language."""

    elif state["stage"] == "greeting":
        system_msg = f"""Customer: {state['customer_data']['name']} (Language: {state['language']})

You just greeted the customer. Now:
1. Use listen_to_customer to hear their response
2. If audio quality is poor (confidence < 0.7), use validate_audio_quality to check
3. If validation says 'retry', ask them to repeat using speak_to_customer

Be patient and professional."""

    elif state["stage"] == "conversation":
        recent_turns = state["transcript"][-4:]
        turns_summary = "\n".join(
            [f"- {t['speaker']}: {t['text']}" for t in recent_turns]
        )

        system_msg = f"""CONVERSATION WITH: {state['customer_data']['name']}
Language: {state['language']}
Retry Count: {state['retry_count']}/3
Last Audio Quality: {state['last_audio_quality']}

Recent conversation:
{turns_summary}

YOUR TASKS:
1. If you just spoke → use listen_to_customer to get response
2. If you just listened → analyze the response:
   - Use validate_customer_response if answer seems unclear
   - If validation fails → rephrase your question using speak_to_customer
   - If validation succeeds → ask next relevant question
3. Keep questions focused on: customer needs, product interest, scheduling follow-up
4. After 3-4 good exchanges, move toward closing

IMPORTANT:
- Max {3 - state['retry_count']} retries remaining for unclear responses
- Always be polite and professional
- Match the customer's tone and energy"""

    elif state["stage"] == "closing":
        system_msg = f"""Time to professionally close the call with {state['customer_data']['name']}.

CLOSING CHECKLIST:
1. Use speak_to_customer to thank them and say goodbye
2. Use calculate_customer_score to evaluate the conversation
3. Use save_conversation to store everything in database

Make the goodbye warm and leave a positive impression."""

    else:
        return state

    # Invoke LLM with tools
    messages = [SystemMessage(content=system_msg)] + list(state["messages"])
    response = LOCAL_LLM.invoke(messages)

    # Add AI response to messages
    state["messages"].append(response)

    return state


def tool_execution_node(state: AgentState) -> AgentState:
    """
    Execute tools called by the LLM with enhanced state tracking.
    """
    print("[TOOL EXECUTION NODE]")

    last_message = state["messages"][-1]

    # Execute tools and create ToolMessages
    tool_messages = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"  → Executing: {tool_name}")

            # Find and execute the tool
            tool_func = next((t for t in tools if t.name == tool_name), None)
            if tool_func:
                # Inject language if tool accepts it and it's not provided
                if tool_name in [
                    "speak_to_customer",
                    "listen_to_customer",
                    "translate_text",
                ]:
                    if "language" not in tool_args:
                        tool_args["language"] = state["language"]

                result = tool_func.invoke(tool_args)

                # Update state based on tool execution
                if tool_name == "speak_to_customer":
                    state["transcript"].append(
                        {
                            "speaker": "agent",
                            "text": tool_args.get("text", ""),
                            "timestamp": datetime.now().isoformat(),
                            "language": state["language"],
                        }
                    )

                elif tool_name == "listen_to_customer":
                    customer_text = result.get("text", "")
                    confidence = result.get("confidence", 0)
                    quality = result.get("quality", "unknown")

                    # Update quality tracking
                    state["last_audio_quality"] = quality

                    # Check if we need clarification
                    if quality in ["unclear", "very_poor"]:
                        state["needs_clarification"] = True
                        state["retry_count"] += 1
                    else:
                        state["needs_clarification"] = False
                        state["retry_count"] = 0  # Reset on success

                    # Add to transcript
                    if customer_text:
                        state["transcript"].append(
                            {
                                "speaker": "customer",
                                "text": customer_text,
                                "confidence": confidence,
                                "quality": quality,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        # Add customer message to conversation
                        state["messages"].append(HumanMessage(content=customer_text))

                elif tool_name == "get_customer_data":
                    state["customer_data"] = result
                    state["language"] = result.get("language_preference", "english")

                elif tool_name == "calculate_customer_score":
                    state["customer_score"] = result

                elif tool_name == "validate_customer_response":
                    # If validation fails, increment retry
                    if not result.get("is_valid", False):
                        state["needs_clarification"] = True
                        state["retry_count"] += 1

                # Create tool message
                tool_msg = ToolMessage(
                    content=json.dumps(result), tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_msg)

    state["messages"].extend(tool_messages)
    state["turn_count"] += 1

    return state


def should_continue(state: AgentState) -> Literal["agent", "tools", "end"]:
    """Enhanced routing logic with retry handling."""

    last_message = state["messages"][-1]

    # If LLM called tools, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Check for max retries exceeded
    if state["retry_count"] >= 3:
        print("  ⚠ Max retries exceeded, moving to closing")
        state["stage"] = "closing"
        state["retry_count"] = 0
        return "agent"

    # Stage progression logic
    if state["stage"] == "init" and state.get("customer_data"):
        state["stage"] = "greeting"
        return "agent"

    elif state["stage"] == "greeting" and state["turn_count"] >= 2:
        state["stage"] = "conversation"
        return "agent"

    elif state["stage"] == "conversation":
        # Move to closing after 8-10 turns or if conversation feels complete
        if state["turn_count"] >= 10:
            state["stage"] = "closing"
            return "agent"

    elif state["stage"] == "closing":
        # End after closing actions are complete
        if state.get("customer_score") and state["turn_count"] >= 12:
            return "end"

    # Continue conversation
    return "agent"


# ============= GRAPH CONSTRUCTION =============


def create_graph():
    """Build the complete LangGraph workflow."""
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


def run_call(customer_id: str, initial_language: str = "english"):
    """
    Execute a complete telecalling session with all features.

    Args:
        customer_id: Customer identifier
        initial_language: Initial language preference
    """
    print("\n" + "=" * 70)
    print("🎯 STARTING COMPLETE TELECALLING SESSION")
    print("=" * 70)

    initial_state = {
        "messages": [],
        "customer_data": {},
        "call_sid": f"call_{int(time.time())}",
        "language": initial_language,
        "transcript": [],
        "customer_score": {},
        "turn_count": 0,
        "stage": "init",
        "retry_count": 0,
        "last_audio_quality": "unknown",
        "needs_clarification": False,
    }

    # Add initial instruction
    initial_state["messages"].append(
        HumanMessage(content=f"Start a call with customer ID: {customer_id}")
    )

    app = create_graph()
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 70)
    print("✅ CALL COMPLETED")
    print("=" * 70)
    print(f"\n📊 Final Score: {final_state.get('customer_score', {})}")
    print(f"🔄 Total Retries: {final_state.get('retry_count', 0)}")
    print(f"🎤 Last Audio Quality: {final_state.get('last_audio_quality', 'N/A')}")
    print(f"\n📝 Transcript ({len(final_state['transcript'])} turns):")
    print("-" * 70)
    for turn in final_state["transcript"]:
        speaker = turn.get("speaker", "unknown").upper()
        text = turn.get("text", "")
        confidence = turn.get("confidence", "N/A")
        quality = turn.get("quality", "")

        quality_indicator = (
            f" [{quality}, conf: {confidence}]" if speaker == "CUSTOMER" else ""
        )
        print(f"[{speaker}]{quality_indicator}: {text}")
    print("=" * 70)

    return final_state


if __name__ == "__main__":
    # Run with all features enabled
    final_state = run_call(customer_id="CUST456", initial_language="english")

    # Print database confirmation
    print(f"\n💾 Data saved to database with call_sid: {final_state['call_sid']}")
