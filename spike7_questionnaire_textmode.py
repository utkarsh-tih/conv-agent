"""
Complete LangGraph Loan Verification Telecalling Agent
Implements full credit underwriter verification workflow with identity checks
"""

import os

USE_TEXT_MODE = os.getenv("TEXT_MODE", "true").lower() == "true"

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
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
import re
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

# ============= DATABASE SETUP =============
DATABASE_URL = "postgresql://telecall_user:lolxd%402025@localhost/telecalling_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class LoanVerificationCall(Base):
    __tablename__ = "loan_verification_calls"
    call_sid = Column(String, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    transcript = Column(JSON)
    audio_paths = Column(JSON)
    verification_data = Column(JSON)  # Captured loan verification details
    identity_verified = Column(Boolean, default=False)
    consent_given = Column(Boolean, default=False)
    verification_status = Column(String)  # 'completed', 'partial', 'failed'
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ============= TTS/STT INITIALIZATION =============
print("Initializing engines...")
tts = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False
)
stt_model = whisper.load_model("base")
print("Engines initialized.")


# ============= LOAN VERIFICATION TOOLS =============


@tool
def get_loan_applicant_data(customer_id: str) -> dict:
    """Fetch loan applicant information from credit underwriter database."""
    print(f"[TOOL] Fetching loan applicant data: {customer_id}")
    # In production, fetch from actual database
    return {
        "customer_id": customer_id,
        "name": "Rajesh Kumar",
        "application_number": "LN2025001234",
        "loan_amount_requested": "500000",
        "loan_type": "Personal Loan",
        "cibil_score": "750",
        "kyc_status": "completed",
        "expected_dob": "15/06/1985",  # For verification
        "expected_aadhar_last4": "4567",  # For verification
        "language_preference": "english",
    }


@tool
def speak_to_customer(text: str, language: str = "english") -> dict:
    """Convert text to speech and play it to the customer."""

    if USE_TEXT_MODE:
        print("\n" + "=" * 60)
        print(f"Agent: {text}")
        print("=" * 60)
        return {
            "status": "success",
            "audio_path": "text_mode",
            "text_spoken": text,
            "language": language,
            "mode": "text",
        }

    print(f"[TOOL] Speaking ({language}): '{text}'")
    timestamp = datetime.now().timestamp()
    audio_path = f"audio_agent_{timestamp}.wav"

    try:
        tts.tts_to_file(text=text, file_path=audio_path)
        # Uncomment for actual playback:
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
    """Capture audio from microphone and transcribe it."""
    # TEXT MODE - for easy testing
    if USE_TEXT_MODE:
        print(f"\n[CUSTOMER INPUT REQUIRED]")
        print("=" * 60)
        try:
            text = input("You (customer): ").strip()

            if not text:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "quality": "failed",
                    "attempts": 1,
                    "mode": "text",
                }

            return {
                "text": text,
                "confidence": 1.0,
                "audio_path": "text_mode",
                "quality": "clear",
                "attempts": 1,
                "mode": "text",
            }
        except (EOFError, KeyboardInterrupt):
            print("\n[Input interrupted]")
            return {
                "text": "",
                "confidence": 0.0,
                "quality": "failed",
                "attempts": 1,
                "mode": "text",
            }

    print(f"[TOOL] Listening to customer (language: {language})...")
    r = sr.Recognizer()

    for attempt in range(max_retries + 1):
        try:
            with sr.Microphone() as source:
                print(f"  → Attempt {attempt + 1}/{max_retries + 1}: Speak now...")
                r.pause_threshold = 1.5
                r.adjust_for_ambient_noise(source, duration=1)
                audio_data = r.listen(source, timeout=10, phrase_time_limit=15)

            audio_path = f"customer_{datetime.now().timestamp()}.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_data.get_wav_data())

            result = stt_model.transcribe(audio_path, language=language)
            text = result["text"].strip()

            confidence = 0.95 if len(text) > 5 else 0.6
            quality = "clear" if confidence > 0.7 else "unclear"

            print(f"  → Heard: '{text}' (confidence: {confidence}, quality: {quality})")

            if confidence > 0.7 or attempt == max_retries:
                return {
                    "text": text,
                    "confidence": confidence,
                    "audio_path": audio_path,
                    "quality": quality,
                    "attempts": attempt + 1,
                }

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
def verify_consent(response: str) -> dict:
    """Verify if customer has given consent to proceed with verification."""
    print(f"[TOOL] Verifying consent from response: '{response}'")

    response_lower = response.lower().strip()

    # Positive consent indicators
    consent_words = ["yes", "yeah", "sure", "okay", "ok", "proceed", "agree", "consent"]
    rejection_words = ["no", "nope", "not", "don't", "refuse", "disagree"]

    has_consent = any(word in response_lower for word in consent_words)
    has_rejection = any(word in response_lower for word in rejection_words)

    if has_rejection:
        return {
            "consent_given": False,
            "status": "rejected",
            "message": "Customer declined consent",
        }
    elif has_consent:
        return {
            "consent_given": True,
            "status": "approved",
            "message": "Customer provided consent",
        }
    else:
        return {
            "consent_given": False,
            "status": "unclear",
            "message": "Response unclear, need explicit yes/no",
        }


@tool
def verify_identity(
    dob_input: str, aadhar_last4: str, expected_dob: str, expected_aadhar: str
) -> dict:
    """Verify customer identity using DOB and Aadhar last 4 digits."""
    print(f"[TOOL] Verifying identity - DOB: {dob_input}, Aadhar: {aadhar_last4}")

    # Clean inputs
    dob_clean = re.sub(r"[^\d]", "", dob_input)
    aadhar_clean = re.sub(r"[^\d]", "", aadhar_last4)
    expected_dob_clean = re.sub(r"[^\d]", "", expected_dob)
    expected_aadhar_clean = re.sub(r"[^\d]", "", expected_aadhar)

    # Validate format
    if len(dob_clean) != 8:
        return {
            "verified": False,
            "reason": "DOB must be 8 digits in DDMMYYYY format",
            "retry_allowed": True,
        }

    if len(aadhar_clean) != 4:
        return {
            "verified": False,
            "reason": "Aadhar last 4 digits must be exactly 4 digits",
            "retry_allowed": True,
        }

    # Match verification
    dob_match = dob_clean == expected_dob_clean
    aadhar_match = aadhar_clean == expected_aadhar_clean

    if dob_match and aadhar_match:
        return {
            "verified": True,
            "message": "Identity verified successfully",
            "dob_verified": True,
            "aadhar_verified": True,
        }
    else:
        return {
            "verified": False,
            "reason": f"Verification failed - DOB: {'✓' if dob_match else '✗'}, Aadhar: {'✓' if aadhar_match else '✗'}",
            "retry_allowed": True,
            "dob_verified": dob_match,
            "aadhar_verified": aadhar_match,
        }


@tool
def extract_verification_data(
    question_id: str, response: str, current_data: dict
) -> dict:
    """Extract and structure verification data from customer responses."""
    print(f"[TOOL] Extracting data for question {question_id}: '{response}'")

    # Store response in structured format
    updated_data = current_data.copy()

    # Map question IDs to data fields
    question_mapping = {
        "1": "discussant_name",
        "2": "current_address_proof_docs",
        "3": "permanent_address_proof_docs",
        "4": "current_address_ownership_proof",
        "5": "permanent_address_ownership_proof",
        "6": "current_residence_details",
        "7": "office_address",
        "8": "permanent_address_ovd",
        "9": "other_property_address",
        "10": "family_income_details",
        "11": "education_details",
        "12": "working_mode",
        "14": "office_email",
        "15": "current_job_details",
        "16": "total_experience",
        "17": "cibil_details",
        "18": "banking_details",
        "19": "loan_end_use",
        "20": "loan_amount_requirement",
    }

    field_name = question_mapping.get(question_id, f"question_{question_id}")

    updated_data[field_name] = {
        "response": response,
        "timestamp": datetime.now().isoformat(),
        "question_id": question_id,
    }

    return {
        "status": "success",
        "field_updated": field_name,
        "updated_data": updated_data,
    }


@tool
def validate_response_completeness(question_id: str, response: str) -> dict:
    """Validate if the response is complete and detailed enough for verification."""
    print(f"[TOOL] Validating completeness for Q{question_id}: '{response}'")

    response_lower = response.lower().strip()

    # Too short responses
    if len(response_lower) < 3:
        return {
            "is_complete": False,
            "reason": "Response too short",
            "action": "ask_again",
        }

    # Confusion indicators
    confusion_words = ["don't know", "not sure", "maybe", "i think", "confused"]
    if any(word in response_lower for word in confusion_words):
        return {
            "is_complete": False,
            "reason": "Customer seems uncertain",
            "action": "clarify_question",
        }

    # Question-specific validation
    detailed_questions = ["6", "7", "8", "10", "15", "16", "17", "18"]
    if question_id in detailed_questions and len(response_lower) < 20:
        return {
            "is_complete": False,
            "reason": "Response needs more detail",
            "action": "ask_for_details",
        }

    return {
        "is_complete": True,
        "reason": "Response appears complete",
        "action": "proceed",
    }


@tool
def save_loan_verification(
    call_sid: str,
    customer_id: str,
    transcript: list,
    verification_data: dict,
    audio_paths: list,
    identity_verified: bool,
    consent_given: bool,
    verification_status: str,
) -> dict:
    """Save complete loan verification call data to database."""
    print(f"[TOOL] Saving loan verification to database: {call_sid}")
    db = SessionLocal()
    try:
        existing = (
            db.query(LoanVerificationCall)
            .filter(LoanVerificationCall.call_sid == call_sid)
            .first()
        )

        if existing:
            existing.transcript = transcript
            existing.verification_data = verification_data
            existing.audio_paths = audio_paths
            existing.identity_verified = identity_verified
            existing.consent_given = consent_given
            existing.verification_status = verification_status
            print(f"  → Updated existing record")
        else:
            new_record = LoanVerificationCall(
                call_sid=call_sid,
                customer_id=customer_id,
                transcript=transcript,
                verification_data=verification_data,
                audio_paths=audio_paths,
                identity_verified=identity_verified,
                consent_given=consent_given,
                verification_status=verification_status,
            )
            db.add(new_record)
            print(f"  → Created new record")

        db.commit()
        return {
            "status": "success",
            "call_sid": call_sid,
            "verification_status": verification_status,
        }
    except Exception as e:
        db.rollback()
        print(f"  → Database error: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


# ============= COMPLETE TOOLS LIST =============
tools = [
    get_loan_applicant_data,
    speak_to_customer,
    listen_to_customer,
    verify_consent,
    verify_identity,
    extract_verification_data,
    validate_response_completeness,
    save_loan_verification,
]


# ============= LLM SETUP =============
def create_llm():
    """Create LLM instance with all tools bound."""
    llm = ChatOllama(
        model="llama3.2", temperature=0.3, base_url="http://localhost:11434"
    )
    return llm.bind_tools(tools)


LOCAL_LLM = create_llm()


# ============= VERIFICATION QUESTIONS =============
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


# ============= ENHANCED STATE DEFINITION =============
class LoanVerificationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    applicant_data: dict
    call_sid: str
    language: str
    transcript: list
    audio_paths: list
    verification_data: dict  # Structured captured data

    # Verification tracking
    consent_given: bool
    identity_verified: bool
    identity_retry_count: int

    # Question flow
    current_question_id: str
    questions_completed: list

    # Call management
    turn_count: int
    stage: Literal[
        "init",
        "greeting",
        "consent",
        "identity_verification",
        "questions",
        "closing",
        "end",
    ]
    retry_count: int
    last_audio_quality: str
    needs_clarification: bool


# ============= NODE FUNCTIONS =============


def agent_node(state: LoanVerificationState) -> LoanVerificationState:
    """Main agent reasoning node for loan verification workflow."""
    print(f"\n[AGENT NODE] Stage: {state['stage']}, Turn: {state['turn_count']}")

    # Build context-aware system prompt based on stage
    if state["stage"] == "init":
        system_msg = f"""You are a professional credit underwriter conducting a loan verification call.

FIRST STEPS:
1. Use get_loan_applicant_data to fetch applicant information
2. Use speak_to_customer to greet them professionally
   Example: "Hello, this is Sonam calling from Poonawalla Fincorp regarding your loan application number [Application Number]. Am I speaking with [Customer Name]?"

Keep it professional and clear."""

    elif state["stage"] == "greeting":
        system_msg = f"""Applicant: {state['applicant_data'].get('name', 'Customer')}
Application: {state['applicant_data'].get('application_number', 'N/A')}

You just greeted the customer. Now:
1. Use listen_to_customer to hear their response
2. Once confirmed, move to asking for consent

Be professional and courteous."""

    elif state["stage"] == "consent":
        system_msg = f"""CONSENT STAGE - This is mandatory for compliance.

ACTION REQUIRED:
1. Use speak_to_customer to explain and request consent:
   "Before we proceed with the verification, I need your consent to ask you some questions for loan processing purposes. This call may be recorded. Do you consent to proceed?"

2. Use listen_to_customer to get their response
3. Use verify_consent tool to validate the response
4. If consent is NOT given clearly, the call must be terminated professionally

CRITICAL: Without explicit consent, we CANNOT proceed."""

    elif state["stage"] == "identity_verification":
        system_msg = f"""IDENTITY VERIFICATION STAGE
Attempts: {state['identity_retry_count']}/3

ACTION REQUIRED:
1. Use speak_to_customer to request identity information:
   "For security purposes, I need to verify your identity. Please provide:
    - Your date of birth in DD MM YYYY format
    - Last 4 digits of your Aadhar card"

2. Use listen_to_customer TWICE (once for DOB, once for Aadhar)
3. Use verify_identity tool with the captured information
4. If verification fails and retries < 3, ask again politely
5. If verification succeeds, move to questions stage

IMPORTANT: Maximum 3 attempts allowed for identity verification."""

    elif state["stage"] == "questions":
        current_q = state.get("current_question_id", "1")
        completed = state.get("questions_completed", [])

        question_text = VERIFICATION_QUESTIONS.get(current_q, "")

        recent_transcript = state["transcript"][-3:]
        recent_summary = "\n".join(
            [f"- {t['speaker']}: {t['text'][:100]}" for t in recent_transcript]
        )

        system_msg = f"""LOAN VERIFICATION QUESTIONS STAGE

Current Question: #{current_q}
Question Text: "{question_text}"
Completed Questions: {len(completed)}/20
Recent conversation:
{recent_summary}

YOUR WORKFLOW:
1. If you haven't asked the current question yet:
   - Use speak_to_customer with the exact question text
   
2. If you just asked a question:
   - Use listen_to_customer to get the response
   - Use validate_response_completeness to check if response is adequate
   - If incomplete, ask follow-up questions for more detail
   - If complete, use extract_verification_data to store the information
   - Move to next question

3. For detailed questions (6, 7, 8, 10, 15, 16, 17, 18), ensure you get:
   - Complete addresses with landmarks
   - All numerical details (amounts, tenure, etc.)
   - Names of organizations/people
   - Dates and durations

IMPORTANT:
- Ask ONE question at a time
- Listen carefully to responses
- Request clarification if answers are vague
- Be patient and professional
- After question 20, move to closing stage"""

    elif state["stage"] == "closing":
        system_msg = f"""CLOSING STAGE

All verification questions completed. Now:
1. Use speak_to_customer to thank the customer:
   "Thank you for your time and cooperation, [Name]. We have completed the verification process. Your information will be reviewed and you'll be notified about the loan decision shortly. Have a great day!"

2. Use save_loan_verification to save all data to database
3. End the call

Be warm and professional in closing."""

    else:
        return state

    # Invoke LLM with tools
    messages = [SystemMessage(content=system_msg)] + list(state["messages"])
    response = LOCAL_LLM.invoke(messages)

    state["messages"].append(response)
    return state


def tool_execution_node(state: LoanVerificationState) -> LoanVerificationState:
    """Execute tools called by the LLM with enhanced state tracking."""
    print("[TOOL EXECUTION NODE]")

    last_message = state["messages"][-1]

    tool_messages = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"  → Executing: {tool_name}")

            tool_func = next((t for t in tools if t.name == tool_name), None)
            if tool_func:
                # Inject language if needed
                if tool_name in ["speak_to_customer", "listen_to_customer"]:
                    if "language" not in tool_args:
                        tool_args["language"] = state["language"]

                result = tool_func.invoke(tool_args)

                # Update state based on tool execution
                if tool_name == "speak_to_customer":
                    audio_path = result.get("audio_path", "")
                    if audio_path:
                        state["audio_paths"].append(audio_path)

                    state["transcript"].append(
                        {
                            "speaker": "agent",
                            "text": tool_args.get("text", ""),
                            "timestamp": datetime.now().isoformat(),
                            "audio_path": audio_path,
                        }
                    )

                elif tool_name == "listen_to_customer":
                    customer_text = result.get("text", "")
                    confidence = result.get("confidence", 0)
                    quality = result.get("quality", "unknown")
                    audio_path = result.get("audio_path", "")

                    if audio_path:
                        state["audio_paths"].append(audio_path)

                    state["last_audio_quality"] = quality

                    if quality in ["unclear", "very_poor"]:
                        state["needs_clarification"] = True
                        state["retry_count"] += 1
                    else:
                        state["needs_clarification"] = False
                        state["retry_count"] = 0

                    if customer_text:
                        state["transcript"].append(
                            {
                                "speaker": "customer",
                                "text": customer_text,
                                "confidence": confidence,
                                "quality": quality,
                                "timestamp": datetime.now().isoformat(),
                                "audio_path": audio_path,
                            }
                        )
                        state["messages"].append(HumanMessage(content=customer_text))

                elif tool_name == "get_loan_applicant_data":
                    state["applicant_data"] = result
                    state["language"] = result.get("language_preference", "english")

                elif tool_name == "verify_consent":
                    if result.get("consent_given", False):
                        state["consent_given"] = True
                        print("  ✓ Consent granted")
                    else:
                        state["consent_given"] = False
                        print("  ✗ Consent not granted")

                elif tool_name == "verify_identity":
                    if result.get("verified", False):
                        state["identity_verified"] = True
                        state["identity_retry_count"] = 0
                        print("  ✓ Identity verified")
                    else:
                        state["identity_retry_count"] += 1
                        print(
                            f"  ✗ Identity verification failed (attempt {state['identity_retry_count']}/3)"
                        )

                elif tool_name == "extract_verification_data":
                    state["verification_data"] = result.get(
                        "updated_data", state["verification_data"]
                    )

                    # Mark question as completed
                    q_id = tool_args.get("question_id", "")
                    if q_id and q_id not in state["questions_completed"]:
                        state["questions_completed"].append(q_id)

                    # Move to next question
                    current_q_num = int(state.get("current_question_id", "1"))
                    next_q_num = current_q_num + 1

                    # Skip question 13 (doesn't exist in the list)
                    if next_q_num == 13:
                        next_q_num = 14

                    if next_q_num <= 20:
                        state["current_question_id"] = str(next_q_num)
                    else:
                        # All questions done
                        state["stage"] = "closing"

                tool_msg = ToolMessage(
                    content=json.dumps(result), tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_msg)

    state["messages"].extend(tool_messages)
    state["turn_count"] += 1

    return state


def should_continue(state: LoanVerificationState) -> Literal["agent", "tools", "end"]:
    """Enhanced routing logic for loan verification workflow."""

    last_message = state["messages"][-1]

    # If LLM called tools, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Stage progression logic
    if state["stage"] == "init" and state.get("applicant_data"):
        state["stage"] = "greeting"
        return "agent"

    elif state["stage"] == "greeting" and state["turn_count"] >= 2:
        state["stage"] = "consent"
        return "agent"

    elif state["stage"] == "consent":
        if state.get("consent_given"):
            state["stage"] = "identity_verification"
            return "agent"
        elif state["turn_count"] >= 5:  # If no consent after multiple attempts
            print("⚠ Consent not obtained, ending call")
            return "end"

    elif state["stage"] == "identity_verification":
        if state.get("identity_verified"):
            state["stage"] = "questions"
            state["current_question_id"] = "1"
            return "agent"
        elif state.get("identity_retry_count", 0) >= 3:
            print("⚠ Identity verification failed after 3 attempts, ending call")
            return "end"

    elif state["stage"] == "questions":
        # Check if all questions are completed
        if (
            len(state.get("questions_completed", [])) >= 19
        ):  # 20 questions minus question 13
            state["stage"] = "closing"
            return "agent"

    elif state["stage"] == "closing":
        # End after closing actions are complete
        if state["turn_count"] >= state.get("turn_count", 0) + 2:
            return "end"

    return "agent"


# ============= GRAPH CONSTRUCTION =============


def create_graph():
    """Build the complete loan verification workflow graph."""
    workflow = StateGraph(LoanVerificationState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"agent": "agent", "tools": "tools", "end": END}
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============= MAIN EXECUTION =============


def run_loan_verification_call(customer_id: str, initial_language: str = "english"):
    """
    Execute a complete loan verification telecalling session.

    Args:
        customer_id: Loan applicant customer identifier
        initial_language: Initial language preference
    """
    print("\n" + "=" * 70)
    print("🏦 STARTING LOAN VERIFICATION TELECALLING SESSION")
    print("=" * 70)

    initial_state = {
        "messages": [],
        "applicant_data": {},
        "call_sid": f"loan_call_{int(time.time())}",
        "language": initial_language,
        "transcript": [],
        "audio_paths": [],
        "verification_data": {},
        "consent_given": False,
        "identity_verified": False,
        "identity_retry_count": 0,
        "current_question_id": "1",
        "questions_completed": [],
        "turn_count": 0,
        "stage": "init",
        "retry_count": 0,
        "last_audio_quality": "unknown",
        "needs_clarification": False,
    }

    initial_state["messages"].append(
        HumanMessage(
            content=f"Start loan verification call with customer ID: {customer_id}"
        )
    )

    app = create_graph()
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 70)
    print("✅ LOAN VERIFICATION CALL COMPLETED")
    print("=" * 70)

    # Print verification summary
    print(f"\n📋 VERIFICATION SUMMARY")
    print("-" * 70)
    print(f"Call SID: {final_state['call_sid']}")
    print(f"Customer: {final_state['applicant_data'].get('name', 'N/A')}")
    print(
        f"Application: {final_state['applicant_data'].get('application_number', 'N/A')}"
    )
    print(f"Consent Given: {'✓ Yes' if final_state.get('consent_given') else '✗ No'}")
    print(
        f"Identity Verified: {'✓ Yes' if final_state.get('identity_verified') else '✗ No'}"
    )
    print(f"Questions Completed: {len(final_state.get('questions_completed', []))}/19")
    print(f"Total Audio Files: {len(final_state['audio_paths'])}")

    # Determine verification status
    if (
        final_state.get("identity_verified")
        and len(final_state.get("questions_completed", [])) >= 19
    ):
        verification_status = "completed"
    elif (
        final_state.get("identity_verified")
        and len(final_state.get("questions_completed", [])) > 0
    ):
        verification_status = "partial"
    else:
        verification_status = "failed"

    print(f"Verification Status: {verification_status.upper()}")

    # Print captured verification data
    print(f"\n📊 CAPTURED VERIFICATION DATA")
    print("-" * 70)
    for key, value in final_state.get("verification_data", {}).items():
        if isinstance(value, dict):
            response_text = value.get("response", "N/A")
            print(
                f"{key}: {response_text[:100]}{'...' if len(response_text) > 100 else ''}"
            )
        else:
            print(f"{key}: {value}")

    # Print full transcript
    print(f"\n📝 FULL CALL TRANSCRIPT ({len(final_state['transcript'])} turns)")
    print("-" * 70)
    for turn in final_state["transcript"]:
        speaker = turn.get("speaker", "unknown").upper()
        text = turn.get("text", "")
        timestamp = turn.get("timestamp", "")
        audio = turn.get("audio_path", "")

        quality_indicator = ""
        if speaker == "CUSTOMER":
            confidence = turn.get("confidence", "N/A")
            quality = turn.get("quality", "")
            quality_indicator = f" [Quality: {quality}, Confidence: {confidence}]"

        audio_indicator = f" [Audio: {audio}]" if audio else ""

        print(f"\n[{speaker}]{quality_indicator}{audio_indicator}")
        print(f"Time: {timestamp}")
        print(f"Text: {text}")

    print("\n" + "=" * 70)

    # Save to database
    save_result = save_loan_verification.invoke(
        {
            "call_sid": final_state["call_sid"],
            "customer_id": customer_id,
            "transcript": final_state["transcript"],
            "verification_data": final_state["verification_data"],
            "audio_paths": final_state["audio_paths"],
            "identity_verified": final_state.get("identity_verified", False),
            "consent_given": final_state.get("consent_given", False),
            "verification_status": verification_status,
        }
    )

    print(f"\n💾 DATABASE SAVE STATUS: {save_result.get('status', 'unknown').upper()}")

    # Generate JSON export
    export_data = {
        "call_metadata": {
            "call_sid": final_state["call_sid"],
            "customer_id": customer_id,
            "customer_name": final_state["applicant_data"].get("name", "N/A"),
            "application_number": final_state["applicant_data"].get(
                "application_number", "N/A"
            ),
            "call_date": datetime.now().isoformat(),
            "language": final_state["language"],
        },
        "verification_status": {
            "overall_status": verification_status,
            "consent_given": final_state.get("consent_given", False),
            "identity_verified": final_state.get("identity_verified", False),
            "questions_completed": len(final_state.get("questions_completed", [])),
            "total_questions": 19,
        },
        "verification_data": final_state.get("verification_data", {}),
        "transcript": final_state["transcript"],
        "audio_files": final_state["audio_paths"],
    }

    # Save JSON to file
    json_filename = f"loan_verification_{final_state['call_sid']}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"📄 JSON export saved to: {json_filename}")
    print("=" * 70)

    return final_state


# ============= UTILITY FUNCTIONS =============


def generate_verification_report(call_sid: str) -> dict:
    """
    Generate a comprehensive verification report from database.

    Args:
        call_sid: Call session identifier
    Returns:
        Complete verification report
    """
    db = SessionLocal()
    try:
        call_record = (
            db.query(LoanVerificationCall)
            .filter(LoanVerificationCall.call_sid == call_sid)
            .first()
        )

        if not call_record:
            return {"error": "Call record not found"}

        report = {
            "call_sid": call_record.call_sid,
            "customer_id": call_record.customer_id,
            "verification_status": call_record.verification_status,
            "identity_verified": call_record.identity_verified,
            "consent_given": call_record.consent_given,
            "created_at": call_record.created_at.isoformat(),
            "transcript_summary": {
                "total_turns": len(call_record.transcript),
                "agent_turns": len(
                    [t for t in call_record.transcript if t.get("speaker") == "agent"]
                ),
                "customer_turns": len(
                    [
                        t
                        for t in call_record.transcript
                        if t.get("speaker") == "customer"
                    ]
                ),
            },
            "verification_data": call_record.verification_data,
            "audio_files": call_record.audio_paths,
        }

        return report
    finally:
        db.close()


def export_verification_to_csv(call_sid: str, output_file: str = None):
    """
    Export verification data to CSV format.

    Args:
        call_sid: Call session identifier
        output_file: Output CSV filename (optional)
    """
    import csv

    db = SessionLocal()
    try:
        call_record = (
            db.query(LoanVerificationCall)
            .filter(LoanVerificationCall.call_sid == call_sid)
            .first()
        )

        if not call_record:
            print(f"Error: Call record {call_sid} not found")
            return

        if not output_file:
            output_file = f"verification_{call_sid}.csv"

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(["Field", "Value"])

            # Write metadata
            writer.writerow(["Call SID", call_record.call_sid])
            writer.writerow(["Customer ID", call_record.customer_id])
            writer.writerow(["Verification Status", call_record.verification_status])
            writer.writerow(["Identity Verified", call_record.identity_verified])
            writer.writerow(["Consent Given", call_record.consent_given])
            writer.writerow(["Call Date", call_record.created_at.isoformat()])
            writer.writerow([])

            # Write verification data
            writer.writerow(["Verification Question", "Response"])
            for key, value in call_record.verification_data.items():
                if isinstance(value, dict):
                    response = value.get("response", "N/A")
                else:
                    response = str(value)
                writer.writerow([key, response])

        print(f"✓ CSV export saved to: {output_file}")

    finally:
        db.close()


def list_recent_verifications(limit: int = 10):
    """
    List recent loan verification calls.

    Args:
        limit: Number of recent calls to retrieve
    """
    db = SessionLocal()
    try:
        calls = (
            db.query(LoanVerificationCall)
            .order_by(LoanVerificationCall.created_at.desc())
            .limit(limit)
            .all()
        )

        print(f"\n📋 RECENT LOAN VERIFICATION CALLS (Last {limit})")
        print("=" * 100)
        print(
            f"{'Call SID':<25} {'Customer ID':<15} {'Status':<12} {'Identity':<10} {'Consent':<10} {'Date':<20}"
        )
        print("-" * 100)

        for call in calls:
            print(
                f"{call.call_sid:<25} {call.customer_id:<15} {call.verification_status:<12} "
                f"{'✓' if call.identity_verified else '✗':<10} "
                f"{'✓' if call.consent_given else '✗':<10} "
                f"{call.created_at.strftime('%Y-%m-%d %H:%M:%S'):<20}"
            )

        print("=" * 100)

    finally:
        db.close()


# ============= MAIN ENTRY POINT =============

if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "run" and len(sys.argv) > 2:
            # Run verification call
            customer_id = sys.argv[2]
            language = sys.argv[3] if len(sys.argv) > 3 else "english"
            final_state = run_loan_verification_call(
                customer_id=customer_id, initial_language=language
            )

        elif command == "report" and len(sys.argv) > 2:
            # Generate report for a call
            call_sid = sys.argv[2]
            report = generate_verification_report(call_sid)
            print(json.dumps(report, indent=2))

        elif command == "export" and len(sys.argv) > 2:
            # Export to CSV
            call_sid = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            export_verification_to_csv(call_sid, output_file)

        elif command == "list":
            # List recent calls
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            list_recent_verifications(limit)

        else:
            print("Usage:")
            print(
                "  python spike7.py run <customer_id> [language]     - Run verification call"
            )
            print(
                "  python spike7.py report <call_sid>                - Generate report"
            )
            print("  python spike7.py export <call_sid> [output.csv]   - Export to CSV")
            print(
                "  python spike7.py list [limit]                     - List recent calls"
            )
    else:
        # Default: Run with sample customer
        print("Running default verification call...")
        final_state = run_loan_verification_call(
            customer_id="Aditya", initial_language="english"
        )

        print("\n💡 TIP: You can also use command line arguments:")
        print("  python spike7.py run CUST123 english")
        print("  python spike7.py list")
        print(f"  python spike7.py report {final_state['call_sid']}")
        print(f"  python spike7.py export {final_state['call_sid']}")
