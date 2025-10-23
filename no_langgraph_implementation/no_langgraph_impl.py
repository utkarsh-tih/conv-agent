"""
Implementation without LangGraph - using pure Python state machine.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import Optional
import os

# --- Configuration ---
# MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
# BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_RETRIES = 3
SYSTEM_RETRIES = 3  # For LLM/system errors

# # --- LLM Initialization ---
# llm = ChatOllama(model=MODEL, temperature=0, base_url=BASE_URL)
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEBlfBBLJhRZ47mGtmwSwXmnFNnJmVzPM"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# --- Enums ---
class SystemPrompts(Enum):
    PROCESS_CONSENT = (
        "You are a consent validator. The user was asked if they consent to being recorded. "
        "Analyze their response and return a JSON object.\n"
        "Consent is given for responses like: 'yes', 'yeah', 'sure', 'ok', 'I agree', 'I consent'. "
        "Consent is NOT given for: 'no', 'nope', unclear responses, or anything ambiguous."
    )
    PROCESS_AUTHENTICATION = (
        "You are an authentication validator. Extract the user's date of birth and last 4 digits of Aadhaar. "
        "Date of birth should be in DD/MM/YYYY format. "
        "Aadhaar last 4 digits should be a 4-digit number. "
        "If information is unclear or missing, set should_retry to true."
    )


class AIMessages(Enum):
    CONSENT_QUESTION = (
        "Hello! Before we begin the verification process, I need your consent. "
        "This conversation will be recorded for compliance purposes. "
        "Do you agree to proceed? (Please say yes or no)"
    )
    POSITIVE_CONSENT_RESPONSE = (
        "Thank you for your consent. Let's proceed with the verification."
    )
    NEGATIVE_CONSENT_RESPONSE = "I understand you did not consent. Goodbye."
    CONSENT_RETRY = (
        "I didn't quite understand. Could you please clearly say 'yes' or 'no'?"
    )
    AUTHENTICATION_QUESTION = (
        "For security purposes, please provide:\n"
        "1. Your date of birth (DD/MM/YYYY)\n"
        "2. Last 4 digits of your Aadhar card"
    )
    POSITIVE_AUTHENTICATION_RESPONSE = (
        "Thank you for your authentication. Let's proceed with the verification."
    )
    NEGATIVE_AUTHENTICATION_RESPONSE = "I am sorry we could not authenticate. Goodbye."
    AUTHENTICATION_RETRY = "I couldn't capture all the details. Please provide both your date of birth and last 4 digits of Aadhaar again."
    SYSTEM_ERROR_FINAL = "I'm experiencing technical difficulties. Please try again later."


# --- Pydantic Models ---
class ConsentResult(BaseModel):
    consent_given: bool = Field(..., description="Whether consent was given")
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str = Field(...)
    should_retry: bool = Field(...)


class AuthenticationResult(BaseModel):
    date_of_birth: str = Field(...)
    aadhaar_last4: str = Field(...)
    confidence_date_of_birth: int = Field(..., ge=0, le=100)
    confidence_aadhaar: int = Field(..., ge=0, le=100)
    reasoning_date_of_birth: str = Field(...)
    reasoning_aadhaar: str = Field(...)
    should_retry: bool = Field(...)


# --- State Class ---
class ConversationState:
    """Manages conversation state."""
    
    def __init__(self):
        self.stage = "consent"  # consent, authenticate, questions, complete
        self.consent_given = False
        self.consent_confidence = 0
        self.consent_reasoning = ""
        self.consent_retry_count = 0
        self.authenticated = False
        self.auth_attempts = 0
        self.user_dob = ""
        self.user_aadhar_digits = ""
        self.conversation_history = []
    
    def add_message(self, role: str, content: str):
        """Add message to history."""
        self.conversation_history.append({"role": role, "content": content})


# --- Helper Function for Retry Logic ---
def call_llm_with_retry(llm_func, max_retries=SYSTEM_RETRIES):
    """Wrapper to retry LLM calls on system errors."""
    for attempt in range(max_retries):
        try:
            return llm_func()
        except (ValidationError, Exception) as e:
            print(f"  [System] LLM error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
    return None


# --- Core Logic Functions ---
def handle_consent_flow(state: ConversationState) -> bool:
    """
    Handle the entire consent flow.
    Returns True if should continue to authentication, False if should end.
    """
    print("\n" + "=" * 70)
    print("CONSENT STAGE")
    print("=" * 70)
    
    while state.consent_retry_count <= MAX_RETRIES:
        # Ask for consent
        if state.consent_retry_count == 0:
            message = AIMessages.CONSENT_QUESTION.value
        else:
            message = AIMessages.CONSENT_RETRY.value
        
        print(f"\nAI: {message}")
        state.add_message("assistant", message)
        
        # Get user input
        user_input = input("\nYou: ").strip()
        if not user_input:
            print("Please enter a response.")
            continue
        
        state.add_message("user", user_input)
        
        # Process consent with retry logic
        try:
            def llm_call():
                messages = [
                    SystemMessage(content=SystemPrompts.PROCESS_CONSENT.value),
                    HumanMessage(content=user_input),
                ]
                return llm.with_structured_output(ConsentResult).invoke(messages)
            
            result = call_llm_with_retry(llm_call)
            
            if result is None:
                raise Exception("LLM failed after all retries")
            
            # Update state
            state.consent_given = result.consent_given
            state.consent_confidence = result.confidence
            state.consent_reasoning = result.reasoning
            
            print(f"\n  [Debug] Consent: {result.consent_given}, Confidence: {result.confidence}")
            print(f"  [Debug] Should retry: {result.should_retry}, Retry count: {state.consent_retry_count}")
            
            # Decision logic
            if result.should_retry and state.consent_retry_count < MAX_RETRIES:
                state.consent_retry_count += 1
                continue
            elif result.consent_given:
                # Success - move to authentication
                response = AIMessages.POSITIVE_CONSENT_RESPONSE.value
                print(f"\nAI: {response}")
                state.add_message("assistant", response)
                state.stage = "authenticate"
                return True
            else:
                # Rejection or max retries
                response = AIMessages.NEGATIVE_CONSENT_RESPONSE.value
                print(f"\nAI: {response}")
                state.add_message("assistant", response)
                state.stage = "complete"
                return False
                
        except Exception as e:
            print(f"\n  [System Error] {e}")
            response = AIMessages.SYSTEM_ERROR_FINAL.value
            print(f"\nAI: {response}")
            state.add_message("assistant", response)
            state.stage = "complete"
            return False
    
    # Max retries exceeded
    response = AIMessages.NEGATIVE_CONSENT_RESPONSE.value
    print(f"\nAI: {response}")
    state.add_message("assistant", response)
    state.stage = "complete"
    return False


def handle_authentication_flow(state: ConversationState) -> bool:
    """
    Handle the entire authentication flow.
    Returns True if authenticated successfully, False otherwise.
    """
    print("\n" + "=" * 70)
    print("AUTHENTICATION STAGE")
    print("=" * 70)
    
    while state.auth_attempts <= MAX_RETRIES:
        # Ask for authentication
        if state.auth_attempts == 0:
            message = AIMessages.AUTHENTICATION_QUESTION.value
        else:
            message = AIMessages.AUTHENTICATION_RETRY.value
        
        print(f"\nAI: {message}")
        state.add_message("assistant", message)
        
        # Get user input
        user_input = input("\nYou: ").strip()
        if not user_input:
            print("Please enter a response.")
            continue
        
        state.add_message("user", user_input)
        
        # Process authentication with retry logic
        try:
            def llm_call():
                messages = [
                    SystemMessage(content=SystemPrompts.PROCESS_AUTHENTICATION.value),
                    HumanMessage(content=user_input),
                ]
                return llm.with_structured_output(AuthenticationResult).invoke(messages)
            
            result = call_llm_with_retry(llm_call)
            
            if result is None:
                raise Exception("LLM failed after all retries")
            
            # Update state
            state.user_dob = result.date_of_birth
            state.user_aadhar_digits = result.aadhaar_last4
            state.authenticated = (
                result.confidence_date_of_birth > 70
                and result.confidence_aadhaar > 70
            )
            
            print(f"\n  [Debug] DOB: {result.date_of_birth} (conf: {result.confidence_date_of_birth})")
            print(f"  [Debug] Aadhaar: {result.aadhaar_last4} (conf: {result.confidence_aadhaar})")
            print(f"  [Debug] Should retry: {result.should_retry}, Attempts: {state.auth_attempts}")
            
            # Decision logic
            if result.should_retry and state.auth_attempts < MAX_RETRIES:
                state.auth_attempts += 1
                continue
            elif state.authenticated and not result.should_retry:
                # Success
                response = AIMessages.POSITIVE_AUTHENTICATION_RESPONSE.value
                print(f"\nAI: {response}")
                state.add_message("assistant", response)
                state.stage = "questions"
                return True
            else:
                # Failed authentication or max retries
                response = AIMessages.NEGATIVE_AUTHENTICATION_RESPONSE.value
                print(f"\nAI: {response}")
                state.add_message("assistant", response)
                state.stage = "complete"
                return False
                
        except Exception as e:
            print(f"\n  [System Error] {e}")
            response = AIMessages.SYSTEM_ERROR_FINAL.value
            print(f"\nAI: {response}")
            state.add_message("assistant", response)
            state.stage = "complete"
            return False
    
    # Max retries exceeded
    response = AIMessages.NEGATIVE_AUTHENTICATION_RESPONSE.value
    print(f"\nAI: {response}")
    state.add_message("assistant", response)
    state.stage = "complete"
    return False


# --- Main Execution ---
def run_conversation():
    """Run the complete conversation flow."""
    print("=" * 70)
    print("STARTING CONVERSATION")
    print("=" * 70)
    
    state = ConversationState()
    
    # Stage 1: Consent
    should_continue = handle_consent_flow(state)
    
    if not should_continue:
        print_final_state(state)
        return
    
    # Stage 2: Authentication
    authenticated = handle_authentication_flow(state)
    
    if not authenticated:
        print_final_state(state)
        return
    
    # Stage 3: Questions (placeholder)
    print("\n" + "=" * 70)
    print("QUESTIONS STAGE")
    print("=" * 70)
    print("\nAI: You can now proceed with questions...")
    state.stage = "complete"
    
    print_final_state(state)


def print_final_state(state: ConversationState):
    """Print final conversation state."""
    print("\n" + "=" * 70)
    print("CONVERSATION COMPLETE")
    print("=" * 70)
    print(f"Stage: {state.stage}")
    print(f"Consent given: {state.consent_given}")
    print(f"Authenticated: {state.authenticated}")
    if state.authenticated:
        print(f"DOB: {state.user_dob}")
        print(f"Aadhaar last 4: {state.user_aadhar_digits}")
    print(f"\nTotal messages exchanged: {len(state.conversation_history)}")


if __name__ == "__main__":
    run_conversation()
