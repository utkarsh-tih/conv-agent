"""
Complete Hard Guardrails Implementation
All places where changes are needed for production compliance
"""

from typing import Literal
from datetime import datetime

# ============= 1. ADD COMPLIANCE EXCEPTIONS =============

class ComplianceViolation(Exception):
    """Raised when compliance rules are violated"""
    def __init__(self, message: str, violation_type: str, state: dict = None):
        self.message = message
        self.violation_type = violation_type
        self.state = state
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)


class SecurityViolation(Exception):
    """Raised when security rules are violated"""
    pass


# ============= 2. ADD COMPLIANCE VALIDATOR CLASS =============

class ComplianceValidator:
    """Centralized compliance validation logic"""
    
    @staticmethod
    def validate_tool_call(tool_name: str, state: dict) -> None:
        """Validate if tool can be called in current state"""
        
        # verify_consent can only be called in consent stage
        if tool_name == "verify_consent":
            if state["stage"] != "consent":
                raise ComplianceViolation(
                    f"verify_consent can only be called in consent stage, not '{state['stage']}'",
                    violation_type="WRONG_STAGE_TOOL_CALL"
                )
        
        # verify_identity requires consent first
        if tool_name == "verify_identity":
            if not state.get("consent_given"):
                raise ComplianceViolation(
                    "Cannot verify identity without customer consent",
                    violation_type="NO_CONSENT"
                )
            if state["stage"] != "identity_verification":
                raise ComplianceViolation(
                    f"verify_identity can only be called in identity_verification stage, not '{state['stage']}'",
                    violation_type="WRONG_STAGE_TOOL_CALL"
                )
        
        # extract_verification_data requires both consent and identity
        if tool_name == "extract_verification_data":
            if not state.get("consent_given"):
                raise ComplianceViolation(
                    "Cannot extract verification data without consent",
                    violation_type="NO_CONSENT"
                )
            if not state.get("identity_verified"):
                raise SecurityViolation(
                    "Cannot extract verification data without identity verification"
                )
            if state["stage"] != "questions":
                raise ComplianceViolation(
                    f"extract_verification_data can only be called in questions stage, not '{state['stage']}'",
                    violation_type="WRONG_STAGE_TOOL_CALL"
                )
        
        # save_loan_verification can only be called at the end
        if tool_name == "save_loan_verification":
            if state["stage"] != "closing":
                raise ComplianceViolation(
                    f"save_loan_verification can only be called in closing stage, not '{state['stage']}'",
                    violation_type="PREMATURE_SAVE"
                )
            if len(state.get("questions_completed", [])) < 19:
                raise ComplianceViolation(
                    f"Cannot save with only {len(state.get('questions_completed', []))} questions completed (need 19)",
                    violation_type="INCOMPLETE_QUESTIONS"
                )
    
    @staticmethod
    def validate_stage_transition(from_stage: str, to_stage: str, state: dict) -> None:
        """Validate stage transitions are legal"""
        
        valid_transitions = {
            "init": ["greeting"],
            "greeting": ["consent"],
            "consent": ["identity_verification", "closing"],  # closing if consent denied
            "identity_verification": ["questions", "closing"],  # closing if identity fails
            "questions": ["closing"],
            "closing": ["end"],
        }
        
        if to_stage not in valid_transitions.get(from_stage, []):
            raise ComplianceViolation(
                f"Invalid transition from '{from_stage}' to '{to_stage}'",
                violation_type="INVALID_TRANSITION"
            )
        
        # Validate required conditions for each transition
        if to_stage == "identity_verification":
            if not state.get("consent_given"):
                raise ComplianceViolation(
                    "Cannot transition to identity_verification without consent",
                    violation_type="NO_CONSENT"
                )
        
        if to_stage == "questions":
            if not state.get("identity_verified"):
                raise SecurityViolation(
                    "Cannot transition to questions without identity verification"
                )
            if not state.get("consent_given"):
                raise ComplianceViolation(
                    "Cannot transition to questions without consent",
                    violation_type="NO_CONSENT"
                )
        
        if to_stage == "closing":
            # Only allow early closing if consent denied or identity failed
            if state["stage"] in ["consent", "identity_verification"]:
                if state.get("consent_given") and state.get("identity_verified"):
                    raise ComplianceViolation(
                        "Cannot close early when consent and identity are verified",
                        violation_type="PREMATURE_CLOSING"
                    )
    
    @staticmethod
    def validate_tool_usage_history(state: dict) -> dict:
        """Validate that required tools were actually called"""
        
        tool_calls = []
        for msg in state["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc["name"])
        
        violations = []
        
        # If consent_given=True, verify_consent must have been called
        if state.get("consent_given") and "verify_consent" not in tool_calls:
            violations.append({
                "type": "MISSING_TOOL_CALL",
                "message": "consent_given=True but verify_consent was never called",
                "severity": "CRITICAL"
            })
        
        # If identity_verified=True, verify_identity must have been called
        if state.get("identity_verified") and "verify_identity" not in tool_calls:
            violations.append({
                "type": "MISSING_TOOL_CALL",
                "message": "identity_verified=True but verify_identity was never called",
                "severity": "CRITICAL"
            })
        
        # If questions completed, extract_verification_data must have been called
        completed_count = len(state.get("questions_completed", []))
        extract_count = tool_calls.count("extract_verification_data")
        if completed_count > 0 and extract_count == 0:
            violations.append({
                "type": "MISSING_TOOL_CALL",
                "message": f"{completed_count} questions marked complete but extract_verification_data never called",
                "severity": "CRITICAL"
            })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "tool_call_history": tool_calls
        }


# ============= 3. ENHANCED tool_execution_node WITH VALIDATION =============

def tool_execution_node_with_guardrails(state: LoanVerificationState) -> LoanVerificationState:
    """Execute tools with strict compliance validation"""
    print("[TOOL EXECUTION NODE WITH GUARDRAILS]")
    
    last_message = state["messages"][-1]
    tool_messages = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  → Attempting to execute: {tool_name}")
            
            try:
                # ========== PRE-EXECUTION VALIDATION ==========
                ComplianceValidator.validate_tool_call(tool_name, state)
                
                # Find and execute tool
                tool_func = next((t for t in tools if t.name == tool_name), None)
                if not tool_func:
                    raise ValueError(f"Tool '{tool_name}' not found")
                
                # Inject language if needed
                if tool_name in ["speak_to_customer", "listen_to_customer"]:
                    if "language" not in tool_args:
                        tool_args["language"] = state["language"]
                
                result = tool_func.invoke(tool_args)
                
                # ========== POST-EXECUTION VALIDATION & STATE UPDATE ==========
                
                if tool_name == "speak_to_customer":
                    audio_path = result.get("audio_path", "")
                    if audio_path:
                        state["audio_paths"].append(audio_path)
                    
                    state["transcript"].append({
                        "speaker": "agent",
                        "text": tool_args.get("text", ""),
                        "timestamp": datetime.now().isoformat(),
                        "audio_path": audio_path,
                    })
                
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
                        state["transcript"].append({
                            "speaker": "customer",
                            "text": customer_text,
                            "confidence": confidence,
                            "quality": quality,
                            "timestamp": datetime.now().isoformat(),
                            "audio_path": audio_path,
                        })
                        state["messages"].append(HumanMessage(content=customer_text))
                
                elif tool_name == "get_loan_applicant_data":
                    state["applicant_data"] = result
                    state["language"] = result.get("language_preference", "english")
                
                elif tool_name == "verify_consent":
                    # STRICT VALIDATION: Must have clear status
                    status = result.get("status")
                    if status not in ["approved", "rejected", "unclear"]:
                        raise ComplianceViolation(
                            f"verify_consent returned invalid status: {status}",
                            violation_type="INVALID_TOOL_RESULT"
                        )
                    
                    if status == "approved":
                        state["consent_given"] = True
                        print("  ✓ Consent granted")
                    elif status == "rejected":
                        state["consent_given"] = False
                        state["stage"] = "closing"  # Must terminate
                        print("  ✗ Consent explicitly rejected - ending call")
                    else:
                        state["consent_given"] = False
                        print("  ⚠ Consent unclear - retry required")
                
                elif tool_name == "verify_identity":
                    # STRICT VALIDATION: Must have verified field
                    if "verified" not in result:
                        raise ComplianceViolation(
                            "verify_identity must return 'verified' field",
                            violation_type="INVALID_TOOL_RESULT"
                        )
                    
                    if result.get("verified"):
                        state["identity_verified"] = True
                        state["identity_retry_count"] = 0
                        print("  ✓ Identity verified")
                    else:
                        state["identity_retry_count"] += 1
                        print(f"  ✗ Identity verification failed (attempt {state['identity_retry_count']}/3)")
                        
                        # AUTO-TERMINATE after 3 failures (security requirement)
                        if state["identity_retry_count"] >= 3:
                            state["stage"] = "closing"
                            print("  🚨 SECURITY: Max identity attempts exceeded - terminating")
                
                elif tool_name == "extract_verification_data":
                    # STRICT VALIDATION: Must have required fields
                    if "updated_data" not in result:
                        raise ComplianceViolation(
                            "extract_verification_data must return 'updated_data'",
                            violation_type="INVALID_TOOL_RESULT"
                        )
                    
                    state["verification_data"] = result.get("updated_data", state["verification_data"])
                    
                    # Mark question as completed
                    q_id = tool_args.get("question_id", "")
                    if q_id and q_id not in state["questions_completed"]:
                        state["questions_completed"].append(q_id)
                        print(f"  ✓ Question {q_id} completed ({len(state['questions_completed'])}/19)")
                    
                    # Move to next question
                    current_q_num = int(state.get("current_question_id", "1"))
                    next_q_num = current_q_num + 1
                    
                    # Skip question 13
                    if next_q_num == 13:
                        next_q_num = 14
                    
                    if next_q_num <= 20:
                        state["current_question_id"] = str(next_q_num)
                    else:
                        # All questions done - validate before transitioning
                        if len(state["questions_completed"]) < 19:
                            raise ComplianceViolation(
                                f"Only {len(state['questions_completed'])} questions completed, need 19",
                                violation_type="INCOMPLETE_QUESTIONS"
                            )
                        state["stage"] = "closing"
                        print("  ✓ All questions completed - moving to closing")
                
                elif tool_name == "save_loan_verification":
                    # STRICT VALIDATION: Must succeed
                    if result.get("status") != "success":
                        raise ComplianceViolation(
                            f"Failed to save verification data: {result.get('error')}",
                            violation_type="SAVE_FAILED"
                        )
                    print("  ✓ Verification data saved to database")
                
                tool_msg = ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_msg)
                
            except ComplianceViolation as e:
                print(f"  ⛔ COMPLIANCE VIOLATION: {e.message}")
                # Log violation
                error_msg = ToolMessage(
                    content=json.dumps({
                        "error": "ComplianceViolation",
                        "message": e.message,
                        "type": e.violation_type
                    }),
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(error_msg)
                # Force agent to retry correctly
                
            except SecurityViolation as e:
                print(f"  🚨 SECURITY VIOLATION: {str(e)}")
                # Security violations are fatal - terminate immediately
                state["stage"] = "closing"
                error_msg = ToolMessage(
                    content=json.dumps({
                        "error": "SecurityViolation",
                        "message": str(e),
                        "terminating": True
                    }),
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(error_msg)
    
    state["messages"].extend(tool_messages)
    state["turn_count"] += 1
    
    return state


# ============= 4. ENHANCED should_continue WITH VALIDATION =============

def should_continue_with_guardrails(state: LoanVerificationState) -> Literal["agent", "tools", "end"]:
    """Enhanced routing logic with strict compliance checks"""
    
    last_message = state["messages"][-1]
    
    # If LLM called tools, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # ========== VALIDATE TOOL USAGE HISTORY ==========
    validation_result = ComplianceValidator.validate_tool_usage_history(state)
    if not validation_result["valid"]:
        print("\n⛔ COMPLIANCE AUDIT FAILED:")
        for violation in validation_result["violations"]:
            print(f"  - {violation['severity']}: {violation['message']}")
        
        # For critical violations, force correction
        if any(v["severity"] == "CRITICAL" for v in validation_result["violations"]):
            # Reset invalid states
            if state.get("consent_given") and "verify_consent" not in validation_result["tool_call_history"]:
                print("  → Resetting consent_given to False")
                state["consent_given"] = False
            
            if state.get("identity_verified") and "verify_identity" not in validation_result["tool_call_history"]:
                print("  → Resetting identity_verified to False")
                state["identity_verified"] = False
    
    # ========== STAGE TRANSITION LOGIC WITH VALIDATION ==========
    
    current_stage = state["stage"]
    
    if current_stage == "init" and state.get("applicant_data"):
        try:
            ComplianceValidator.validate_stage_transition("init", "greeting", state)
            state["stage"] = "greeting"
            return "agent"
        except ComplianceViolation as e:
            print(f"⛔ Cannot transition to greeting: {e.message}")
            return "agent"
    
    elif current_stage == "greeting" and state["turn_count"] >= 2:
        # VALIDATE: Must have customer response
        customer_responses = [t for t in state["transcript"] if t.get("speaker") == "customer"]
        if not customer_responses:
            print("⛔ Cannot proceed: No customer response detected")
            return "agent"
        
        try:
            ComplianceValidator.validate_stage_transition("greeting", "consent", state)
            state["stage"] = "consent"
            return "agent"
        except ComplianceViolation as e:
            print(f"⛔ Cannot transition to consent: {e.message}")
            return "agent"
    
    elif current_stage == "consent":
        if state.get("consent_given"):
            try:
                ComplianceValidator.validate_stage_transition("consent", "identity_verification", state)
                state["stage"] = "identity_verification"
                return "agent"
            except ComplianceViolation as e:
                print(f"⛔ Cannot transition to identity verification: {e.message}")
                return "agent"
        elif state["turn_count"] >= 8:
            # Too many attempts without consent - close call
            print("⚠ Consent not obtained after 8 turns - closing call")
            state["stage"] = "closing"
            return "agent"
    
    elif current_stage == "identity_verification":
        if state.get("identity_verified"):
            try:
                ComplianceValidator.validate_stage_transition("identity_verification", "questions", state)
                state["stage"] = "questions"
                state["current_question_id"] = "1"
                return "agent"
            except ComplianceViolation as e:
                print(f"⛔ Cannot transition to questions: {e.message}")
                return "agent"
        elif state.get("identity_retry_count", 0) >= 3:
            print("🚨 Identity verification failed after 3 attempts - closing call")
            state["stage"] = "closing"
            return "agent"
    
    elif current_stage == "questions":
        # STRICT CHECK: All questions must be completed
        completed_count = len(state.get("questions_completed", []))
        if completed_count >= 19:
            # VALIDATE: Must have verification data for all questions
            verification_data = state.get("verification_data", {})
            if len(verification_data) < 19:
                print(f"⛔ Only {len(verification_data)} questions have data, need 19")
                return "agent"
            
            try:
                ComplianceValidator.validate_stage_transition("questions", "closing", state)
                state["stage"] = "closing"
                return "agent"
            except ComplianceViolation as e:
                print(f"⛔ Cannot transition to closing: {e.message}")
                return "agent"
    
    elif current_stage == "closing":
        # STRICT CHECK: Must have saved to database
        save_tool_called = any(
            hasattr(msg, "tool_calls") and 
            any(tc["name"] == "save_loan_verification" for tc in msg.tool_calls)
            for msg in state["messages"]
        )
        
        if save_tool_called:
            print("✅ All compliance checks passed - ending call")
            return "end"
        else:
            # Force save before ending
            if state["turn_count"] >= state.get("_last_closing_turn", 0) + 5:
                print("⛔ Closing stage timeout - forcing end (data may not be saved)")
                return "end"
            state["_last_closing_turn"] = state["turn_count"]
            return "agent"
    
    return "agent"


# ============= 5. ENHANCED run_loan_verification_call WITH EXCEPTION HANDLING =============

def run_loan_verification_call_with_guardrails(customer_id: str, initial_language: str = "english"):
    """Execute loan verification with full compliance enforcement"""
    
    print("\n" + "=" * 70)
    print("🏦 STARTING LOAN VERIFICATION WITH STRICT COMPLIANCE")
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
        "_last_closing_turn": 0,
    }
    
    initial_state["messages"].append(
        HumanMessage(content=f"Start loan verification call with customer ID: {customer_id}")
    )
    
    try:
        # Build graph with guardrail nodes
        workflow = StateGraph(LoanVerificationState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_execution_node_with_guardrails)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue_with_guardrails,
            {"agent": "agent", "tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        
        app = workflow.compile()
        final_state = app.invoke(initial_state)
        
        # ========== FINAL COMPLIANCE AUDIT ==========
        print("\n" + "=" * 70)
        print("📋 FINAL COMPLIANCE AUDIT")
        print("=" * 70)
        
        audit_result = ComplianceValidator.validate_tool_usage_history(final_state)
        
        if audit_result["valid"]:
            print("✅ All compliance checks PASSED")
        else:
            print("⛔ COMPLIANCE VIOLATIONS DETECTED:")
            for violation in audit_result["violations"]:
                print(f"  - [{violation['severity']}] {violation['message']}")
        
        # Determine final verification status
        if (
            final_state.get("consent_given") and
            final_state.get("identity_verified") and
            len(final_state.get("questions_completed", [])) >= 19 and
            audit_result["valid"]
        ):
            verification_status = "completed"
        elif final_state.get("identity_verified") and len(final_state.get("questions_completed", [])) > 0:
            verification_status = "partial"
        else:
            verification_status = "failed"
        
        print(f"\nFinal Status: {verification_status.upper()}")
        
        return final_state
        
    except ComplianceViolation as e:
        print(f"\n🚨 CALL TERMINATED DUE TO COMPLIANCE VIOLATION")
        print(f"   Type: {e.violation_type}")
        print(f"   Message: {e.message}")
        print(f"   Time: {e.timestamp}")
        raise
    
    except SecurityViolation as e:
        print(f"\n🚨 CALL TERMINATED DUE TO SECURITY VIOLATION")
        print(f"   Message: {str(e)}")
        raise
    
    except Exception as e:
        print(f"\n❌ CALL FAILED DUE TO UNEXPECTED ERROR")
        print(f"   Error: {str(e)}")
        raise


# ============= SUMMARY OF CHANGES NEEDED =============

"""
COMPLETE LIST OF CHANGES FOR STRICT GUARDRAILS:

1. ✅ Add exception classes (ComplianceViolation, SecurityViolation)
   - Location: Top of file after imports

2. ✅ Add ComplianceValidator class
   - Location: Before node functions
   - Methods: validate_tool_call, validate_stage_transition, validate_tool_usage_history

3. ✅ Replace tool_execution_node with tool_execution_node_with_guardrails
   - Location: Node functions section
   - Changes: Pre-execution validation, post-execution validation, exception handling

4. ✅ Replace should_continue with should_continue_with_guardrails
   - Location: Routing logic section
   - Changes: Tool history validation, strict stage transition checks

5. ✅ Replace run_loan_verification_call with run_loan_verification_call_with_guardrails
   - Location: Main execution section
   - Changes: Use guardrail nodes, final compliance audit, exception handling

6. ✅ Update graph construction in main execution
   - Location: run_loan_verification_call function
   - Changes: Use tool_execution_node_with_guardrails and should_continue_with_guardrails

TOTAL: 6 major changes across multiple functions
NOT just a single snippet!
"""
