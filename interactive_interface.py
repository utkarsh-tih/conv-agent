"""
Interactive Terminal Interface for Loan Verification System
Provides a clean, user-friendly way to run verification calls
"""

import os
import sys
from datetime import datetime

# Set text mode by default
os.environ["TEXT_MODE"] = "true"

# Suppress verbose output from libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

def print_header():
    """Print a nice header for the application"""
    print("\n" + "=" * 80)
    print("🏦  LOAN VERIFICATION TELECALLING SYSTEM - INTERACTIVE MODE")
    print("=" * 80)
    print("This is a text-based simulation of the verification call workflow")
    print("You'll be prompted to respond as the customer at each step")
    print("=" * 80 + "\n")

def print_section(title):
    """Print a section divider"""
    print("\n" + "-" * 80)
    print(f"📋 {title}")
    print("-" * 80)

def get_customer_info():
    """Get customer information from user"""
    print_section("SETUP")
    print("\nEnter customer details:")
    
    customer_id = input("Customer ID (or press Enter for 'CUST001'): ").strip()
    if not customer_id:
        customer_id = "CUST001"
    
    language = input("Language preference (english/hindi, or press Enter for 'english'): ").strip().lower()
    if not language:
        language = "english"
    
    return customer_id, language

def print_instructions():
    """Print instructions for the user"""
    print_section("HOW TO USE")
    print("""
When you see "Agent: [message]", that's what the telecaller is saying to you.
When you see "You (customer):", type your response as if you were the loan applicant.

TIPS:
- For identity verification, the system expects:
  * DOB: 15/06/1985 (or 15061985)
  * Aadhar last 4 digits: 4567
  
- For consent: Say "yes" or "I consent" to proceed

- For verification questions: Provide detailed, realistic answers
  Example: "I live at 123 MG Road, Bangalore. I've been here for 3 years..."

- To exit anytime: Press Ctrl+C
""")

def run_verification_session():
    """Main function to run interactive verification"""
    print_header()
    print_instructions()
    
    # Get customer info
    customer_id, language = get_customer_info()
    
    print(f"\n✓ Starting verification call for Customer: {customer_id}")
    print(f"✓ Language: {language}")
    print(f"✓ Text mode enabled - No audio processing required")
    
    input("\nPress Enter to begin the verification call...")
    
    # Import the main module (this will trigger some initialization messages)
    print("\n" + "=" * 80)
    print("Initializing verification system...")
    print("=" * 80)
    
    try:
        # Import after setting environment variables
        from spike7_questionnaire_textmode import run_loan_verification_call
        
        # Run the actual verification
        print("\n" + "=" * 80)
        print("🚀 STARTING VERIFICATION CALL")
        print("=" * 80)
        
        final_state = run_loan_verification_call(
            customer_id=customer_id, 
            initial_language=language
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 CALL SUMMARY")
        print("=" * 80)
        print(f"Status: {final_state.get('stage', 'unknown')}")
        print(f"Questions answered: {len(final_state.get('questions_completed', []))}/19")
        print(f"Consent given: {'✓' if final_state.get('consent_given') else '✗'}")
        print(f"Identity verified: {'✓' if final_state.get('identity_verified') else '✗'}")
        
        return final_state
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Call interrupted by user")
        print("=" * 80)
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during verification: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Entry point with menu"""
    while True:
        print("\n" + "=" * 80)
        print("🏦  LOAN VERIFICATION SYSTEM - MAIN MENU")
        print("=" * 80)
        print("\nOptions:")
        print("  1. Start New Verification Call")
        print("  2. View Recent Calls")
        print("  3. Generate Report")
        print("  4. Export Call Data")
        print("  5. Exit")
        print()
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            run_verification_session()
            input("\nPress Enter to return to main menu...")
            
        elif choice == "2":
            print_section("RECENT CALLS")
            try:
                from spike7_questionnaire_textmode import list_recent_verifications
                list_recent_verifications(10)
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to return to main menu...")
            
        elif choice == "3":
            print_section("GENERATE REPORT")
            call_sid = input("Enter Call SID: ").strip()
            if call_sid:
                try:
                    from spike7_questionnaire_textmode import generate_verification_report
                    import json
                    report = generate_verification_report(call_sid)
                    print(json.dumps(report, indent=2))
                except Exception as e:
                    print(f"Error: {e}")
            input("\nPress Enter to return to main menu...")
            
        elif choice == "4":
            print_section("EXPORT TO CSV")
            call_sid = input("Enter Call SID: ").strip()
            output_file = input("Output filename (or press Enter for default): ").strip()
            if call_sid:
                try:
                    from spike7_questionnaire_textmode import export_verification_to_csv
                    export_verification_to_csv(call_sid, output_file or None)
                except Exception as e:
                    print(f"Error: {e}")
            input("\nPress Enter to return to main menu...")
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            print("=" * 80 + "\n")
            break
            
        else:
            print("❌ Invalid option. Please select 1-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        print("=" * 80 + "\n")
        sys.exit(0)
