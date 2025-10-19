"""
Deep test: Does LangGraph detect mutations INSIDE nodes?
Testing your hypothesis that changes between nodes trigger reducers
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node 1: Sets initial messages
def node1_setup(state: MessagesState) -> dict:
    """Sets up initial messages properly"""
    print("\n[NODE 1] Setting up initial messages")
    return {
        "messages": [
            HumanMessage(content="Message from Node 1 - Part A"),
            AIMessage(content="Message from Node 1 - Part B")
        ]
    }

# Node 2: MUTATES state directly (your approach)
def node2_mutate(state: MessagesState) -> MessagesState:
    """Mutates state directly and returns full state"""
    print(f"\n[NODE 2] Received {len(state['messages'])} messages")
    for i, msg in enumerate(state['messages'], 1):
        print(f"  {i}. {msg.content}")
    
    print("[NODE 2] Directly assigning new messages...")
    state["messages"] = [AIMessage(content="Message from Node 2 - MUTATED")]
    
    print(f"[NODE 2] After mutation: {len(state['messages'])} messages")
    print(f"  1. {state['messages'][0].content}")
    
    return state

# Node 3: Returns dict properly
def node3_proper(state: MessagesState) -> dict:
    """Returns dict with new messages (correct way)"""
    print(f"\n[NODE 3] Received {len(state['messages'])} messages")
    for i, msg in enumerate(state['messages'], 1):
        print(f"  {i}. {msg.content}")
    
    print("[NODE 3] Returning dict with new message...")
    return {
        "messages": [HumanMessage(content="Message from Node 3 - PROPER")]
    }

# Build graph with 3 nodes
print("=" * 70)
print("TESTING: Does mutation inside node + return state work?")
print("=" * 70)

graph = StateGraph(MessagesState)
graph.add_node("setup", node1_setup)
graph.add_node("mutate", node2_mutate)
graph.add_node("proper", node3_proper)

graph.add_edge(START, "setup")
graph.add_edge("setup", "mutate")
graph.add_edge("mutate", "proper")
graph.add_edge("proper", END)

app = graph.compile()

# Run the graph
initial_state = {
    "messages": [HumanMessage(content="Initial message before any nodes")]
}

print("\n[INITIAL STATE] Starting with:")
print(f"  1. {initial_state['messages'][0].content}")

result = app.invoke(initial_state)

print("\n" + "=" * 70)
print("FINAL STATE:")
print("=" * 70)
print(f"Total messages: {len(result['messages'])}")
for i, msg in enumerate(result['messages'], 1):
    print(f"  {i}. {msg.__class__.__name__}: {msg.content}")

print("\n" + "=" * 70)
print("ANALYSIS:")
print("=" * 70)

expected_if_reducer_works = [
    "Initial message before any nodes",
    "Message from Node 1 - Part A", 
    "Message from Node 1 - Part B",
    "Message from Node 2 - MUTATED",
    "Message from Node 3 - PROPER"
]

expected_if_mutation_replaces = [
    "Message from Node 2 - MUTATED",
    "Message from Node 3 - PROPER"
]

actual_messages = [msg.content for msg in result['messages']]

if actual_messages == expected_if_reducer_works:
    print("✅ HYPOTHESIS CONFIRMED: Mutation + return state DOES trigger reducer!")
    print("   LangGraph detected the change between nodes and applied add_messages")
elif actual_messages == expected_if_mutation_replaces:
    print("❌ HYPOTHESIS REJECTED: Mutation + return state REPLACES messages!")
    print("   Direct mutation bypassed add_messages reducer")
else:
    print("🤔 UNEXPECTED RESULT:")
    print(f"   Got {len(actual_messages)} messages:")
    for msg in actual_messages:
        print(f"     - {msg}")

print("=" * 70)
