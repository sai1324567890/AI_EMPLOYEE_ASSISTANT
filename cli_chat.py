"""
Minimal terminal chat loop — useful for quick testing without Streamlit.
Maintains conversation memory across turns (ask a follow-up question and
the assistant will remember what you just talked about).

Usage:
    python cli_chat.py
    python cli_chat.py --retrieval faiss
    python cli_chat.py --router llm

Type 'new' to start a fresh conversation (clears memory), 'exit'/'quit' to leave.
"""
import argparse
import uuid

from src.assistant import TrainingAssistant


def main():
    parser = argparse.ArgumentParser(description="AI Training Assistant CLI chat.")
    parser.add_argument("--retrieval", choices=["tfidf", "faiss"], default=None)
    parser.add_argument("--router", choices=["rule_based", "llm"], default=None)
    args = parser.parse_args()

    print("AI Training Assistant (CLI mode). Type 'exit' to quit, 'new' to reset memory.\n")
    assistant = TrainingAssistant(retrieval_backend=args.retrieval, router_backend=args.router)
    session_id = str(uuid.uuid4())
    print(f"[llm backend: {assistant.llm.backend}] "
          f"[retrieval: {assistant.retrieval_backend}] "
          f"[router: {assistant.router_backend}]\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if question.lower() == "new":
            assistant.reset_conversation(session_id)
            print("(conversation memory cleared)\n")
            continue

        result = assistant.ask(question, session_id=session_id)
        print(f"\nAssistant [{result.route}, conf={result.routing_confidence:.2f}]:")
        print(result.answer)
        if result.citations:
            print(f"\nSources: {', '.join(result.citations)}")
        print()


if __name__ == "__main__":
    main()
