"""
chat_cli.py
-----------
Command-line interface to talk to the fine-tuned chatbot.

Run:
    python chat_cli.py
Type 'exit' to quit.
"""

from src.pipeline.prediction_pipeline import PredictionPipeline


def main():
    print("=" * 60)
    print(" Mental Health Support Chatbot (type 'exit' to quit) ")
    print("=" * 60)
    print(
        "Note: This is a student/demo project, not a medical device.\n"
        "If you are in crisis, please contact a real crisis line or\n"
        "emergency services.\n"
    )

    pipeline = PredictionPipeline()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Take care of yourself. Goodbye for now. 💙")
            break
        if not user_input:
            continue

        reply = pipeline.generate_reply(user_input)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
