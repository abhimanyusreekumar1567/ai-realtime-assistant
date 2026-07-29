from agent import RealtimeAgent


def main():
    print("🤖 AI Real-Time Assistant (type 'exit' to quit)\n")

    try:
        agent = RealtimeAgent()
    except Exception as e:
        print("Startup error:")
        print(e)
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("\nAssistant:", end=" ")
        response = agent.process(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()