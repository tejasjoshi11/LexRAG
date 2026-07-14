from src.llm.groq_provider import GroqProvider
from src.llm.models import GROQ_QWEN_32B


def main() -> None:
    provider = GroqProvider(
        model=GROQ_QWEN_32B,
    )

    response = provider.generate(
        system_prompt="You are a legal assistant.",
        user_prompt="What is judicial review?",
    )

    print("=" * 80)
    print("LLM RESPONSE OBJECT")
    print("=" * 80)
    print(response)

    print()
    print("=" * 80)
    print("GROQ PROVIDER TEST")
    print("=" * 80)
    print(f"Provider          : {response.provider}")
    print(f"Model             : {response.model}")
    print(f"Latency (ms)      : {response.latency_ms:.2f}")
    print(f"Prompt Tokens     : {response.prompt_tokens}")
    print(f"Completion Tokens : {response.completion_tokens}")
    print(f"Reasoning Tokens  : {response.reasoning_tokens}")
    print(f"Total Tokens      : {response.total_tokens}")
    print(f"Finish Reason     : {response.finish_reason.value}")

    print()
    print("Response")
    print("-" * 80)
    print(response.text)

    print()
    print("=" * 80)
    print("Groq provider test completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()