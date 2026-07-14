"""Manual integration test for the Gemini provider."""

from __future__ import annotations

from src.llm.gemini_provider import GeminiProvider
from src.llm.models import GEMINI_FLASH_LITE

def main() -> None:
    """Run the Gemini provider integration test."""

    provider = GeminiProvider(
        model=GEMINI_FLASH_LITE,
    )

    response = provider.generate(
        system_prompt=(
            "You are a helpful assistant. "
            "Respond in one complete sentence."
        ),
        user_prompt="What is judicial review?",
        temperature=0.0,
        max_output_tokens=128,
    )

    print("\n" + "=" * 80)
    print("LLM RESPONSE OBJECT")
    print("=" * 80)
    print(response)

    print("\n" + "=" * 80)
    print("GEMINI PROVIDER TEST")
    print("=" * 80)

    print(f"Provider          : {response.provider}")
    print(f"Model             : {response.model}")
    print(f"Latency (ms)      : {response.latency_ms:.2f}")
    print(f"Prompt Tokens     : {response.prompt_tokens}")
    print(f"Completion Tokens : {response.completion_tokens}")
    print(f"Reasoning Tokens  : {response.reasoning_tokens}")
    print(f"Total Tokens      : {response.total_tokens}")
    print(f"Finish Reason     : {response.finish_reason.value}")

    print("\nResponse")
    print("-" * 80)
    print(response.text)

    print("\n" + "=" * 80)
    print("Gemini provider test completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()