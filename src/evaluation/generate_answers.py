from __future__ import annotations

from pathlib import Path

from src.shared.config import config
from src.contracts.runner_config import RunnerConfig
from .answer_generation_runner import (
    AnswerGenerationRunner,
)
from .generated_answer_writer import (
    GeneratedAnswerWriter,
)
from .golden_set_loader import (
    GoldenSetLoader,
)
from src.llm.models import SUPPORTED_MODELS
from src.main import create_pipeline


def _resolve_model(model_id: str):
    """Resolve a configured model id."""

    for model in SUPPORTED_MODELS:
        if model.id == model_id:
            return model

    raise ValueError(f"Unsupported model: {model_id}")


def main() -> None:
    application_config = config()

    evaluation = application_config["evaluation"]

    model = _resolve_model(
        evaluation["model"]
    )

    pipeline = create_pipeline()

    loader = GoldenSetLoader()

    output_path = Path(evaluation["generated_answers_path"])
    writer = GeneratedAnswerWriter()

    runner_config = RunnerConfig(
        checkpoint_interval=int(evaluation.get("checkpoint_interval", 10)),
        max_consecutive_failures=int(evaluation.get("max_consecutive_failures", 3)),
    )

    runner = AnswerGenerationRunner(
        pipeline=pipeline,
        model=model,
        top_k=evaluation["top_k"],
        writer=writer,
        output_path=output_path,
        config=runner_config,
    )

    print("Loading Golden Set...")
    golden_set = loader.load(
        Path(evaluation["golden_set_path"])
    )
    print(f"Loaded {len(golden_set)} questions.")

    existing_answers = writer.read(output_path)
    if existing_answers:
        print(f"Resuming with {len(existing_answers)} existing answers.")

    print("Generating answers...")
    generated_answers = runner.run(
        golden_set=golden_set,
        existing_answers=existing_answers,
    )
    print("Answer generation finished.")

    print("Writing Excel...")
    writer.write(
        output_path,
        generated_answers,
    )
    print("Done.")


if __name__ == "__main__":
    main()