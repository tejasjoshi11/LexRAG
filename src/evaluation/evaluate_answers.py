from __future__ import annotations

from pathlib import Path

from src.shared.config import config
from src.contracts.runner_config import RunnerConfig
from src.evaluation.golden_set_loader import GoldenSetLoader
from src.evaluation.generated_answer_writer import GeneratedAnswerWriter
from src.evaluation.evaluation_pair_builder import EvaluationPairBuilder
from src.evaluation.deepeval_evaluator import DeepEvalEvaluator
from src.evaluation.evaluation_writer import EvaluationWriter
from src.evaluation.evaluation_runner import EvaluationRunner
from src.evaluation.summary_generator import SummaryGenerator


def main() -> None:
    application_config = config()
    evaluation = application_config.get("evaluation", {})

    print("Loading Golden Set...")
    loader = GoldenSetLoader()
    golden_set = loader.load(Path(evaluation["golden_set_path"]))
    print(f"Loaded {len(golden_set)} questions.")

    print("Loading Generated Answers...")
    answer_writer = GeneratedAnswerWriter()
    generated_answers = answer_writer.read(Path(evaluation["generated_answers_path"]))
    print(f"Loaded {len(generated_answers)} generated answers.")

    pair_builder = EvaluationPairBuilder()
    pairs = pair_builder.build(golden_set, generated_answers)

    output_path = Path(evaluation.get("evaluation_results_path", "evaluation_results.xlsx"))
    writer = EvaluationWriter()
    existing_results = writer.read(output_path)

    if existing_results:
        print(f"Resuming with {len(existing_results)} existing evaluation results.")

    judge_model = evaluation.get("judge_model", "gpt-4")
    threshold = float(evaluation.get("faithfulness_threshold", 0.5))

    evaluator = DeepEvalEvaluator(
        judge_model=judge_model,
        threshold=threshold,
    )

    runner_config = RunnerConfig(
        checkpoint_interval=int(evaluation.get("checkpoint_interval", 10)),
        max_consecutive_failures=int(evaluation.get("max_consecutive_failures", 3)),
    )

    runner = EvaluationRunner(
        evaluator=evaluator,
        writer=writer,
        output_path=output_path,
        config=runner_config,
    )

    results = runner.run(
        evaluation_pairs=pairs,
        existing_results=existing_results,
    )

    print("Writing evaluation results...")
    writer.write(output_path, results)

    print("Generating summary...")
    summary_generator = SummaryGenerator()
    summary_path = Path(evaluation.get("evaluation_summary_path", "artifacts/evaluation/evaluation_summary.json"))
    summary_generator.generate(results, summary_path)

    print("Done.")


if __name__ == "__main__":
    main()
