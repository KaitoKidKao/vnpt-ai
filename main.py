"""Entry point for running the RAG pipeline on test data."""

import csv
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import DATA_INPUT_DIR, DATA_OUTPUT_DIR
from src.graph import GraphState, get_graph
from src.utils.ingestion import ingest_knowledge_base


class QuestionInput(BaseModel):
    """Input schema for a multiple-choice question."""

    id: str = Field(description="Question identifier")
    question: str = Field(description="Question text in Vietnamese")
    A: str = Field(description="Option A")
    B: str = Field(description="Option B")
    C: str = Field(description="Option C")
    D: str = Field(description="Option D")
    category: str | None = Field(default=None, description="Question category")


class PredictionOutput(BaseModel):
    """Output schema for a prediction."""

    id: str = Field(description="Question identifier")
    answer: str = Field(description="Predicted answer: A, B, C, or D")


def load_test_data(file_path: Path) -> list[QuestionInput]:
    """Load test questions from CSV file."""
    questions = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(QuestionInput(**row))
    return questions


def run_pipeline(questions: list[QuestionInput], force_reingest: bool = False) -> list[PredictionOutput]:
    """Run the RAG pipeline on a list of questions.
    
    Args:
        questions: List of questions to process.
        force_reingest: If True, force re-ingestion of knowledge base. Defaults to False.
    """
    print("[Pipeline] Initializing knowledge base...")
    ingest_knowledge_base(force=force_reingest)

    graph = get_graph()
    predictions = []

    for i, q in enumerate(questions, 1):
        print(f"\n[Pipeline] Processing question {i}/{len(questions)}: {q.id}")
        print(f"  Question: {q.question}")
        print(f"  A. {q.A}")
        print(f"  B. {q.B}")
        print(f"  C. {q.C}")
        print(f"  D. {q.D}")

        state: GraphState = {
            "question_id": q.id,
            "question": q.question,
            "option_a": q.A,
            "option_b": q.B,
            "option_c": q.C,
            "option_d": q.D,
        }

        result = graph.invoke(state)

        answer = result.get("answer", "A")
        if answer not in ["A", "B", "C", "D"]:
            answer = "A"

        route = result.get("route", "unknown")
        print(f"  Route: {route}")
        print(f"  Answer: {answer}")

        predictions.append(PredictionOutput(id=q.id, answer=answer))

    return predictions


def save_predictions(predictions: list[PredictionOutput], output_path: Path) -> None:
    """Save predictions to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        for pred in predictions:
            writer.writerow({"id": pred.id, "answer": pred.answer})
    print(f"\n[Pipeline] Predictions saved to: {output_path}")


def main() -> None:
    """Main entry point."""
    input_file = DATA_INPUT_DIR / "private_test.csv"
    if not input_file.exists():
        input_file = DATA_INPUT_DIR / "public_test.csv"

    if not input_file.exists():
        print("[Main] Test file not found. Generating dummy data...")
        from scripts.generate_data import generate_knowledge_base
        generate_knowledge_base()

    print(f"[Main] Loading test data from: {input_file}")
    questions = load_test_data(input_file)
    print(f"[Main] Loaded {len(questions)} questions")

    predictions = run_pipeline(questions)

    output_file = DATA_OUTPUT_DIR / "pred.csv"
    save_predictions(predictions, output_file)

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for pred in predictions:
        print(f"  {pred.id}: {pred.answer}")


if __name__ == "__main__":
    main()
