from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from dialog_emo_models.models import EmotionModel
from dialog_emo_models.pipeline import parse_telegram_json, score_parsed_frame
from dialog_emo_models.registry import available_model_names, create_model
from dialog_emo_models.schema import load_parsed_csv, write_full_csv, write_parsed_csv
from dialog_emo_models.training import available_trainable_model_names, train_from_full_csv


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialog-emo",
        description="Parse and score dialogue emotion datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse Telegram JSON into parsed CSV")
    parse_parser.add_argument("--input", required=True, type=Path, help="Telegram result.json")
    parse_parser.add_argument("--output", required=True, type=Path, help="Parsed CSV path")
    parse_parser.set_defaults(func=_parse_command)

    score_parser = subparsers.add_parser("score", help="Score parsed CSV into full CSV")
    score_parser.add_argument("--input", required=True, type=Path, help="Parsed CSV path")
    score_parser.add_argument("--output", required=True, type=Path, help="Full scored CSV path")
    score_parser.add_argument(
        "--model",
        default="dummy",
        choices=available_model_names(),
        help="Registered model name",
    )
    score_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a saved joblib EmotionModel. Overrides --model.",
    )
    score_parser.set_defaults(func=_score_command)

    run_parser = subparsers.add_parser("run", help="Parse Telegram JSON and score it")
    run_parser.add_argument("--input", required=True, type=Path, help="Telegram result.json")
    run_parser.add_argument("--output", required=True, type=Path, help="Full scored CSV path")
    run_parser.add_argument(
        "--parsed-output",
        type=Path,
        default=None,
        help="Optional parsed CSV path",
    )
    run_parser.add_argument(
        "--model",
        default="dummy",
        choices=available_model_names(),
        help="Registered model name",
    )
    run_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a saved joblib EmotionModel. Overrides --model.",
    )
    run_parser.set_defaults(func=_run_command)

    train_parser = subparsers.add_parser("train", help="Train a model from full CSV")
    train_parser.add_argument("--input", required=True, type=Path, help="Full scored CSV path")
    train_parser.add_argument("--output", required=True, type=Path, help="Saved model path")
    train_parser.add_argument(
        "--model",
        default="ridge-tfidf",
        choices=available_trainable_model_names(),
        help="Trainable model name",
    )
    train_parser.set_defaults(func=_train_command)

    return parser


def _parse_command(args: argparse.Namespace) -> None:
    parsed = parse_telegram_json(args.input)
    output = write_parsed_csv(parsed, args.output)
    print(f"parsed {len(parsed)} rows -> {output}")


def _score_command(args: argparse.Namespace) -> None:
    parsed = load_parsed_csv(args.input)
    model = _load_model(args.model, args.model_path)
    scored = score_parsed_frame(parsed, model, show_progress=True)
    output = write_full_csv(scored, args.output)
    print(f"scored {len(scored)} rows with {_model_label(args)} -> {output}")


def _run_command(args: argparse.Namespace) -> None:
    parsed = parse_telegram_json(args.input)
    if args.parsed_output is not None:
        parsed_output = write_parsed_csv(parsed, args.parsed_output)
        print(f"parsed {len(parsed)} rows -> {parsed_output}")

    model = _load_model(args.model, args.model_path)
    scored = score_parsed_frame(parsed, model, show_progress=True)
    output = write_full_csv(scored, args.output)
    print(f"scored {len(scored)} rows with {_model_label(args)} -> {output}")


def _train_command(args: argparse.Namespace) -> None:
    train_from_full_csv(args.input, model_name=args.model, output_path=args.output)
    print(f"trained {args.model!r} -> {args.output}")


def _load_model(model_name: str, model_path: Path | None) -> EmotionModel:
    if model_path is None:
        return create_model(model_name)
    model = joblib.load(model_path)
    if not isinstance(model, EmotionModel):
        raise TypeError(f"Expected saved EmotionModel, got {type(model).__name__}")
    return model


def _model_label(args: argparse.Namespace) -> str:
    if args.model_path is not None:
        return str(args.model_path)
    return repr(args.model)
