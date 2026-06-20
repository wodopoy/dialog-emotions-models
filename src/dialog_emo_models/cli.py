from __future__ import annotations

import argparse
from pathlib import Path

from dialog_emo_models.checkpoints import available_checkpoints, load_checkpoint
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
        help="Path to a saved checkpoint (joblib file or model directory). Overrides --model.",
    )
    score_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Trained checkpoint name from artifacts/models (see `dialog-emo list`). Overrides --model.",
    )
    score_parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: ./artifacts/models or $DIALOG_EMO_MODELS_DIR)",
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
        help="Path to a saved checkpoint (joblib file or model directory). Overrides --model.",
    )
    run_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Trained checkpoint name from artifacts/models (see `dialog-emo list`). Overrides --model.",
    )
    run_parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: ./artifacts/models or $DIALOG_EMO_MODELS_DIR)",
    )
    run_parser.set_defaults(func=_run_command)

    list_parser = subparsers.add_parser("list", help="List trained checkpoints in artifacts/models")
    list_parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: ./artifacts/models or $DIALOG_EMO_MODELS_DIR)",
    )
    list_parser.set_defaults(func=_list_command)

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
    model = _load_model(args)
    scored = score_parsed_frame(parsed, model, show_progress=True)
    output = write_full_csv(scored, args.output)
    print(f"scored {len(scored)} rows with {_model_label(args)} -> {output}")


def _run_command(args: argparse.Namespace) -> None:
    parsed = parse_telegram_json(args.input)
    if args.parsed_output is not None:
        parsed_output = write_parsed_csv(parsed, args.parsed_output)
        print(f"parsed {len(parsed)} rows -> {parsed_output}")

    model = _load_model(args)
    scored = score_parsed_frame(parsed, model, show_progress=True)
    output = write_full_csv(scored, args.output)
    print(f"scored {len(scored)} rows with {_model_label(args)} -> {output}")


def _train_command(args: argparse.Namespace) -> None:
    train_from_full_csv(args.input, model_name=args.model, output_path=args.output)
    print(f"trained {args.model!r} -> {args.output}")


def _list_command(args: argparse.Namespace) -> None:
    checkpoints = available_checkpoints(args.models_dir)
    if not checkpoints:
        print("no checkpoints found")
        return
    width = max(len(name) for name in checkpoints)
    for name, path in sorted(checkpoints.items()):
        kind = "dir" if path.is_dir() else "joblib"
        print(f"{name:{width}}  {kind:6}  {path}")


def _load_model(args: argparse.Namespace) -> EmotionModel:
    # Unified loader: a named checkpoint or an explicit path (joblib OR directory)
    # both go through load_checkpoint, which dispatches on the on-disk format.
    if args.checkpoint is not None:
        return load_checkpoint(args.checkpoint, models_dir=args.models_dir)
    if args.model_path is not None:
        return load_checkpoint(args.model_path, models_dir=args.models_dir)
    return create_model(args.model)


def _model_label(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        return repr(args.checkpoint)
    if args.model_path is not None:
        return str(args.model_path)
    return repr(args.model)


if __name__ == "__main__":
    main()
