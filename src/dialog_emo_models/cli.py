from __future__ import annotations

import argparse
from pathlib import Path

from dialog_emo_models.pipeline import parse_telegram_json, score_parsed_frame
from dialog_emo_models.registry import available_model_names, create_model
from dialog_emo_models.schema import load_parsed_csv, write_full_csv, write_parsed_csv


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
    run_parser.set_defaults(func=_run_command)

    return parser


def _parse_command(args: argparse.Namespace) -> None:
    parsed = parse_telegram_json(args.input)
    output = write_parsed_csv(parsed, args.output)
    print(f"parsed {len(parsed)} rows -> {output}")


def _score_command(args: argparse.Namespace) -> None:
    parsed = load_parsed_csv(args.input)
    scored = score_parsed_frame(parsed, create_model(args.model))
    output = write_full_csv(scored, args.output)
    print(f"scored {len(scored)} rows with {args.model!r} -> {output}")


def _run_command(args: argparse.Namespace) -> None:
    parsed = parse_telegram_json(args.input)
    if args.parsed_output is not None:
        parsed_output = write_parsed_csv(parsed, args.parsed_output)
        print(f"parsed {len(parsed)} rows -> {parsed_output}")

    scored = score_parsed_frame(parsed, create_model(args.model))
    output = write_full_csv(scored, args.output)
    print(f"scored {len(scored)} rows with {args.model!r} -> {output}")
