import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model

from src.selector import FunctionSelector
from src.validator import FunctionValidator


def parse_arguments() -> argparse.Namespace:
    """Read the command-line arguments used by the program."""
    parser = argparse.ArgumentParser(
        description="LLM-based function calling."
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )

    return parser.parse_args()


def load_json_file(file_path: str) -> Any:
    """Read a JSON file and report clear errors when it is invalid.

    Args:
        file_path: Path to the JSON file.

    Returns:
        The parsed JSON content.

    Raises:
        ValueError: If the file does not exist or contains invalid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {file_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {file_path}: {error}"
        ) from error


def parse_prompts(raw_prompts: Any) -> list[str]:
    """Validate the input file and return the natural-language prompts.

    Args:
        raw_prompts: Parsed JSON content from the input file.

    Returns:
        The prompts that must be processed.

    Raises:
        ValueError: If the input does not contain a list of prompts.
    """
    if isinstance(raw_prompts, dict):
        raw_prompts = raw_prompts.get("tests", raw_prompts)

    if not isinstance(raw_prompts, list):
        raise ValueError("Input prompts must be a JSON array.")

    prompts: list[str] = []

    for item in raw_prompts:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompts.append(item["prompt"])
        else:
            raise ValueError(f"Invalid prompt entry: {item!r}")

    return prompts


def main() -> None:
    """Run the function-calling pipeline."""
    arguments = parse_arguments()

    try:
        functions_data = load_json_file(arguments.functions_definition)
        tests_data = load_json_file(arguments.input)
        validator = FunctionValidator()
        function_definitions = validator.parse_function_definitions(
            functions_data
        )
        prompts = parse_prompts(tests_data)
    except (OSError, ValueError, TypeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return

    try:
        model = Small_LLM_Model(device="cpu")
        selector = FunctionSelector(model)
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"[ERROR] Could not initialize the language model: {error}",
            file=sys.stderr,
        )
        return

    results: list[dict[str, Any]] = []

    for prompt in prompts:
        try:
            result = selector.select(prompt, function_definitions)

            if not validator.validate_function_call(
                result,
                function_definitions,
            ):
                print(
                    "[WARNING] Invalid function call for prompt "
                    f"{prompt!r}: {result}",
                    file=sys.stderr,
                )

            results.append(validator.build_output(prompt, result))
        except (ValueError, KeyError, TypeError) as error:
            print(
                f"[ERROR] Could not process prompt {prompt!r}: {error}",
                file=sys.stderr,
            )
            results.append({
                "prompt": prompt,
                "name": "",
                "parameters": {},
            })

    output_path = Path(arguments.output)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                results,
                output_file,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as error:
        print(
            f"[ERROR] Could not write output file {output_path}: {error}",
            file=sys.stderr,
        )
        return

    print(f"Processed {len(results)} prompts.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
