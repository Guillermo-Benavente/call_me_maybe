from argparse import Namespace, ArgumentParser
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class Prompt(BaseModel):
    """Represent an input prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(strict=True, min_length=1)


class ParameterDefinition(BaseModel):
    """Represent a function parameter definition."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(strict=True, min_length=1)


class FunctionDefinition(BaseModel):
    """Represent an available function."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(strict=True, min_length=1)
    description: str = Field(strict=True, min_length=1)
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class FunctionGenerated(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(strict=True, min_length=1)
    parameters: dict[str, int | float | str]


class Parser:
    """Parse command-line arguments and validate input JSON files."""

    def __init__(self) -> None:
        """Parse arguments and load input files."""

        self.args: Namespace = self._parse_arguments()

        self.input_path: Path = self.args.input
        self.functions_path: Path = self.args.functions_definition
        self.output_path: Path = self.args.output

        self.prompts: list[Prompt] = []
        self.functions: list[FunctionDefinition] = []

        try:
            self.prompts_json: str = self.compact_json(self.input_path)
            self.functions_json: str = self.compact_json(self.functions_path)

            self.prompts = self._load_json(
                self.input_path,
                list[Prompt],
            )

            self.functions = self._load_json(
                self.functions_path,
                list[FunctionDefinition],
            )

        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"Input configuration error: {error}") from error

    @staticmethod
    def _parse_arguments() -> Namespace:
        """Parse command-line arguments."""

        project_root: Path = Path(__file__).resolve().parent.parent

        arg_parser: ArgumentParser = ArgumentParser(
            description="Call Me Maybe - Function Calling"
        )

        arg_parser.add_argument(
            "--input",
            type=Path,
            default=project_root / "data/input/function_calling_tests.json",
        )

        arg_parser.add_argument(
            "--functions_definition",
            type=Path,
            default=project_root / "data/input/functions_definition.json",
        )

        arg_parser.add_argument(
            "--output",
            type=Path,
            default=project_root / "data/output/function_calling_results.json",
        )

        return arg_parser.parse_args()

    @staticmethod
    def _load_json(file_path: Path, schema: Any) -> Any:
        """Load and validate a JSON file using Pydantic."""

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        adapter: TypeAdapter[Any] = TypeAdapter(schema)

        return adapter.validate_json(
            file_path.read_text(encoding="utf-8")
        )

    @staticmethod
    def validate_str_to_json(possible_json: str, schema: Any) -> Any:
        adapter: TypeAdapter[Any] = TypeAdapter(schema)

        return adapter.validate_json(possible_json)

    @staticmethod
    def compact_json(file: Path) -> str:
        data = json.loads(file.read_text(encoding="utf-8"))
        final_json = json.dumps(
            data, ensure_ascii=False,
            separators=(",", ":")
        )

        return final_json
