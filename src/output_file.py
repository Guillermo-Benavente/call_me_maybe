import json
from pathlib import Path
from src.parser import Prompt, FunctionGenerated


class OutputFile():
    """Manage the generation of function calling result files."""

    def __init__(self, prompts: list[Prompt]) -> None:
        """Initialize the output file manager.

        Args:
            prompts: List of input prompts associated with the generated
                function calls.
        """
        self.prompts: list[Prompt] = prompts
        self.functions: list[FunctionGenerated] = []

    def create_output(self, output_path: Path) -> None:
        """Generate a JSON file containing the function calling results.

        Combines each input prompt with its corresponding generated
        function name and parameters, then writes the results to a JSON file.

        Args:
            output_path: Destination path where the output JSON file
                will be created.

        Raises:
            IndexError: If there are fewer generated functions than prompts.
            OSError: If the output file cannot be created or written.
        """
        output = []
        for index, prompt in enumerate(self.prompts):
            output_json = {
                "prompt": prompt.prompt,
                "name": self.functions[index].name,
                "parameters": self.functions[index].parameters
            }
            output.append(output_json)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)
