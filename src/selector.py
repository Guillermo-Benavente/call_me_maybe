"""Selection of a function and its arguments for a natural-language prompt."""

from typing import Any

from .parser import ResponseParser
from .prompt_builder import PromptBuilder
from .schema_generator import LanguageModel, SchemaConstrainedGenerator
from .validator import FunctionDefinition


class FunctionSelector:
    """Selects one function and fills its arguments using the schema."""

    def __init__(
        self,
        model: LanguageModel,
        max_new_tokens: int = 256,
    ) -> None:
        """Create the schema-constrained generator.

        Args:
            model: Language model that scores the next token.
            max_new_tokens: Maximum number of tokens the model may generate.
        """
        self.generator = SchemaConstrainedGenerator(model, max_new_tokens)

    def select(
        self,
        user_prompt: str,
        functions: list[FunctionDefinition],
    ) -> dict[str, Any]:
        """Generate the function call for a natural-language request.

        Args:
            user_prompt: Request written in natural language.
            functions: Available function definitions.

        Returns:
            The parsed function call produced by the model.

        Raises:
            ValueError: If generation does not produce valid JSON.
        """
        prompt = PromptBuilder.build(
            user_prompt=user_prompt,
            functions=functions,
        )
        response = self.generator.generate(prompt, functions)
        return ResponseParser.parse(response)
