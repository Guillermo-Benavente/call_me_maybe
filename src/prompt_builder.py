class PromptBuilder:
    """Build prompts for function calling."""

    @staticmethod
    def build(
        user_prompt: str,
        functions_json: str,
    ) -> str:
        """Build the complete model prompt."""

        return (
            "You are a function calling assistant."
            "Choose exactly one function from the list and fill its parameters."
            "Answer with ONE JSON object and nothing else.\n\n"
            f"Functions: {functions_json}\n\n"
            f"User request: {user_prompt}\n\n"
            "Required output format:"
            '{"name": "<function_name>", "parameters": {<arguments>}}\n\n'
            "JSON:\n"
        )