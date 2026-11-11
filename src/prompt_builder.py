class PromptBuilder:
    """Build prompts for function calling."""

    @staticmethod
    def build(
        user_prompt: str,
        functions_json: str,
    ) -> str:
        """Build the complete model prompt."""

        return (
            "You are a function calling assistant.\n"
            "Choose exactly one function from the list and fill its parameters.\n"
            "Answer with ONE JSON object and nothing else.\n"
            "No markdown, no code fences, no explanations, no extra text.\n\n"
            f"Functions: {functions_json}\n\n"
            f"User request: {user_prompt}\n\n"
            "Required output format:\n"
            '{"name": "<function_name>", "parameters": {<arguments>}}\n\n'
            "JSON:\n"
        )