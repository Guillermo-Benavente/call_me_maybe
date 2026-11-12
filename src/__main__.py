import time
from src.llm import LLM
from src.parser import Parser, FunctionGenerated
from src.prompt_builder import PromptBuilder
from src.output_file import OutputFile


def main() -> None:
    """Execute the function calling pipeline.

    Loads the input configurations, initializes the language model,
    processes each prompt, validates the generated function calls,
    measures execution times, and saves the results to an output file.

    Raises:
        ValueError: If an input configuration or generated function call
            contains invalid data.
    """
    try:
        print(f"{'*' * 10} Loading input files {'*' * 10}")
        parser = Parser()
        print(f"* Input Path: {parser.input_path}")
        print(f"* Functions Definition Path: {parser.functions_path}")
        print(f"{'*' * 41}")

        print("\nLoading LLM model...")
        llm = LLM()
        print("Model loaded!")

        total_start = time.perf_counter()

        output_file = OutputFile(parser.prompts)
        total_prompts = len(parser.prompts)

        print(f"\nProcessing {total_prompts} prompts...\n")

        for index, prompt in enumerate(parser.prompts, start=1):
            start = time.perf_counter()

            user_prompt = prompt.prompt

            print(f"Prompt {index}: ", end="")

            complete_prompt = PromptBuilder.build(
                user_prompt=user_prompt,
                functions_json=parser.functions_json,
            )

            input_ids = llm.encode(complete_prompt).squeeze(0).tolist()
            generated = llm.generate_json(input_ids)
            decoded = llm.decode(generated)

            output_file.functions.append(
                Parser.validate_str_to_json(decoded, FunctionGenerated)
            )

            elapsed = time.perf_counter() - start
            minutes, seconds = divmod(int(elapsed), 60)

            print(f"{minutes} min {seconds} s")

        total_elapsed = time.perf_counter() - total_start
        minutes, seconds = divmod(int(total_elapsed), 60)

        print(f"\n{'=' * 50}")
        print(f"Finished processing {total_prompts} prompts")
        print(f"Total execution time: {minutes} min {seconds} s")
        print(f"{'=' * 50}")
        output_file.create_output(parser.output_path)

    except ValueError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
