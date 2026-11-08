from src.llm import LLM
from src.parser import Parser
from src.prompt_builder import PromptBuilder


def main() -> None:
    try:
        print("Loading input files...")
        parser = Parser()
        print(f"Input Path: {parser.input_path}")
        print(f"Functions Definition Path: {parser.functions_path}")

        print("\nLoading LLM model...")
        llm = LLM()
        print("Model loaded!")

        user_prompt = parser.prompts[0].prompt
        
        complete_prompt = PromptBuilder.build(
            user_prompt=user_prompt,
            functions_json=parser.functions_json,
        )
        
        print("\nuser prompt:")
        print(user_prompt)
        
        input_ids = llm.encode(complete_prompt).squeeze(0).tolist()
        
        print(f"\nNumber of input tokens: {len(input_ids)}")
        
        logits = llm.get_logits(input_ids)
        
        print(f"Number of logits: {len(logits)}")
        print(f"First logits: {logits[:5]}")
        
        decoded = llm.decode(input_ids)
        
        print(f"\nDecoded prompt:\n{decoded}")
    except ValueError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()