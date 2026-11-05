from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model(device="cpu")

    text = "What is the sum of 2 and 3?"

    input_ids = model.encode(text)

    logits = model.get_logits_from_input_ids(
        input_ids[0].tolist()
    )

    print(f"Input IDs: {input_ids}")
    print(f"Number of logits: {len(logits)}")
    print(f"First logits: {logits[:10]}")


if __name__ == "__main__":
    main()
