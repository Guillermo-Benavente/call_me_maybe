*This project has been created as part of the 42 curriculum by gbenaven*

# Call Me Maybe

### Introduction to Function Calling in Large Language Models

**Version:** 1.5

## Description

Call Me Maybe is a Python project that explores function calling in Large Language Models (LLMs).

The main objective is to translate natural language requests into structured function calls. Instead of directly answering a user's question, the model must identify the appropriate function and generate its corresponding arguments in JSON format.

For example, given the request:

> What is the sum of 40 and 2?

The expected output is:

```json
{
    "name": "fn_add_numbers",
    "parameters": {
        "a": 40,
        "b": 2
    }
}
```

The project uses the **Qwen3-0.6B** model through the provided `llm_sdk` package.

A key aspect of this implementation is constrained decoding, which restricts token selection during generation to help maintain valid JSON output.

### Main Features

- Natural language processing through a small language model.
- Function selection based on function descriptions.
- Automatic extraction of function arguments.
- Token-by-token generation using model logits.
- JSON structural constraints during generation.
- Input and output validation using Pydantic.
- Configurable input, function definition and output paths.
- Execution time measurement for individual prompts and the complete process.

## Instructions

### Requirements

The project requires:

- Python 3.10 or later.
- [uv](https://docs.astral.sh/uv/).
- The `numpy` package.
- The `pydantic` package.
- The provided `llm_sdk` package.
- The Qwen3-0.6B model and its required resources.

### Installation

Clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

Synchronize the project dependencies:

```bash
uv sync
```

The `llm_sdk` package must be available in the project directory, alongside the `src` directory.

### Execution

Run the project using:

```bash
uv run python -m src
```

By default, the application uses:

| Argument | Default path |
|---|---|
| `--input` | `data/input/function_calling_tests.json` |
| `--functions_definition` | `data/input/functions_definition.json` |
| `--output` | `data/output/function_calling_results.json` |

You can also specify custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Makefile

The project can also use a Makefile to automate common development tasks:

| Command | Description |
|---|---|
| `make install` | Install project dependencies. |
| `make run` | Execute the application. |
| `make debug` | Execute the application using Python's debugger. |
| `make clean` | Remove temporary files and caches. |
| `make lint` | Run Flake8 and Mypy checks. |

## Project Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── llm.py
│   ├── output_file.py
│   ├── parser.py
│   ├── prompt_builder.py
│   └── validator.py
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Input Files

The application requires two JSON input files.

### 1. Function Calling Tests

`data/input/function_calling_tests.json`

Contains the natural language requests that the model must process.

Example:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?"
    },
    {
        "prompt": "Greet John"
    },
    {
        "prompt": "Reverse the string hello"
    }
]
```

Each entry must contain a non-empty string named `prompt`.

### 2. Function Definitions

`data/input/functions_definition.json`

Defines the functions available to the model.

Each function contains:

- `name`: Function identifier.
- `description`: Explanation of what the function does.
- `parameters`: Dictionary containing the argument names and types.
- `returns`: Definition of the return type.

Example:

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person.",
        "parameters": {
            "name": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    }
]
```

Pydantic models validate the input configuration and reject unexpected fields.

## Algorithm Explanation

The application follows a generation pipeline in which the LLM is responsible for selecting the function and generating its arguments.

### 1. Prompt Construction

`PromptBuilder` combines three elements:

- System instructions.
- Available function definitions.
- User request.

The resulting prompt instructs the model to select exactly one function and return a JSON object containing its name and parameters.

### 2. Tokenization

The constructed prompt is passed to the LLM tokenizer.

```python
input_ids = llm.encode(complete_prompt).squeeze(0).tolist()
```

The tokenizer converts the text into numerical token IDs that the model can process.

### 3. Logit Generation

The model processes the input token IDs and produces logits representing the scores of all possible next tokens.

```python
logits = np.asarray(
    self.get_logits(input_ids),
    dtype=np.float32
)
```

These scores determine which token the model considers the most appropriate next step.

### 4. Constrained Decoding

The project implements a lightweight JSON structural validator.

The `Validator` class maintains two important pieces of information:

- `buffer`: The decoded text generated so far.
- `generated`: The sequence of generated token IDs.

It also initializes two token collections:

| Collection | Purpose |
|---|---|
| `all_tokens` | Contains all vocabulary token IDs. |
| `structural_tokens` | Contains tokens whose decoded text consists exclusively of allowed JSON structural characters. |

The validator tracks whether the current generation position is inside a JSON string.

This distinction is important because JSON strings can contain ordinary characters, while outside strings the generation must follow JSON structural syntax.

The token selection process works as follows:

1. Retrieve the current allowed tokens.
2. Select the allowed token with the highest logit.
3. Append the selected token to the generated sequence.
4. Decode the generated sequence and update the validator state.
5. Check whether the generated content is a valid JSON object.
6. If it is not complete, obtain new logits and repeat.

The central token selection operation is:

```python
next_id = int(allowed[np.argmax(logits[allowed])])
```

This is a greedy decoding strategy: the highest-scoring allowed token is selected at every iteration.

Generation stops when the validator recognizes a valid JSON dictionary.

**Important limitation:** This implementation enforces a basic JSON structural constraint, not a complete JSON grammar or function-specific schema during token generation. It does not guarantee that every generated function name, parameter or parameter type matches the selected function definition. Those checks must be performed separately.

### 5. Output Validation

Once generation finishes, the generated token IDs are decoded into a string.

The `FunctionGenerated` Pydantic model validates the resulting function call.

The expected structure is:

```json
{
    "name": "fn_add_numbers",
    "parameters": {
        "a": 40,
        "b": 2
    }
}
```

### Generation Pipeline Overview

```text
        User Request
             |
             v
      Prompt Builder
             |
             v
       Tokenization
             |
             v
         LLM Model
             |
             v
           Logits
             |
             v
    Constrained Decoding
             |
             v
       JSON Validation
             |
             v
     Pydantic Validation
             |
             v
       Output Generation
             |
             v
          JSON File
```

## Design Decisions

### Small Language Model

The project uses `Qwen/Qwen3-0.6B` as its default model.

The objective is to investigate how constrained generation can improve the reliability of relatively small language models when producing structured output.

### Greedy Token Selection

Instead of sampling randomly from the model's output distribution, the implementation selects the highest-scoring allowed token.

This makes generation deterministic for a fixed model state and input, assuming deterministic model execution.

### NumPy for Token Selection

NumPy is used to efficiently represent token IDs and select the highest-scoring allowed token.

### Pydantic for Validation

Pydantic provides structured validation for input configurations and generated function calls.

The models use `extra="forbid"` to reject unexpected fields.

### Separation of Responsibilities

The project separates its responsibilities into different classes:

| Component | Responsibility |
|---|---|
| `Parser` | Parse command-line arguments and validate input files. |
| `PromptBuilder` | Construct the prompt sent to the model. |
| `LLM` | Interface with the language model and generate tokens. |
| `Validator` | Track generation state and validate JSON structure. |
| `OutputFile` | Generate the final JSON output file. |
| `main` | Coordinate the complete execution pipeline. |

This separation makes the implementation easier to understand, maintain and extend.

## Output File Format

The application generates a JSON array containing one result for each processed prompt.

Each result contains exactly three fields:

- `prompt`: Original user request.
- `name`: Selected function name.
- `parameters`: Generated function arguments.

Example:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2,
            "b": 3
        }
    },
    {
        "prompt": "Greet John",
        "name": "fn_greet",
        "parameters": {
            "name": "John"
        }
    }
]
```

The output is written using UTF-8 encoding and formatted with indentation for readability.

## Performance and Reliability

The application measures execution time at two levels:

- Individual prompt processing.
- Total execution time.

The model runs on the CPU, which avoids requiring a dedicated GPU but can increase generation time.

### Reliability

The implementation uses several mechanisms to improve reliability:

- JSON parsing through Python's standard `json` module.
- Input validation using Pydantic.
- Structural token restrictions during generation.
- Validation of the generated function call.
- Exception handling for input configuration errors.

### Performance Considerations

The generation process requests new logits after every generated token.

Consequently, generation time depends on:

- Input prompt length.
- Number of generated tokens.
- Model inference speed.
- Vocabulary size.
- Number of allowed tokens at each generation step.

## Challenges Faced

### Generating Valid JSON with a Small Model

Small language models can struggle to consistently produce structured output.

**Approach:** Implementing token restrictions helps guide generation towards valid JSON structures instead of relying exclusively on prompt instructions.

### Tracking JSON String Boundaries

JSON strings can contain characters that would otherwise be interpreted as structural syntax.

For example:

```json
{
    "message": "Hello, world!"
}
```

**Approach:** The validator tracks quotation marks and escaped characters to determine whether generation is currently inside a string.

### Token-Level Generation

The model operates on token IDs rather than complete words or characters.

**Approach:** The implementation uses the model's decoding interface to reconstruct the generated text and update the validation state after each token.

### Separating Generation and Validation

Producing syntactically valid JSON does not automatically guarantee that the generated data follows the expected function definition.

**Approach:** Pydantic is used as an additional validation layer after generation.

## Testing Strategy

The implementation should be tested using different prompts and function definitions rather than relying exclusively on the provided examples.

Recommended test cases include:

| Test case | Objective |
|---|---|
| Simple arithmetic | Verify numeric argument extraction. |
| String manipulation | Verify string argument generation. |
| Multiple parameters | Verify correct parameter assignment. |
| Special characters | Verify JSON string handling. |
| Empty or malformed input files | Verify input error handling. |
| Unknown function requests | Evaluate function selection reliability. |
| Large numbers | Evaluate numeric generation. |
| Escaped quotation marks | Test JSON string boundary detection. |

The following aspects should be evaluated independently:

1. JSON syntax validity.
2. Pydantic validation success.
3. Correct function selection.
4. Correct argument extraction.
5. Correct argument types.
6. Execution time.

## Resources

The following resources provide additional information about the technologies and concepts used in this project.

### Documentation

- [Python Documentation](https://docs.python.org/3/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [NumPy Documentation](https://numpy.org/doc/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Qwen Documentation](https://qwen.readthedocs.io/)
- [JSON Specification](https://www.json.org/json-en.html)

### Technical References

- [Constrained Decoding](https://arxiv.org/abs/2307.09702) — Research on controlling language model generation through structured constraints.
- [Large Language Models](https://arxiv.org/abs/2303.18223) — Overview of the capabilities and limitations of large language models.

### AI Usage

AI tools were used as supporting resources during development.

Their assistance included:

- Helping understand and document Python classes and methods.
- Improving code documentation following PEP 257 and Google-style docstrings.
- Assisting with the organization and writing of this README.
- Clarifying technical concepts related to LLMs, tokenization and constrained decoding.