from src.parser import Parser


def main() -> None:
    try:
        parser = Parser()

        print(f"Input Path: {parser.input_path}")
        print(f"Functions Definition Path: {parser.functions_path}")
    except ValueError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()