import json
from pathlib import Path
from src.parser import Prompt, FunctionGenerated

class OutputFile():
	def __init__(self, prompts):
		self.prompts: list[Prompt] = prompts
		self.functions: list[FunctionGenerated] = []

	def create_output(self, output_path: Path):
		output = []
		for index, prompt in enumerate(self.prompts):
			output_json = {
				"prompt" : prompt.prompt,
				"name" : self.functions[index].name,
				"parameters" : self.functions[index].parameters
			}
			output.append(output_json)
		with open(output_path, "w", encoding="utf-8") as file:
			json.dump(output, file, ensure_ascii=False, indent=4)