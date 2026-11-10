import json
from pathlib import Path

class OutputFile():
	def __init__(self, prompts):
		self.prompts = prompts
		self.functions = []

	def create_output(self, output_path: Path):
		output = []
		for index, prompt in enumerate(self.prompts):
			prompt_json = json.loads(prompt)
			function_json = json.loads(self.functions[index])
			output_json = {
				"prompt" : prompt_json['prompt'],
				"name" : function_json['name'],
				"parameters" : function_json['parameters']
			}
			output.append(output_json)
		with open(output_path, "w", encoding="utf-8") as file:
			file.writelines(output)