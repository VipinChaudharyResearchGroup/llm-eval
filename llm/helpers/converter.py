import csv
import json
from typing import Literal

import yaml
from jinja2 import Template


def export_csv_data(
    csv_path: str, output_format: Literal["json", "yaml", "jinja"]
) -> None:

    with open(csv_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = list(csv_reader)

    for row in data:
        for key, value in row.items():
            if isinstance(value, str):
                row[key] = value.replace("\\n", "\n")
    output_path = csv_path.replace(".csv", f".{output_format}")

    if output_format == "json":
        with open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    elif output_format == "yaml":
        with open(output_path, "w") as yaml_file:
            yaml.dump(
                data,
                yaml_file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    elif output_format == "jinja":

        with open("template.j2", "r") as template_file:
            template = Template(template_file.read())

        output = template.render(data=data)

        with open(output_path.replace(".jinja", ".txt"), "w") as output_file:
            output_file.write(output)


if __name__ == "__main__":
    csv_path = "./output/ex1/meta-llama_Llama-2-7b-chat-hf_2024-07-22_10-45-27/unsuccessful_responses.csv"

    for i in ["json", "yaml", "jinja"]:
        export_csv_data(csv_path, i)
