import jsonschema
import json

def get_tool_schema():
    return {
        "name": "json_validator",
        "description": "Validates the system's configuration files and JSON data.",
        "parameters": {
            "type": "object",
            "properties": {
                "config": {"type": "string", "description": "Optional label for what is being checked."},
                "json_data": {"type": "string"},
                "schema_path": {"type": "string"}
            },
            "required": ["json_data", "schema_path"]
        }
    }

def run(json_data, schema_path, config=""):
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    except FileNotFoundError:
        return "Error: Schema file not found."
    except json.JSONDecodeError:
        return "Error: Invalid schema format."

    try:
        jsonschema.validate(json.loads(json_data), schema)
    except jsonschema.exceptions.ValidationError as e:
        return f"Error: JSON data is invalid: {e.message}"
    except jsonschema.exceptions.SchemaError as e:
        return f"Error: the schema itself is invalid: {e.message}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON data format."

    return "JSON data is valid."