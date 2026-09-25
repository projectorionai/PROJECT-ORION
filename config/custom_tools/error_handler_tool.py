"""Reviewed compatibility entry point with explicit, verified outcomes."""
from orion_core.local_tool_outcomes import record_error


def get_tool_schema():
    return {'name': 'error_handler',
     'description': 'Persist and verify a diagnostic record. Does not apply repairs.',
     'parameters': {'type': 'object',
                    'properties': {'error_message': {'type': 'string'},
                                   'error_type': {'type': 'string'},
                                   'context': {'type': 'string'}},
                    'required': ['error_message', 'error_type']}}


def run(error_message, error_type, context=""):
    return record_error(error_message, error_type, context)
