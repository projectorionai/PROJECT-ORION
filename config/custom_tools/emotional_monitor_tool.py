"""Reviewed compatibility entry point with explicit, verified outcomes."""
from orion_core.local_tool_outcomes import analyse_text


def get_tool_schema():
    return {'name': 'emotional_monitor',
     'description': 'Estimate lexical sentiment from supplied text; not a verified emotional '
                    'state.',
     'parameters': {'type': 'object',
                    'properties': {'user_input': {'type': 'string'},
                                   'emotional_threshold': {'type': 'number'},
                                   'max_results': {'type': 'integer'}},
                    'required': ['user_input', 'emotional_threshold', 'max_results']}}


def run(user_input, emotional_threshold, max_results):
    return analyse_text(user_input, emotional_threshold, max_results)
