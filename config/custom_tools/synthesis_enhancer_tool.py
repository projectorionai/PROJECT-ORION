"""Reviewed compatibility entry point with explicit, verified outcomes."""
from orion_core.local_tool_outcomes import save_tool_spec


def get_tool_schema():
    return {'name': 'synthesis_enhancer',
     'description': 'Persist and verify a draft tool specification. Use forge to implement it; '
                    'this does not create executable tools.',
     'parameters': {'type': 'object',
                    'properties': {'tool_name': {'type': 'string',
                                                 'description': 'The name of the tool to be '
                                                                'synthesized.'},
                                   'capability_plan': {'type': 'string',
                                                       'description': 'A brief description of the '
                                                                      'capability plan for the '
                                                                      'tool.'}},
                    'required': ['tool_name', 'capability_plan']}}


def run(tool_name, capability_plan):
    return save_tool_spec(tool_name, capability_plan)
