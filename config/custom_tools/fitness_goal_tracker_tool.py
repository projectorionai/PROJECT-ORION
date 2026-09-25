"""Reviewed compatibility entry point with explicit, verified outcomes."""
from orion_core.local_tool_outcomes import save_goal


def get_tool_schema():
    return {'name': 'fitness_goal_tracker',
     'description': 'Persist and verify a fitness goal with an ISO YYYY-MM-DD deadline and '
                    'progress from 0 to 100. Does not schedule reminders.',
     'parameters': {'type': 'object',
                    'properties': {'goal': {'type': 'string'},
                                   'deadline': {'type': 'string',
                                                'description': 'ISO date YYYY-MM-DD'},
                                   'progress': {'type': 'integer'},
                                   'status': {'type': 'string'}},
                    'required': ['goal', 'deadline']}}


def run(goal, deadline, progress=0, status="In Progress"):
    return save_goal(goal, deadline, progress, status)
