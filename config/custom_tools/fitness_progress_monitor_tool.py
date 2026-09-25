"""Reviewed compatibility entry point with explicit, verified outcomes."""
from orion_core.local_tool_outcomes import save_progress


def get_tool_schema():
    return {'name': 'fitness_progress_monitor',
     'description': 'Persist and verify supplied workout data and requested reminders. Does not '
                    'schedule reminders or measure workouts.',
     'parameters': {'type': 'object',
                    'properties': {'user_id': {'type': 'string'},
                                   'workout_data': {'type': 'array', 'items': {'type': 'object'}},
                                   'reminders': {'type': 'array', 'items': {'type': 'string'}},
                                   'performance_metrics': {'type': 'object'}},
                    'required': ['user_id', 'workout_data', 'reminders', 'performance_metrics']}}


def run(user_id, workout_data, reminders, performance_metrics):
    return save_progress(user_id, workout_data, reminders, performance_metrics)
