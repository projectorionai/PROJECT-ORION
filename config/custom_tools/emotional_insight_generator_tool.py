import json

def get_tool_schema() -> dict:
    return {
        "name": "emotional_insight_generator",
        "description": "Analyzes user emotions over time and provides actionable insights.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_emotions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp": {"type": "string"},
                            "emotion": {"type": "string"},
                            "intensity": {"type": "number"}
                        },
                        "required": ["timestamp", "emotion", "intensity"]
                    }
                },
                "threshold": {
                    "type": "number"
                }
            },
            "required": ["user_emotions", "threshold"]
        }
    }

def run(user_emotions, threshold) -> str:
    try:
        if not isinstance(user_emotions, list) or not all(isinstance(emotion, dict) for emotion in user_emotions):
            raise ValueError("user_emotions must be a list of dictionaries.")
        
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a number.")
        
        insights = []
        for emotion in user_emotions:
            if emotion['intensity'] >= threshold:
                insights.append(f"High intensity of {emotion['emotion']} detected at {emotion['timestamp']}.")
        
        return json.dumps(insights) if insights else "No significant emotions detected."
    
    except Exception as e:
        return str(e)