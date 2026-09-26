import json

class SelfRepairAgent:
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(random_state=42, n_jobs=1)
        self.data = []
        self.labels = []

    def train(self, data, labels):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        self.data = data
        self.labels = labels
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)

    def predict_failure(self, input_data):
        return self.model.predict([input_data])[0]

def get_tool_schema():
    return {
        "name": "self_repair_agent",
        "description": "A tool to enhance self-repair efficiency and predict failures using machine learning.",
        "parameters": {
            "type": "object",
            "properties": {
                "training_data": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "training_labels": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "input_data": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            },
            "required": ["training_data", "training_labels", "input_data"]
        }
    }

def run(training_data, training_labels, input_data):
    try:
        agent = SelfRepairAgent()
        accuracy = agent.train(training_data, training_labels)
        prediction = agent.predict_failure(input_data)
        return f"Training completed with accuracy: {accuracy:.2f}. Prediction: {prediction}."
    except Exception as e:
        return f"Error: {str(e)}"