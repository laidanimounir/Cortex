from flask import Flask, request, jsonify
from flask_cors import CORS
from qa_engine import QAEngine
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Initialize QA Engine
base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "data" / "knowledge_base.csv"

engine = QAEngine()
engine.load_knowledge_base(csv_path)

@app.route('/')
def home():
    return jsonify({
        "message": "Cortex QA API",
        "version": "0.3",
        "endpoints": ["/ask"]
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question']
    result = engine.find_answer(question)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
