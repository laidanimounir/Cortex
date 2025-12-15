from flask import Flask, request, jsonify
from flask_cors import CORS
from qa_engine import QAEngine
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "data" / "knowledge_base.csv"

engine = QAEngine()
engine.load_knowledge_base(csv_path)

@app.route('/')
def home():
    return jsonify({
        "message": "Cortex QA API",
        "version": "0.4",
        "endpoints": ["/ask", "/stats"]
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        
     
        if not data or 'question' not in data:
            return jsonify({
                "error": "Question is required",
                "example": {"question": "What is Python?"}
            }), 400
        
        question = data['question'].strip()
        
    
        if not question:
            return jsonify({
                "error": "Question cannot be empty"
            }), 400
        
       
        if len(question) > 500:
            return jsonify({
                "error": "Question too long (max 500 characters)"
            }), 400
        
       
        logger.info(f"Received question: {question[:50]}...")
        result = engine.find_answer(question)
        logger.info(f"Answer confidence: {result['confidence']:.2f}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "total_questions": len(engine.questions),
        "languages": ["English", "Arabic"],
        "version": "0.4",
        "status": "running"
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print(" Cortex API Server Running")
    print(" Access: http://127.0.0.1:5000")
    print(" Stats:  http://127.0.0.1:5000/stats")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
