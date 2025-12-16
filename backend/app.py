from flask import Flask, request, jsonify
from flask_cors import CORS
from qa_engine import QAEngine
from pathlib import Path
import logging
from duckduckgo_search import DDGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "data" / "knowledge_base.csv"

engine = QAEngine(min_confidence=0.6)
engine.load_knowledge_base(csv_path)


def search_web(query: str):
    try:
        logger.info(f"Searching web for: {query}")
        results = DDGS().text(query, max_results=1)
        if results:
            return {
                "answer": results[0]["body"],
                "source": results[0]["href"],
                "title": results[0]["title"],
            }
    except Exception as e:
        logger.error(f"Web search error: {e}")
    return None


@app.route("/")
def home():
    return jsonify(
        {
            "message": "Cortex QA API",
            "version": "0.5",
            "endpoints": ["/ask", "/stats"],
        }
    )


@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return (
                jsonify(
                    {
                        "error": "Question is required",
                        "example": {"question": "What is Python?"},
                    }
                ),
                400,
            )

        question = data["question"].strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        if len(question) > 500:
            return (
                jsonify(
                    {"error": "Question too long (max 500 characters)"}
                ),
                400,
            )

        logger.info(f"Received question: {question[:80]}...")

        result = engine.find_answer(question)
        best_conf = float(result["confidence"])
        logger.info(f"Answer confidence (local): {best_conf:.2f}")

        top_matches = result.get("top_matches", [])

        if best_conf < engine.min_confidence:
            logger.info("Low confidence, trying web search...")
            web_result = search_web(question)

            if web_result:
                return (
                    jsonify(
                        {
                            "question": question,
                            "answer": web_result["answer"],
                            "confidence": 1.0,
                            "source": "web",
                            "url": web_result["source"],
                            "title": web_result["title"],
                            "top_matches": top_matches,
                        }
                    ),
                    200,
                )
            else:
                return (
                    jsonify(
                        {
                            "question": question,
                            "answer": "عذرًا، لم أجد إجابة مناسبة في قاعدة البيانات أو على الويب. جرب إعادة صياغة السؤال أو اسأل عن موضوع آخر.",
                            "confidence": best_conf,
                            "source": "none",
                            "top_matches": top_matches,
                        }
                    ),
                    200,
                )

        source = result.get("source", "local")
        return (
            jsonify(
                {
                    "question": result.get("question") or question,
                    "answer": result.get("answer"),
                    "confidence": best_conf,
                    "source": source,
                    "top_matches": top_matches,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return (
            jsonify(
                {
                    "error": "Internal server error",
                    "message": str(e),
                }
            ),
            500,
        )


@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify(
        {
            "total_questions": len(engine.questions),
            "languages": ["Multilingual (depends on KB)"],
            "version": "0.5",
            "status": "running",
        }
    )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" Cortex API Server Running")
    print(" Access: http://127.0.0.1:5000")
    print(" Stats:  http://127.0.0.1:5000/stats")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
