from flask import Flask, request, jsonify
from flask_cors import CORS
from qa_engine import QAEngine
from conversation_manager import ConversationManager
from pathlib import Path
import logging
from duckduckgo_search import DDGS
from datetime import datetime

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


conversation_manager = ConversationManager(max_history=5, session_timeout_minutes=30)


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
    return jsonify({
        "message": "Cortex QA API with Conversation Memory",
        "version": "1.0",
        "endpoints": ["/ask", "/stats", "/session/new", "/session/clear", "/session/info"],
    })


@app.route("/session/new", methods=["POST"])
def create_new_session():
   
    try:
        session_id = conversation_manager.create_session()
        logger.info(f"Created new session: {session_id}")
        
        return jsonify({
            "session_id": session_id,
            "message": "تم إنشاء جلسة محادثة جديدة",
            "created_at": datetime.now().isoformat(),
        }), 201
    
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/session/clear", methods=["POST"])
def clear_session():
    
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        conversation_manager.clear_session(session_id)
        logger.info(f"Cleared session: {session_id}")
        
        return jsonify({
            "message": "تم مسح المحادثة",
            "session_id": session_id,
        }), 200
    
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/session/info", methods=["GET"])
def get_session_info():
  
    try:
        session_id = request.args.get("session_id")
        
        if not session_id:
            return jsonify({"error": "session_id parameter is required"}), 400
        
        info = conversation_manager.get_session_info(session_id)
        
        if not info:
            return jsonify({"error": "Session not found"}), 404
        
        return jsonify(info), 200
    
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
  
    try:
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({
                "error": "Question is required",
                "example": {"question": "What is Python?", "session_id": "optional-uuid"},
            }), 400
        
        question = data["question"].strip()
        session_id = data.get("session_id") 
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        if len(question) > 500:
            return jsonify({"error": "Question too long (max 500 characters)"}), 400
        
      
        if not session_id:
            session_id = conversation_manager.create_session()
            logger.info(f"Auto-created session: {session_id}")
        
        logger.info(f"[{session_id[:8]}] Question: {question[:80]}...")
        
      
        context = conversation_manager.get_conversation_context(session_id)
        
       
        result = engine.find_answer(question, context=context)
        best_conf = float(result["confidence"])
        logger.info(f"[{session_id[:8]}] Confidence: {best_conf:.2f}")
        
        top_matches = result.get("top_matches", [])
        
       
        if best_conf < engine.min_confidence:
            logger.info("Low confidence, trying web search...")
            web_result = search_web(question)
            
            if web_result:
              
                conversation_manager.add_message(
                    session_id, question, web_result["answer"], 1.0
                )
                
                return jsonify({
                    "session_id": session_id,
                    "question": question,
                    "answer": web_result["answer"],
                    "confidence": 1.0,
                    "source": "web",
                    "url": web_result["source"],
                    "title": web_result["title"],
                    "top_matches": top_matches,
                    "has_context": bool(context),
                }), 200
            else:
                fallback_answer = "عذرًا، لم أجد إجابة مناسبة في قاعدة البيانات أو على الويب. جرب إعادة صياغة السؤال أو اسأل عن موضوع آخر."
                
                conversation_manager.add_message(
                    session_id, question, fallback_answer, best_conf
                )
                
                return jsonify({
                    "session_id": session_id,
                    "question": question,
                    "answer": fallback_answer,
                    "confidence": best_conf,
                    "source": "none",
                    "top_matches": top_matches,
                    "has_context": bool(context),
                }), 200
        
       
        answer = result.get("answer")
        
       
        conversation_manager.add_message(session_id, question, answer, best_conf)
        
      
        conversation_manager.cleanup_old_sessions()
        
        return jsonify({
            "session_id": session_id,
            "question": result.get("question") or question,
            "answer": answer,
            "confidence": best_conf,
            "source": "local",
            "top_matches": top_matches,
            "has_context": bool(context),
        }), 200
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
        }), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    
    return jsonify({
        "total_questions": len(engine.questions),
        "active_sessions": len(conversation_manager.sessions),
        "languages": ["Arabic", "English", "German", "Multilingual"],
        "version": "1.0",
        "status": "running",
        "features": ["conversation_memory", "web_search", "multilingual"],
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Cortex API Server with Conversation Memory")
    print("  Access: http://127.0.0.1:5000")
    print("  Stats:  http://127.0.0.1:5000/stats")
    print("  Features: Conversation Memory, Web Search, Multilingual")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
