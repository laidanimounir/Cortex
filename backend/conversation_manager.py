

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid


class ConversationManager:
    
    
    def __init__(self, max_history: int = 5, session_timeout_minutes: int = 30):
       
        self.max_history = max_history
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
       
        self.sessions: Dict[str, dict] = {}
    
    
    def create_session(self) -> str:
        
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "history": [],
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
        }
        return session_id
    
    
    def add_message(self, session_id: str, question: str, answer: str, confidence: float):
  
        if session_id not in self.sessions:
         
            self.sessions[session_id] = {
                "history": [],
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
            }
        
   
        self.sessions[session_id]["last_activity"] = datetime.now()
        
    
        self.sessions[session_id]["history"].append({
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })
        
       
        if len(self.sessions[session_id]["history"]) > self.max_history:
            self.sessions[session_id]["history"].pop(0)
    
    
    def get_conversation_context(self, session_id: str) -> str:
        """
        الحصول على سياق المحادثة كنص منسق
        
        Returns:
            نص يحتوي على آخر الرسائل في المحادثة
        """
        if session_id not in self.sessions:
            return ""
        
        history = self.sessions[session_id]["history"]
        if not history:
            return ""
        
       
        context_parts = []
        for i, msg in enumerate(history, 1):
            context_parts.append(f"المستخدم: {msg['question']}")
            context_parts.append(f"المساعد: {msg['answer'][:200]}...")  
        
        return "\n".join(context_parts)
    
    
    def get_last_n_questions(self, session_id: str, n: int = 3) -> List[str]:
        """الحصول على آخر N أسئلة"""
        if session_id not in self.sessions:
            return []
        
        history = self.sessions[session_id]["history"]
        return [msg["question"] for msg in history[-n:]]
    
    
    def clear_session(self, session_id: str):
        """حذف جلسة محادثة"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    
    def cleanup_old_sessions(self):
        """حذف الجلسات القديمة (انتهت مدتها)"""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, data in self.sessions.items():
            if now - data["last_activity"] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            print(f"تم حذف {len(expired_sessions)} جلسة منتهية الصلاحية")
    
    
    def get_session_info(self, session_id: str) -> Optional[dict]:
        """الحصول على معلومات الجلسة"""
        if session_id not in self.sessions:
            return None
        
        data = self.sessions[session_id]
        return {
            "session_id": session_id,
            "message_count": len(data["history"]),
            "created_at": data["created_at"].isoformat(),
            "last_activity": data["last_activity"].isoformat(),
            "is_expired": datetime.now() - data["last_activity"] > self.session_timeout,
        }
