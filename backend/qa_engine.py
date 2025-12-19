import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class QAEngine:
    
    def __init__(self, min_confidence: float = 0.75, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.question_embeddings: np.ndarray | None = None
        self.min_confidence = min_confidence

    
    def load_knowledge_base(self, csv_path: str | Path) -> None:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, encoding="utf-8")
        
        self.questions = df["question"].astype(str).tolist()
        self.answers = df["answer"].astype(str).tolist()
        
        print("Computing embeddings for knowledge base...")
        self.question_embeddings = self.model.encode(
            self.questions,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        print(f"Loaded {len(self.questions)} questions with embeddings")

    
    def find_answers(self, user_question: str, top_k: int = 3, context: str = "") -> list[dict]:
       
        if self.question_embeddings is None:
            raise ValueError("Knowledge base not loaded")
        
      
        search_query = user_question
        if context:
          
            search_query = f"{user_question} [السياق: {context[-200:]}]"
        
        user_embedding = self.model.encode([search_query], convert_to_tensor=False)
        similarities = cosine_similarity(user_embedding, self.question_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results: list[dict] = []
        for idx in top_indices:
            confidence = float(similarities[idx])
            results.append({
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "confidence": confidence,
                "index": int(idx),
                "source": "local" if confidence >= self.min_confidence else "local_low",
            })
        return results

    
    def find_answer(self, user_question: str, min_confidence: float | None = None, context: str = "") -> dict:
    
        if min_confidence is None:
            min_confidence = self.min_confidence
        
    
        results = self.find_answers(user_question, top_k=5, context=context)
        best = results[0]
        confidence = best["confidence"]
        
       
        if context and confidence < min_confidence:
            results_no_context = self.find_answers(user_question, top_k=3, context="")
            if results_no_context[0]["confidence"] > confidence:
                best = results_no_context[0]
                confidence = best["confidence"]
        
        if confidence < min_confidence:
            return {
                "question": None,
                "answer": None,
                "confidence": float(confidence),
                "source": "none",
                "top_matches": results,
            }
        
        return {
            "question": best["question"],
            "answer": best["answer"],
            "confidence": float(confidence),
            "source": "local",
            "top_matches": results,
        }

    
    def add_to_knowledge_base(self, question: str, answer: str, csv_path: str | Path | None = None) -> None:
        self.questions.append(question)
        self.answers.append(answer)
        
        new_embedding = self.model.encode([question], convert_to_tensor=False)
        
        if self.question_embeddings is None:
            self.question_embeddings = new_embedding
        else:
            self.question_embeddings = np.vstack([self.question_embeddings, new_embedding])
        
        if csv_path is not None:
            csv_path = Path(csv_path)
        else:
            base_dir = Path(__file__).resolve().parent.parent
            csv_path = base_dir / "data" / "knowledge_base.csv"
        
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([question, answer])
        
        print(f"Added new Q&A. Total questions: {len(self.questions)}")


def build_qa_engine() -> QAEngine:
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "knowledge_base.csv"
    
    engine = QAEngine(min_confidence=0.75)
    engine.load_knowledge_base(csv_path)
    return engine


if __name__ == "__main__":
    engine = build_qa_engine()
    
    print("Multilingual Semantic QA System (Embeddings)")
    print("Enter your question in any language (type 'exit' to quit)\n")
    
    while True:
        question = input("Your question: ")
        if question.lower().strip() == "exit":
            print("\nGoodbye!")
            break
        if not question.strip():
            print("Please enter a valid question\n")
            continue
        
        result = engine.find_answer(question)
        print(f"\n{'=' * 50}")
        if result["question"]:
            print(f"Matched Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Match Score: {result['confidence']:.1%}")
        else:
            print("No good match found in knowledge base.")
            print(f"Best score: {result['confidence']:.1%}")
        print(f"{'=' * 50}\n")
