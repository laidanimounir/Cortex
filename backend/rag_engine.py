import csv
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid


class QAEngineRag:
  
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        persist_directory: str = "./chroma_db"
    ):
      
        self.min_confidence = min_confidence
        self.model = SentenceTransformer(model_name)
        
        
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
       
        try:
            self.collection = self.chroma_client.get_collection(name="knowledge_base")
            print(f" Loaded existing collection with {self.collection.count()} items")
        except:
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"} 
            )
            print(" Created new collection")
    
    
    def load_from_csv(self, csv_path: str | Path, clear_existing: bool = False) -> None:
       
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
       
        if clear_existing:
            try:
                self.chroma_client.delete_collection(name="knowledge_base")
                self.collection = self.chroma_client.create_collection(
                    name="knowledge_base",
                    metadata={"hnsw:space": "cosine"}
                )
                print(" Cleared existing data")
            except:
                pass
        
       
        df = pd.read_csv(csv_path, encoding="utf-8")
        
        print(f"📂 Loading {len(df)} Q&A pairs from CSV...")
        
        questions = df["question"].astype(str).tolist()
        answers = df["answer"].astype(str).tolist()
        
        
        print(" Computing embeddings...")
        embeddings = self.model.encode(
            questions,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        
     
        ids = [str(uuid.uuid4()) for _ in range(len(questions))]
        metadatas = [
            {
                "question": q,
                "language": self._detect_language(q),
                "length": len(a)
            }
            for q, a in zip(questions, answers)
        ]
        
       
        print(" Saving to ChromaDB...")
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=answers,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f" Successfully loaded {len(questions)} Q&A pairs")
        print(f" Total items in collection: {self.collection.count()}")
    
    
    def _detect_language(self, text: str) -> str:
    
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return "arabic"
        elif any('\u0041' <= c <= '\u007A' for c in text.lower()):
            return "english"
        else:
            return "other"
    
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        context: str = "",
        language_filter: Optional[str] = None
    ) -> List[Dict]:
 
        search_query = query
        if context:
            search_query = f"{query} [السياق: {context[-200:]}]"
        
       
        query_embedding = self.model.encode([search_query], convert_to_tensor=False)
        
       
        where_clause = None
        if language_filter:
            where_clause = {"language": language_filter}
        
       
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where_clause
        )
        
      
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if 'distances' in results else 0
                
                similarity = 1 - distance
                
                formatted_results.append({
                    "question": results['metadatas'][0][i].get('question', ''),
                    "answer": results['documents'][0][i],
                    "confidence": float(similarity),
                    "metadata": results['metadatas'][0][i],
                    "id": results['ids'][0][i],
                    "source": "local" if similarity >= self.min_confidence else "local_low"
                })
        
        return formatted_results
    
    
    def find_answer(
        self,
        user_question: str,
        min_confidence: Optional[float] = None,
        context: str = ""
    ) -> Dict:
   
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        
        results = self.search(user_question, top_k=5, context=context)
        
        if not results:
            return {
                "question": None,
                "answer": None,
                "confidence": 0.0,
                "source": "none",
                "top_matches": []
            }
        
        best = results[0]
        confidence = best["confidence"]
        
      
        if context and confidence < min_confidence:
            results_no_context = self.search(user_question, top_k=3, context="")
            if results_no_context and results_no_context[0]["confidence"] > confidence:
                best = results_no_context[0]
                confidence = best["confidence"]
                results = results_no_context
        
        if confidence < min_confidence:
            return {
                "question": None,
                "answer": None,
                "confidence": confidence,
                "source": "none",
                "top_matches": results
            }
        
        return {
            "question": best["question"],
            "answer": best["answer"],
            "confidence": confidence,
            "source": "local",
            "metadata": best.get("metadata", {}),
            "top_matches": results
        }
    
    
    def add_qa_pair(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict] = None
    ) -> str:
      
        
        embedding = self.model.encode([question], convert_to_tensor=False)
        
       
        item_metadata = {
            "question": question,
            "language": self._detect_language(question),
            "length": len(answer)
        }
        if metadata:
            item_metadata.update(metadata)
        
        
        item_id = str(uuid.uuid4())
        
        
        self.collection.add(
            embeddings=embedding.tolist(),
            documents=[answer],
            metadatas=[item_metadata],
            ids=[item_id]
        )
        
        print(f"Added new Q&A pair. Total: {self.collection.count()}")
        return item_id
    
    
    def get_stats(self) -> Dict:
      
        return {
            "total_items": self.collection.count(),
            "collection_name": self.collection.name,
            "metadata": self.collection.metadata
        }
    
    
    def delete_by_id(self, item_id: str) -> None:
       
        self.collection.delete(ids=[item_id])
        print(f"🗑️ Deleted item {item_id}")


def build_rag(csv_path: Optional[Path] = None, reload: bool = False) -> QAEngineRag:
 
    base_dir = Path(__file__).resolve().parent.parent
    if csv_path is None:
        csv_path = base_dir / "data" / "knowledge_base.csv"
    
    engine = QAEngineRag(min_confidence=0.6)
    
 
    if engine.collection.count() == 0 or reload:
        engine.load_from_csv(csv_path, clear_existing=reload)
    
    return engine


if __name__ == "__main__":
   
    print(" ChromaDB Rag Test\n")
    
    engine = build_rag(reload=True)
    
    print("\n" + "="*60)
    print(" Stats:", engine.get_stats())
    print("="*60 + "\n")
    
   
    while True:
        question = input("Your question (or 'exit'): ")
        if question.lower() == 'exit':
            break
        
        result = engine.find_answer(question)
        
        print(f"\n{'='*60}")
        if result['answer']:
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Language: {result.get('metadata', {}).get('language', 'N/A')}")
        else:
            print(" No answer found")
            print(f"Best confidence: {result['confidence']:.1%}")
        print(f"{'='*60}\n")
