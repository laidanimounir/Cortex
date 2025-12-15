import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class QAEngine:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = None

    def load_knowledge_base(self, csv_path):
        
        df = pd.read_csv(csv_path, encoding='utf-8')
        
       
        self.questions = df['question'].tolist()
        self.answers = df['answer'].tolist()
        
       
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        
        print(f"Loaded{len(self.questions)}Questions ")


    def find_answer(self, user_question):
       
        user_vector = self.vectorizer.transform([user_question])
        
      
        similarities = cosine_similarity(user_vector, self.question_vectors)[0]
        
      
        best_match_index = similarities.argmax()
        confidence = similarities[best_match_index]
        
      
        return {
              "question": self.questions[best_match_index],
              "answer": self.answers[best_match_index],
              "confidence": float(confidence)
           }

def build_qa_engine():
    from pathlib import Path
    
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "knowledge_base.csv"
    
    engine = QAEngine()
    engine.load_knowledge_base(csv_path)
    return engine

if __name__ == "__main__":
  
    engine = build_qa_engine()
    
    print(" Multilingual QA System")
    print("Enter your question in any language (type 'exit' to quit)\n")
    
    while True:
        question = input("Your question ")
        if question.lower() == 'exit':
            print("\nGoodbye!")
            break
        if not question.strip():
            print("Please enter a valid question\n")
            continue

        
        result = engine.find_answer(question)
        print(f"\nMatched Question {result['question']}")
        print(f" answer {result['answer']}")
        print(f" match score {result['confidence']:.2%}\n")

