import pandas as pd
from pathlib import Path

print(" دمج قواعد البيانات...\n")


base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"


original = pd.read_csv(data_dir / "knowledge_base.csv", encoding='utf-8')
print(f" الملف الأصلي: {len(original)} سؤال")


try:
    new_data = pd.read_csv(data_dir / "technical_qa.csv", encoding='utf-8')
    print(f" الملف الجديد: {len(new_data)} سؤال")
    
   
    if 'question' not in new_data.columns or 'answer' not in new_data.columns:
        print(" خطأ: الملف يجب أن يحتوي على عمودي question و answer")
        exit(1)
    
    
    merged = pd.concat([original, new_data], ignore_index=True)
    
   
    merged = merged.dropna(subset=['question', 'answer'])
    merged['question'] = merged['question'].str.strip()
    merged['answer'] = merged['answer'].str.strip()
    merged = merged.drop_duplicates(subset=['question'], keep='first')
    
  
    backup_path = data_dir / "knowledge_base_backup.csv"
    original.to_csv(backup_path, index=False, encoding='utf-8')
    print(f" نسخة احتياطية: {backup_path}")
    
    
    output_path = data_dir / "knowledge_base.csv"
    merged.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n marge acev succse!")
    print(f" كان: {len(original)} سؤال")
    print(f" أضيف: {len(new_data)} سؤال")
    print(f" أصبح: {len(merged)} سؤال")
    
except FileNotFoundError:
    print(" لم يتم العثور على batch_1.cs ")
except Exception as e:
    print(f" خطأ: {e}")
