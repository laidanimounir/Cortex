from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"

# نقرأ الملف الخام كما هو (فاصل ; )
raw = pd.read_csv(data_dir / "python_qa.csv",
                  encoding="utf-8",
                  sep=";",
                  header=None,
                  names=["col"])

# كل صف عبارة عن: question answer language category مفصولة بمسافة
rows = []
for text in raw["col"].dropna():
    parts = text.split("  ")  # جرّب مسافة مزدوجة، أو غيّرها حسب الواقع
    if len(parts) < 4:
        continue
    q = parts[0].strip()
    a = parts[1].strip()
    lang = parts[2].strip()
    cat = parts[3].strip()
    rows.append({"question": q, "answer": a, "language": lang, "category": cat})

clean = pd.DataFrame(rows)
clean.to_csv(data_dir / "python_qa_fixed.csv",
             index=False,
             encoding="utf-8")
print(len(clean), "rows cleaned")
