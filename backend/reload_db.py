from rag_engine import build_rag
from pathlib import Path

print(" إعادة تحميل ChromaDB...\n")

base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "data" / "knowledge_base.csv"


engine = build_rag(csv_path=csv_path, reload=True)

print(f"\n dawnload !")
stats = engine.get_stats()
print(f" total quetion {stats['total_items']}")
print(f" nom de groupe  {stats['collection_name']}")
print(f" ready to use!")
