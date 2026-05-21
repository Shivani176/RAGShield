# import sqlite3
# import os
# import sys
# sys.path.append(".")

# from memory_manager import MemoryManager

# # Check which papers.db the memory manager uses
# mm = MemoryManager()
# print("MemoryManager sqlite path:", mm.sqlite_path)

# # Check current directory
# print("Current directory:", os.getcwd())

# # Check if papers.db exists in current directory
# print("papers.db exists here:", os.path.exists("papers.db"))

# # Check the count in the memory manager's db
# conn = sqlite3.connect(mm.sqlite_path)
# cursor = conn.cursor()
# cursor.execute("SELECT COUNT(*) FROM papers")
# count = cursor.fetchone()[0]
# print("Papers in MemoryManager db:", count)

# cursor.execute("SELECT id, title FROM papers WHERE id = 'poison_citation_001'")
# row = cursor.fetchone()
# print("Poison paper in MemoryManager db:", row)
# conn.close()

import sys
sys.path.append(".")
from memory_manager import MemoryManager

mm = MemoryManager()
results = mm.hybrid_search("transformer attention mechanisms", top_k=15)
print("Top 15 retrieved papers:")
for i, p in enumerate(results, 1):
    print(f"{i}. {p['title'][:60]} | score: {p.get('hybrid_score', 0):.3f}")