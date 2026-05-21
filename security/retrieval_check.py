import sqlite3
conn = sqlite3.connect("papers.db")
cursor = conn.cursor()
cursor.execute("SELECT id, title FROM papers WHERE id = 'poison_citation_001'")
row = cursor.fetchone()
print("In SQLite:", row)
conn.close()

from memory_manager import MemoryManager
mm = MemoryManager()
results = mm.hybrid_search("transformer attention mechanisms", top_k=15)
print("\nTop 15 retrieved papers:")
for i, p in enumerate(results, 1):
    print(f"{i}. {p['title'][:60]} | score: {p.get('hybrid_score', 0):.3f}")

# import sqlite3
# conn = sqlite3.connect("papers.db")
# cursor = conn.cursor()

# # Check if any poison records exist at all
# cursor.execute("SELECT id, title FROM papers WHERE title LIKE '%Transformer Attention Mechanisms%'")
# rows = cursor.fetchall()
# print("Matching records:", rows)

# # Check the table schema
# cursor.execute("PRAGMA table_info(papers)")
# schema = cursor.fetchall()
# print("\nTable schema:")
# for col in schema:
#     print(col)

# conn.close()