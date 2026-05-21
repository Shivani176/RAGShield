import sqlite3

conn = sqlite3.connect("papers.db")
cur = conn.cursor()

cur.execute("SELECT id, title, embedding_id FROM papers WHERE id LIKE 'poison%'")
rows = cur.fetchall()

print("Poison rows:")
for r in rows:
    print(r)

cur.execute("SELECT COUNT(*) FROM papers WHERE id LIKE 'poison%'")
print("Total poison rows:", cur.fetchone()[0])

conn.close()