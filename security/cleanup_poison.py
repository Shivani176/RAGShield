import sqlite3
conn = sqlite3.connect("papers.db")
conn.execute("DELETE FROM papers WHERE id LIKE 'poison%'")
conn.commit()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM papers")
print("Papers remaining:", cursor.fetchone()[0])
conn.close()
print("Cleanup done.")

# import sqlite3

# conn = sqlite3.connect("papers.db")
# cursor = conn.cursor()

# # Total count
# cursor.execute("SELECT COUNT(*) FROM papers")
# print("Total papers:", cursor.fetchone()[0])

# # Check poison papers specifically
# cursor.execute("SELECT id, title, abstract FROM papers WHERE id LIKE 'poison%'")
# rows = cursor.fetchall()
# print(f"\nPoison papers found: {len(rows)}")
# for row in rows:
#     print(f"\nID: {row[0]}")
#     print(f"Title: {row[1]}")
#     print(f"Abstract preview: {row[2][:100]}...")

# # Last 10 added papers
# cursor.execute("SELECT id, title FROM papers ORDER BY rowid DESC LIMIT 10")
# rows = cursor.fetchall()
# print("\nLast 10 added papers:")
# for row in rows:
#     print(f"  {row[0]} — {row[1][:60]}")

# conn.close()