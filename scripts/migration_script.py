
from memory_manager import MemoryManager
from tools import PaperDatabase

print("Migrating papers to Memory Manager...")

db = PaperDatabase()
memory = MemoryManager()

papers = db.get_stored_papers(limit=10000)  # Get ALL papers
print(f"Found {len(papers)} papers to migrate")

migrated = 0
for paper in papers:
    try:
        memory.store_paper(paper)
        migrated += 1
        if migrated % 10 == 0:
            print(f"  Migrated {migrated}/{len(papers)}...")
    except Exception as e:
        print(f"  Skipped paper (already exists or error): {paper['title'][:40]}")

print(f"\nMigration complete: {migrated} papers migrated")
print(f"Final stats: {memory.get_stats()}")