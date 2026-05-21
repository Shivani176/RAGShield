"""
Upgrade memory manager to use better embeddings
This script re-embeds all papers with the better model
"""

import sqlite3
from sentence_transformers import SentenceTransformer
import time
from memory_manager import MemoryManager

print("="*70)
print("EMBEDDING UPGRADE SCRIPT")
print("="*70)
print()

# Step 1: Check current papers
print("Step 1: Checking database...")
conn = sqlite3.connect("papers.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL")
paper_count = cursor.fetchone()[0]
conn.close()

print(f"  Found {paper_count} papers with abstracts")
print()

# Step 2: Initialize new memory manager with better model
print("Step 2: Initializing new memory manager...")
print("  Model: all-mpnet-base-v2 (768 dimensions)")
print("  Embedding: title + abstract (title weighted 2x)")
print()

# Create new memory manager with upgraded model
memory = MemoryManager(
    chroma_path="./chroma_db_mpnet",  # New database to avoid confusion
    sqlite_path="papers.db",
    embedding_model="all-mpnet-base-v2"
)

print("✓ Memory manager created with new model")
print()

# Step 3: Re-embed all papers
print("Step 3: Re-embedding all papers...")
print("  This will take about 2-5 minutes for 456 papers")
print()

conn = sqlite3.connect("papers.db")
cursor = conn.cursor()
cursor.execute("""
    SELECT id, title, authors, abstract, year, source, arxiv_id, doi
    FROM papers 
    WHERE abstract IS NOT NULL
""")
papers = cursor.fetchall()
conn.close()

start_time = time.time()
success_count = 0
error_count = 0

for i, paper in enumerate(papers, 1):
    try:
        paper_data = {
            'id': paper[0],
            'title': paper[1],
            'authors': paper[2],
            'abstract': paper[3],
            'year': paper[4],
            'source': paper[5],
            'arxiv_id': paper[6],
            'doi': paper[7]
        }
        
        memory.store_paper(paper_data)
        success_count += 1
        
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(papers) - i) / rate
            print(f"  Progress: {i}/{len(papers)} ({i/len(papers)*100:.1f}%) - "
                  f"ETA: {remaining:.0f}s")
    
    except Exception as e:
        error_count += 1
        if error_count < 5:  # Only show first few errors
            print(f"  Error on paper {i}: {e}")

elapsed = time.time() - start_time
print()
print("="*70)
print("UPGRADE COMPLETE")
print("="*70)
print(f"  Successfully embedded: {success_count} papers")
print(f"  Errors: {error_count} papers")
print(f"  Time taken: {elapsed:.1f} seconds")
print(f"  Rate: {success_count/elapsed:.1f} papers/second")
print()
print(f"New database location: ./chroma_db_mpnet")
print()

# Step 4: Verify
print("Step 4: Verifying embeddings...")
stats = memory.get_stats()
print(f"  Papers in vector DB: {stats['papers']}")
print()

if stats['papers'] >= paper_count * 0.95:  # Allow 5% failure
    print("✓ Upgrade successful!")
    print()
    print("NEXT STEPS:")
    print("1. Update memory_manager.py to use 'all-mpnet-base-v2'")
    print("2. Update memory_manager.py to point to './chroma_db_mpnet'")
    print("3. Test with a query")
else:
    print("⚠ Warning: Some papers may not have been embedded")
    print("  Review errors above")

print()
print("="*70)