"""
Quick Diagnosis Script - Test Synthesis Engine
"""

# Test 1: Check if papers.db has data
import sqlite3
print("="*70)
print("TEST 1: Checking papers.db")
print("="*70)

try:
    conn = sqlite3.connect('papers.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM papers")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL")
    with_abs = cursor.fetchone()[0]
    
    print(f"✓ Total papers: {total}")
    print(f"✓ Papers with abstracts: {with_abs}")
    
    # Get a sample paper
    cursor.execute("SELECT title, abstract FROM papers WHERE abstract IS NOT NULL LIMIT 1")
    sample = cursor.fetchone()
    
    if sample:
        print(f"\n✓ Sample paper:")
        print(f"  Title: {sample[0][:60]}...")
        print(f"  Abstract length: {len(sample[1])} chars")
    
    conn.close()
    print("\n✓ Papers.db is working!\n")
    
except Exception as e:
    print(f"❌ Error accessing papers.db: {e}\n")


# Test 2: Check if ChromaDB is accessible
print("="*70)
print("TEST 2: Checking ChromaDB")
print("="*70)

try:
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db_mpnet")
    collections = client.list_collections()
    print(f"✓ ChromaDB accessible")
    print(f"✓ Collections found: {len(collections)}")
    for col in collections:
        print(f"  - {col.name}: {col.count()} items")
    print("\n✓ ChromaDB is working!\n")
    
except Exception as e:
    print(f"❌ Error accessing ChromaDB: {e}\n")


# Test 3: Check if memory_manager works
print("="*70)
print("TEST 3: Testing Memory Manager")
print("="*70)

try:
    from memory_manager import MemoryManager
    
    memory = MemoryManager()
    stats = memory.get_stats()
    
    print(f"✓ Memory Manager initialized")
    print(f"✓ Stats: {stats}")
    
    # Try hybrid search
    papers = memory.hybrid_search("transformers", top_k=5)
    print(f"\n✓ Hybrid search found {len(papers)} papers")
    
    if papers:
        print(f"  Top result: {papers[0]['title'][:60]}...")
    
    print("\n✓ Memory Manager is working!\n")
    
except Exception as e:
    print(f"❌ Error with Memory Manager: {e}\n")
    import traceback
    traceback.print_exc()


# Test 4: Check Anthropic API
print("="*70)
print("TEST 4: Testing Anthropic API")
print("="*70)

try:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in .env file!")
        print("   This is likely the problem!")
    else:
        print(f"✓ ANTHROPIC_API_KEY found: {api_key[:20]}...{api_key[-4:]}")
        
        # Try a simple API call
        from langchain_anthropic import ChatAnthropic
        
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            max_tokens=100
        )
        
        response = llm.invoke("Say 'API test successful' and nothing else.")
        print(f"✓ API Response: {response.content}")
        print("\n✓ Anthropic API is working!\n")
        
except Exception as e:
    print(f"❌ Error with Anthropic API: {e}\n")
    import traceback
    traceback.print_exc()


# Test 5: Test synthesis engine directly
print("="*70)
print("TEST 5: Testing Synthesis Engine Directly")
print("="*70)

try:
    from synthesis_engine import SynthesisEngine
    from memory_manager import MemoryManager
    
    memory = MemoryManager()
    engine = SynthesisEngine()
    
    # Get some papers
    papers = memory.hybrid_search("transformers", top_k=3)
    
    if not papers:
        print("❌ No papers found for 'transformers'")
    else:
        print(f"✓ Found {len(papers)} papers")
        
        # Try to generate review
        print("  Attempting to generate review...")
        
        result = engine.generate_literature_review(
            papers=papers,
            query="transformers",
            max_papers=3,
            include_gaps=False  # Simpler test
        )
        
        if 'error' in result:
            print(f"❌ Synthesis error: {result['error']}")
        else:
            print(f"✓ Review generated!")
            print(f"  Length: {len(result['review_text'])} chars")
            print(f"  Papers cited: {len(result['citations_used'])}/{result['paper_count']}")
            print(f"\n  First 200 chars of review:")
            print(f"  {result['review_text'][:200]}...")
        
    print("\n✓ Synthesis Engine test complete!\n")
    
except Exception as e:
    print(f"❌ Error with Synthesis Engine: {e}\n")
    import traceback
    traceback.print_exc()


print("="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nIf all tests passed, the system should work.")
print("If any test failed, that's where the problem is.")