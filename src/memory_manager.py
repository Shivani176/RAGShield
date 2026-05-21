"""
Memory Manager - Central controller for RAG memory system
Handles both vector (ChromaDB) and structured (SQLite) storage
"""

import chromadb
from sentence_transformers import SentenceTransformer
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import time

from rank_bm25 import BM25Okapi
import numpy as np


class MemoryManager:
    """Manages all memory operations for the research assistant"""
    
    def __init__(self, 
                 chroma_path="./chroma_db_mpnet",
                 sqlite_path="papers.db",
                 embedding_model="all-mpnet-base-v2"):
        """
        Initialize the memory manager
        
        Args:
            chroma_path: Where to store ChromaDB data
            sqlite_path: Your existing papers.db path
            embedding_model: Sentence transformer model name
        """
        print("Initializing Memory Manager...")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Initialize embedding model (you already have this installed!)
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # SQLite connection (your existing database)
        self.sqlite_path = sqlite_path
        
        # Create/get collections
        self._init_collections()
        
        # Extend SQLite schema with new tables
        self._init_sqlite_tables()

        # Initialize BM25 for hybrid search
        self.bm25 = None
        self.bm25_paper_ids = []
        
        # Build BM25 index if papers exist
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL")
            paper_count = cursor.fetchone()[0]
            conn.close()
            
            if paper_count > 0:
                print("Building BM25 index...")
                self.build_bm25_index()
        except Exception as e:
            print(f"Could not build BM25 index: {e}")
        
        print("✓ Memory Manager initialized")
        

    
    def _init_collections(self):
        """Create ChromaDB collections (like tables)"""
        # Collection 1: Store all conversations
        self.conversations_collection = self.chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"description": "All user queries and agent responses"}
        )
        
        # Collection 2: Store paper embeddings
        self.papers_collection = self.chroma_client.get_or_create_collection(
            name="papers",
            metadata={"description": "Research paper embeddings", "hnsw:space": "cosine"}
        )
        
        # Collection 3: Store analysis results
        self.analyses_collection = self.chroma_client.get_or_create_collection(
            name="analyses",
            metadata={"description": "Bridge/connection analysis results"}
        )
        
        print(f"✓ Collections initialized: {self.conversations_collection.count()} conversations stored")
    
    def _init_sqlite_tables(self):
        """Add new tables to your existing papers.db"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Table 1: Conversation log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                user_query TEXT,
                agent_response TEXT,
                query_type TEXT,
                session_id TEXT,
                embedding_id TEXT
            )
        ''')
        
        # Table 2: Analysis log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_log (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                analysis_type TEXT,
                paper_ids TEXT,
                result_summary TEXT,
                full_result TEXT,
                embedding_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ SQLite tables initialized")
    
    # ============ CONVERSATION MEMORY ============
    
    def store_conversation(self, 
                          user_query: str, 
                          agent_response: str, 
                          query_type: str,
                          session_id: str = "default") -> str:
        """
        Store a conversation exchange in both vector and SQL storage
        
        Returns: The ID of the stored conversation
        """

        conv_id = self._generate_id("conv")
        timestamp = time.time()

        #combined text for embedding (query + response)
        combined_text = f"Query: {user_query}\nResponse: {agent_response}"
        embedding = self._create_embedding(combined_text) #creating embedding

        # Store in ChromaDB (vector storage)
        self.conversations_collection.add(
            documents=[combined_text],
            embeddings=[embedding],
            ids=[conv_id],
            metadatas=[{
                "query_type": query_type,
                "timestamp": timestamp,
                "session_id": session_id,
                "user_query": user_query[:500] #first 500 chars for quick reference
            }]
        )

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversation_log 
            (id, timestamp, user_query, agent_response, query_type, session_id, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (conv_id, timestamp, user_query, agent_response, query_type, session_id, conv_id))
        conn.commit()
        conn.close()

        return conv_id


    def search_conversations(self, 
                        query: str, 
                        top_k: int = 5,
                        filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search past conversations semantically
        
        Args:
            query: What to search for
            top_k: How many results to return
            filters: Optional filters (time range, query_type, etc)
            
        Returns: List of matching conversations
        """
        # Create embedding for the search query
        query_embedding = self._create_embedding(query)
        
        # Build ChromaDB filter if provided
        where_filter = None
        if filters:
            where_filter = {}
            if "query_type" in filters:
                where_filter["query_type"] = filters["query_type"]
            if "session_id" in filters:
                where_filter["session_id"] = filters["session_id"]
        
        # Search in vector database
        results = self.conversations_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        
        # Format results
        conversations = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                conv_id = results['ids'][0][i]
                
                # Get full details from SQLite
                conn = sqlite3.connect(self.sqlite_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_query, agent_response, query_type, timestamp, session_id
                    FROM conversation_log
                    WHERE id = ?
                ''', (conv_id,))
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    conversations.append({
                        'id': conv_id,
                        'user_query': row[0],
                        'agent_response': row[1],
                        'query_type': row[2],
                        'timestamp': row[3],
                        'session_id': row[4],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
        
        return conversations
    
    # ============ PAPER MEMORY ============
    
    def store_paper(self, paper_data: Dict) -> str:
        """
        Store paper in both vector DB and SQL
        
        Args:
            paper_data: Dict with keys: title, authors, abstract, year, source, arxiv_id, doi, id
        
        Returns: The embedding ID
        """

        embedding_id = self._generate_id("paper_emb")
        timestamp = time.time()

        #create text for embedding(title + abstract)    
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')

        combined_text = f"{title}\n{title}\n\n{abstract}"  # Title weighted 2x

        #create only if you have any text
        if len(combined_text.strip()) > 50:
            embedding = self._create_embedding(combined_text)

            self.papers_collection.add(
                documents=[combined_text],
                embeddings=[embedding],
                ids=[embedding_id],
                metadatas=[{
                    "paper_id": paper_data.get('id',''),
                    "title": title[:200],
                    "year": paper_data.get('year'),
                    "source": paper_data.get('source', ''),
                    "timestamp": timestamp 
                }]
            )
        else:

            #if no good abtract, still create an ID ut no embedding
            embedding_id = f"no_embedding_{embedding_id}"

        #store in sqlite (existing papers table already handles this)
        # just need to update it with the embedding_id

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(papers)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'embedding_id' not in columns:
            cursor.execute('ALTER TABLE papers ADD COLUMN embedding_id TEXT')

        # Update the paper with embedding_id
        cursor.execute('''
            UPDATE papers 
            SET embedding_id = ?
            WHERE id = ?
        ''', (embedding_id, paper_data.get('id')))

        conn.commit()
        conn.close()

        return embedding_id


    def search_papers(self,
                    query: str,
                    top_k: int = 10,
                    filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search papers semantically with optional SQL filters
        
        Args:
            query: Semantic search query
            top_k: Number of results
            filters: SQL filters like {"year": 2023, "source": "arxiv", "min_year": 2020}
            
        Returns: List of matching papers with similarity scores
        """
        # Step 1: Apply SQL filters first (if provided)
        candidate_paper_ids = None
        
        if filters:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Build SQL query with filters
            where_clauses = []
            params = []
            
            if "year" in filters:
                where_clauses.append("year = ?")
                params.append(filters["year"])
            
            if "min_year" in filters:
                where_clauses.append("year >= ?")
                params.append(filters["min_year"])
            
            if "max_year" in filters:
                where_clauses.append("year <= ?")
                params.append(filters["max_year"])
            
            if "source" in filters:
                where_clauses.append("source = ?")
                params.append(filters["source"])
            
            # Execute SQL filter
            sql = "SELECT id, embedding_id FROM papers"
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            
            cursor.execute(sql, params)
            filtered_results = cursor.fetchall()
            conn.close()
            
            # Get list of embedding IDs that passed SQL filters
            candidate_paper_ids = [row[1] for row in filtered_results if row[1] and not row[1].startswith('no_embedding')]
            
            if not candidate_paper_ids:
                return []  # No papers match the filters
        
        # Step 2: Create embedding for search query
        query_embedding = self._create_embedding(query)
        
        # Step 3: Search in vector database
        search_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k if not candidate_paper_ids else min(top_k, len(candidate_paper_ids))
        }
        
        # If we have filtered candidates, only search among those
        if candidate_paper_ids:
            search_params["ids"] = candidate_paper_ids
        
        try:
            results = self.papers_collection.query(**search_params)
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
        
        # Step 4: Format results and get full details from SQL
        papers = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                embedding_id = results['ids'][0][i]
                
                # Get full paper details from SQLite
                conn = sqlite3.connect(self.sqlite_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, title, authors, abstract, year, source, arxiv_id, doi
                    FROM papers
                    WHERE embedding_id = ?
                ''', (embedding_id,))
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    papers.append({
                        'id': row[0],
                        'title': row[1],
                        'authors': json.loads(row[2]) if row[2] else [],
                        'abstract': row[3],
                        'year': row[4],
                        'source': row[5],
                        'arxiv_id': row[6],
                        'doi': row[7],
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'embedding_id': embedding_id
                    })
        
        return papers
            
    def build_bm25_index(self):
        """
        Build BM25 index for keyword-based search
        Call this once when initializing or when papers are updated
        """
        print("Building BM25 index...")
        
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Get all papers with abstracts
        cursor.execute("SELECT id, title, abstract FROM papers WHERE abstract IS NOT NULL")
        papers = cursor.fetchall()
        conn.close()
        
        if not papers:
            print("⚠️  No papers found for BM25 indexing!")
            return
        
        # Store paper IDs for lookup
        self.bm25_paper_ids = [p[0] for p in papers]
        
        # Create corpus: combine title + abstract for each paper
        corpus = []
        for paper in papers:
            paper_id, title, abstract = paper
            # Combine title and abstract
            text = f"{title} {abstract}".lower()
            corpus.append(text)
        
        # Tokenize corpus (simple split by whitespace)
        tokenized_corpus = [doc.split() for doc in corpus]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"✓ BM25 index built with {len(papers)} papers")

    def bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Keyword-based search using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of papers with BM25 scores
        """
        # Build index if not exists
        if not hasattr(self, 'bm25') or self.bm25 is None:
            self.build_bm25_index()
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Fetch full paper details from database
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        results = []
        for idx in top_indices:
            paper_id = self.bm25_paper_ids[idx]
            bm25_score = float(scores[idx])
            
            # Skip if score is too low (no relevance)
            if bm25_score < 0.1:
                continue
            
            # Get paper details
            cursor.execute("""
                SELECT id, title, authors, abstract, year, source, arxiv_id, doi
                FROM papers WHERE id = ?
            """, (paper_id,))
            
            row = cursor.fetchone()
            if row:
                results.append({
                    'id': row[0],
                    'title': row[1],
                    'authors': row[2],
                    'abstract': row[3],
                    'year': row[4],
                    'source': row[5],
                    'arxiv_id': row[6],
                    'doi': row[7],
                    'bm25_score': bm25_score,
                    'search_method': 'bm25'
                })
        
        conn.close()
        return results

    """
    HYBRID_SEARCH FIX - READY TO APPLY
    Replace lines 513-616 in memory_manager.py with this function
    """

    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10, 
        alpha: float = 0.5,
        filters: Optional[Dict] = None,
        min_score: float = 0.15
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 (keyword) and semantic (embedding) search
        
        Args:
            query: Search query
            top_k: Number of final results
            alpha: Weight for BM25 (0.0-1.0)
                0.0 = pure semantic
                0.5 = balanced (recommended)
                1.0 = pure BM25
            filters: Optional SQL filters like {"min_year": 2020}
            min_score: Minimum combined score threshold (default 0.15)
        
        Returns:
            List of papers ranked by combined score
        """
        # Get more results from each method for better fusion
        retrieval_k = top_k * 3  # Get 3x results for reranking
        
        # Step 1: BM25 search
        bm25_results = self.bm25_search(query, top_k=retrieval_k)
        
        # Step 2: Semantic search
        semantic_results = self.search_papers(query, top_k=retrieval_k, filters=filters)
        
        # Step 3: Normalize scores to 0-1 range
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(r['bm25_score'] for r in bm25_results)
            min_bm25 = min(r['bm25_score'] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
            
            for r in bm25_results:
                r['normalized_bm25'] = (r['bm25_score'] - min_bm25) / bm25_range
        
        # Normalize semantic scores (already 0-1 from similarity)
        for r in semantic_results:
            r['normalized_semantic'] = r.get('similarity', 0.0)
        
        # Step 4: Combine scores with FILTERING
        combined_scores = {}
        
        # Add BM25 scores (with threshold)
        for r in bm25_results:
            paper_id = r['id']
            bm25_score = r.get('normalized_bm25', 0.0)
            
            # Only keep if BM25 score is meaningful
            if bm25_score > 0.1:
                combined_scores[paper_id] = {
                    'paper': r,
                    'bm25_score': bm25_score,
                    'semantic_score': 0.0,
                    'combined_score': alpha * bm25_score
                }
        
        # Add semantic scores (with threshold)
        for r in semantic_results:
            paper_id = r['id']
            semantic_score = r.get('normalized_semantic', 0.0)
            
            if paper_id in combined_scores:
                # Paper found in both searches - add semantic score
                combined_scores[paper_id]['semantic_score'] = semantic_score
                combined_scores[paper_id]['combined_score'] += (1 - alpha) * semantic_score
            elif semantic_score > 0.3:  # Only if strong semantic match
                # Paper only in semantic search but with strong match
                combined_scores[paper_id] = {
                    'paper': r,
                    'bm25_score': 0.0,
                    'semantic_score': semantic_score,
                    'combined_score': (1 - alpha) * semantic_score
                }
        
        # CRITICAL FIX: Filter by minimum combined score
        combined_scores = {
            pid: data for pid, data in combined_scores.items()
            if data['combined_score'] >= min_score
        }
        
        # Return empty if no papers meet threshold
        if not combined_scores:
            return []
        
        # Step 5: Rank by combined score
        ranked = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        
        # Step 6: Format final results
        final_results = []
        for i, item in enumerate(ranked, 1):
            paper = item['paper'].copy()
            paper['rank'] = i
            paper['bm25_score'] = item['bm25_score']
            paper['semantic_score'] = item['semantic_score']
            paper['hybrid_score'] = item['combined_score']
            paper['search_method'] = 'hybrid'
            
            # Determine which method contributed more
            if item['bm25_score'] > item['semantic_score']:
                paper['primary_match'] = 'keyword'
            elif item['semantic_score'] > item['bm25_score']:
                paper['primary_match'] = 'semantic'
            else:
                paper['primary_match'] = 'balanced'
            
            final_results.append(paper)
        
        return final_results

    def compare_search_methods(self, query: str, top_k: int = 5) -> Dict:
        """
        Compare BM25, semantic, and hybrid search side-by-side
        Useful for evaluation and thesis
        
        Args:
            query: Search query
            top_k: Number of results per method
        
        Returns:
            Dictionary with results from each method
        """
        print(f"\n{'='*70}")
        print(f"SEARCH METHOD COMPARISON")
        print(f"Query: '{query}'")
        print(f"{'='*70}\n")
        
        # Run all three searches
        bm25_results = self.bm25_search(query, top_k=top_k)
        semantic_results = self.search_papers(query, top_k=top_k)
        hybrid_results = self.hybrid_search(query, top_k=top_k, alpha=0.5)
        
        comparison = {
            'query': query,
            'bm25_results': bm25_results,
            'semantic_results': semantic_results,
            'hybrid_results': hybrid_results,
            'bm25_count': len(bm25_results),
            'semantic_count': len(semantic_results),
            'hybrid_count': len(hybrid_results)
        }
        
        # Print comparison
        print("\n" + "="*70)
        print("METHOD 1: BM25 (Keyword Search)")
        print("="*70)
        for i, paper in enumerate(bm25_results[:3], 1):
            print(f"{i}. {paper['title'][:60]}...")
            print(f"   Score: {paper['bm25_score']:.3f}")
        
        print("\n" + "="*70)
        print("METHOD 2: Semantic (Embedding Search)")
        print("="*70)
        for i, paper in enumerate(semantic_results[:3], 1):
            print(f"{i}. {paper['title'][:60]}...")
            print(f"   Similarity: {paper.get('similarity', 0):.3f}")
        
        print("\n" + "="*70)
        print("METHOD 3: Hybrid (BM25 + Semantic)")
        print("="*70)
        for i, paper in enumerate(hybrid_results[:3], 1):
            print(f"{i}. {paper['title'][:60]}...")
            print(f"   Hybrid: {paper['hybrid_score']:.3f} "
                  f"(BM25: {paper['bm25_score']:.3f}, "
                  f"Semantic: {paper['semantic_score']:.3f})")
        
        print("\n" + "="*70 + "\n")
        
        return comparison
        
    # ============ ANALYSIS MEMORY ============
    
    def store_analysis(self,
                      analysis_type: str,
                      paper_ids: List[str],
                      result: str) -> str:
        """Store an analysis result"""
        # TODO: Implement this (Day 8)
        pass
    
    def search_analyses(self,
                       query: str,
                       top_k: int = 5) -> List[Dict]:
        """Search past analyses"""
        # TODO: Implement this (Day 9)
        pass
    
    # ============ HELPER METHODS ============
    
    def _create_embedding(self, text: str) -> List[float]:
        """Convert text to embedding vector"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = int(time.time() * 1000)
        return f"{prefix}_{timestamp}"
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "conversations": self.conversations_collection.count(),
            "papers": self.papers_collection.count(),
            "analyses": self.analyses_collection.count()
        }

