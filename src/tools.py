import os, tempfile
import json
from dotenv import load_dotenv
load_dotenv()  # ensure .env is read even if main forgets

from langchain_community.tools import WikipediaQueryRun
from langchain_tavily import TavilySearch
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import Tool
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import requests
import arxiv
from langchain_community.tools import Tool
from typing import Literal
        
# Dynamic domain clustering using embedding similarity
from sklearn.cluster import KMeans
import numpy as np
        
import sqlite3
import json
import networkx as nx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bibtex_export import export_from_database, export_papers_to_bibtex
from output_manager import get_output_manager



# Global memory manager reference
_memory_manager = None

def set_memory_manager(manager):
    """Allow main.py to inject the memory manager"""
    global _memory_manager
    _memory_manager = manager


# --Adding various tools--
#  Tools are the things that the LLM/ agent can use that we either write
#  ourself or we can bring in from things like Langchain community hub.



""" 1. search tool """

search = TavilySearch()
search_tool = Tool(
    name = "search",
    func= search.run,
    description= "Search the web for information"
)

""" 2. wikipedia tool """

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

""" 3. personal custom tool: save o/p to txt file (overwrite | append | versioned)"""

def save_to_txt(
    data: str,
    filename: str = "research_output.txt",
    mode: Literal["overwrite", "append", "versioned"] = "overwrite",
) -> str:
    """
    Save research output to a text file.
    Only saves when user explicitly requests it in their query.
    """
    # Check if save was requested in this turn
    allow_save = os.getenv("ALLOW_SAVE_THIS_TURN")
    
    if allow_save != "1":
        return "Save not requested: User did not explicitly ask to save in this query."
    
    # Organize output based on file type
    from output_manager import get_output_manager
    manager = get_output_manager()
    
    if filename.endswith('.bib'):
        filename = manager.get_bibtex_path(filename)
    else:
        filename = manager.get_search_path(filename)
    
    # Proceed with save since it was requested
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = f" --- Research Output ---\nTimestamp: {ts}\n\n{data}\n\n"

    try:
        if mode == "overwrite":
            dir_ = os.path.dirname(filename) or "."
            with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_, encoding="utf-8") as tmp:
                tmp.write(payload)
                tmp_path = tmp.name
            os.replace(tmp_path, filename)
            return f"File saved successfully: {os.path.abspath(filename)}"

        elif mode == "append":
            with open(filename, "a", encoding="utf-8") as f:
                f.write(payload)
            return f"Content appended to: {os.path.abspath(filename)}"

        else:  # "versioned"
            stamp = ts.replace(":", "-").replace(" ", "_")
            root, ext = os.path.splitext(filename)
            versioned_name = f"{root}_{stamp}{ext or '.txt'}"
            with open(versioned_name, "w", encoding="utf-8") as f:
                f.write(payload)
            return f"New versioned file created: {os.path.abspath(versioned_name)}"
            
    except Exception as e:
        return f"Error saving file: {str(e)}"

class SaveArgs(BaseModel):
    data: str = Field(..., description="Exact text (JSON/summary) to write.")
    filename: str = Field("research_output.txt", description="Target file path.")
    mode: Literal["overwrite", "append", "versioned"] = Field(
        "overwrite",
        description="How to save: overwrite | append | versioned"
    )

save_tool = StructuredTool.from_function(
    name="save_text_to_file",
    description="Write research output to disk. Use ONLY if the user explicitly asks to save.",
    func=save_to_txt,
    args_schema=SaveArgs,
    return_direct=False,
)   

"""Search ArXiv for scholarly papers.
    Returns a list of dicts with keys: source, title, authors, abstract, year, pdf_url, id
"""
def arxiv_search(query: str, max_results: int = 8):
    # Create database instance
    db = PaperDatabase()
    
    try:
        s = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        out = []
        for r in s.results():
            paper_data = {
                "source": "arxiv",
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "abstract": (r.summary or "").strip(),
                "year": (r.published.year if getattr(r, "published", None) else None),
                "pdf_url": r.pdf_url,
                "id": r.entry_id,
                "arxiv_id": r.entry_id,
                "doi": None
            }
            
            # Store in database
            db.store_paper(paper_data)
            
            # Store in memory manager if available
            if _memory_manager:
                try:
                    _memory_manager.store_paper(paper_data)
                except Exception:
                    pass  # Silently fail if memory storage has issues
            
            out.append(paper_data)
        
        return out  # ← YOU WERE MISSING THIS!
        
    except Exception as e:
        return {"error": f"arxiv_search failed: {e}"}

"""
OpenAlex returns an 'abstract_inverted_index' dict: {word: [positions]}.
    Reconstruct to a plain string; return None if not reconstructable.
"""

def _flatten_openalex_abstract(inv_idx):

    if not inv_idx:
        return None
    try:
        n = max(max(pos_list) for pos_list in inv_idx.values()) + 1
        arr = [""] * n
        for word, positions in inv_idx.items():
            for p in positions:
                arr[p] = word
        return " ".join(arr)
    except Exception:
        return None


"""
    Search OpenAlex for works.
    Returns a list of dicts with keys: source, title, authors, abstract, year, doi, openalex_id, pdf_url
"""
def openalex_search(query: str, per_page: int = 8, mailto: str = "shivanikalal6@gmail.com"):
    try:
        url = "https://api.openalex.org/works"
        params = {"search": query, "per_page": per_page, "mailto": mailto}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        items = []
        data = r.json().get("results", [])
        
        # Store in database first
        db = PaperDatabase()
        
        for w in data:
            abstract_txt = _flatten_openalex_abstract(w.get("abstract_inverted_index"))
            primary_loc = (w.get("primary_location") or {})
            paper_data = {
                "source": "openalex",
                "title": w.get("title"),
                "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
                "abstract": abstract_txt,
                "year": w.get("publication_year"),
                "doi": w.get("doi"),
                "openalex_id": w.get("id"),
                "pdf_url": primary_loc.get("pdf_url") or primary_loc.get("landing_page_url"),
                "id": w.get("id")  # Add ID for consistency
            }
            
            # Store in SQL database
            db.store_paper(paper_data)
            
            # Store in memory manager if available
            if _memory_manager:
                try:
                    _memory_manager.store_paper(paper_data)
                except Exception:
                    pass
            
            items.append(paper_data)
        
        return items
    except Exception as e:
        return {"error": f"openalex_search failed: {e}"}

# Wrap them as LangChain Tools so your agent can call them directly
arxiv_tool = Tool(
    name="arxiv_search",
    func=arxiv_search,
    description="""Search ArXiv repository for NEW papers from the internet (not local database).
    Use this when the user wants to:
    - Find new/recent papers not yet in the database
    - Search the broader scientific literature
    - Discover papers to add to their collection
    This fetches fresh results from ArXiv.org, not the local database."""
)

openalex_tool = Tool(
    name="openalex_search",
    func=openalex_search,
    description="""Search OpenAlex database for NEW papers from the internet (not local database).
    Use this when the user wants to:
    - Find new papers not yet in the database
    - Search broader academic literature beyond ArXiv
    - Discover papers from various sources
    This fetches fresh results from OpenAlex API, not the local database."""
)


class PaperDatabase:
    def __init__(self, db_path="papers.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table to store paper information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                abstract TEXT,
                year INTEGER,
                source TEXT,
                arxiv_id TEXT,
                doi TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_paper(self, paper_data):
        """Store a single paper in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO papers 
            (id, title, authors, abstract, year, source, arxiv_id, doi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            paper_data.get('id'),
            paper_data.get('title'),
            json.dumps(paper_data.get('authors', [])),  # Store as JSON string
            paper_data.get('abstract'),
            paper_data.get('year'),
            paper_data.get('source'),
            paper_data.get('arxiv_id'),
            paper_data.get('doi')
        ))
        
        conn.commit()
        conn.close()

    def get_stored_papers(self, limit=100):
        """Retrieve stored papers from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM papers LIMIT ?', (limit,))
        rows = cursor.fetchall()
        
        papers = []
        for row in rows:
            papers.append({
                'id': row[0],
                'title': row[1], 
                'authors': json.loads(row[2]),
                'abstract': row[3],
                'year': row[4],
                'source': row[5],
                'arxiv_id': row[6],
                'doi': row[7]
            })
        
        conn.close()
        return papers


def check_stored_papers(query: str = ""):
    """Check what papers are stored in our database"""
    db = PaperDatabase()
    papers = db.get_stored_papers(limit=100)
    
    if not papers:
        return "No papers stored in database yet."
    
    result = f"Found {len(papers)} papers in database:\n\n"
    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(paper['authors'][:2])  # First 2 authors
        if len(paper['authors']) > 2:
            authors_str += " et al."
        result += f"{i}. {paper['title']}\n"
        result += f"   Authors: {authors_str}\n"
        result += f"   Year: {paper['year']}, Source: {paper['source']}\n\n"
    
    return result

# Create the tool
stored_papers_tool = Tool(
    name="check_stored_papers",
    func=check_stored_papers,
    description="""List all papers currently stored in the local database (simple list, no search/ranking).
    Use this when the user wants to:
    - See what papers they have
    - Get an overview of their collection
    - List their stored papers
    For SEARCHING stored papers by relevance, use hybrid_paper_search instead."""
)

""" adding grapgh functionality - to find connection between papers - mapping function"""

# Replace these functions in your tools.py with detailed output versions

def find_paper_connections(query: str = ""):
    """Find connections between stored papers using title similarity - DETAILED VERSION"""
    db = PaperDatabase()
    papers = db.get_stored_papers()
    
    if len(papers) < 2:
        return "Need at least 2 papers to find connections."
    
    connections = []
    
    # Simple keyword-based connection finding
    for i, paper1 in enumerate(papers):
        for j, paper2 in enumerate(papers[i+1:], i+1):
            # Find common keywords in titles
            title1_words = set(paper1['title'].lower().split())
            title2_words = set(paper2['title'].lower().split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'using', 'via'}
            title1_words -= stop_words
            title2_words -= stop_words
            
            common_words = title1_words.intersection(title2_words)
            
            if len(common_words) >= 2:  # At least 2 common meaningful words
                connections.append({
                    'paper1': paper1['title'],
                    'paper2': paper2['title'],
                    'paper1_year': paper1['year'],
                    'paper2_year': paper2['year'],
                    'common_words': list(common_words),
                    'strength': len(common_words)
                })
    
    if not connections:
        return "No strong connections found between stored papers."
    
    # Sort by connection strength
    connections.sort(key=lambda x: x['strength'], reverse=True)
    
    # DETAILED OUTPUT FORMAT
    result = f"TITLE-BASED PAPER CONNECTIONS ANALYSIS\n"
    result += f"Total papers analyzed: {len(papers)}\n"
    result += f"Connections found: {len(connections)}\n\n"
    
    result += "DETAILED CONNECTION LIST:\n"
    for i, conn in enumerate(connections[:15], 1):  # Show top 15
        result += f"{i}. CONNECTION STRENGTH: {conn['strength']} common words\n"
        result += f"   Paper 1: {conn['paper1']} ({conn['paper1_year']})\n"
        result += f"   Paper 2: {conn['paper2']} ({conn['paper2_year']})\n"
        result += f"   Common themes: {', '.join(conn['common_words'])}\n"
        result += f"   Connection score: {conn['strength']}/10\n\n"
    
    if len(connections) > 15:
        result += f"... and {len(connections) - 15} additional connections found.\n\n"
    
    # Add summary statistics
    strength_distribution = {}
    for conn in connections:
        strength = conn['strength']
        strength_distribution[strength] = strength_distribution.get(strength, 0) + 1
    
    result += "CONNECTION STRENGTH DISTRIBUTION:\n"
    for strength in sorted(strength_distribution.keys(), reverse=True):
        count = strength_distribution[strength]
        result += f"  {strength} common words: {count} connections\n"
    
    return result

connections_tool = Tool(
    name="find_paper_connections",
    func=find_paper_connections,
    description="Find connections and relationships between stored papers based on title similarity and common themes."
)


def find_semantic_connections(query: str = ""):
    """Find connections using semantic similarity of abstracts - DETAILED VERSION"""
    db = PaperDatabase()
    papers = db.get_stored_papers()
    
    if len(papers) < 2:
        return "Need at least 2 papers for semantic analysis."
    
    # Load embedding model
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        return f"Error loading embedding model: {e}"
    
    # Filter papers with good abstracts
    valid_papers = []
    abstracts = []
    
    for paper in papers:
        if paper.get('abstract') and len(paper['abstract'].strip()) > 50:
            abstracts.append(paper['abstract'])
            valid_papers.append({
                'title': paper['title'],
                'year': paper['year'],
                'id': paper.get('id', 'unknown'),
                'full_title': paper['title']
            })
    
    if len(valid_papers) < 2:
        return f"Need at least 2 papers with substantial abstracts. Found {len(valid_papers)} valid papers from {len(papers)} total."
    
    try:
        # Compute embeddings
        embeddings = model.encode(abstracts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find connections above threshold
        connections = []
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.3:  # Meaningful similarity threshold
                    connections.append({
                        'paper1': valid_papers[i],
                        'paper2': valid_papers[j],
                        'similarity': float(similarity)
                    })
        
        if not connections:
            return "No strong semantic connections found (similarity > 0.3). Papers may be from very different research areas."
        
        # Sort by similarity score
        connections.sort(key=lambda x: x['similarity'], reverse=True)
        
        # DETAILED OUTPUT FORMAT
        result = f"SEMANTIC SIMILARITY ANALYSIS RESULTS\n"
        result += f"Papers processed: {len(valid_papers)}\n"
        result += f"Embedding dimensions: {embeddings.shape[1]}\n"
        result += f"Connections found (similarity > 0.3): {len(connections)}\n\n"
        
        result += "TOP SEMANTIC CONNECTIONS (sorted by similarity):\n"
        for i, conn in enumerate(connections[:20], 1):  # Show top 20
            result += f"{i}. SIMILARITY SCORE: {conn['similarity']:.4f}\n"
            result += f"   Paper A: {conn['paper1']['title']} ({conn['paper1']['year']})\n"
            result += f"   Paper B: {conn['paper2']['title']} ({conn['paper2']['year']})\n"
            
            # Add similarity interpretation
            if conn['similarity'] > 0.7:
                interpretation = "Very High Similarity"
            elif conn['similarity'] > 0.5:
                interpretation = "High Similarity"
            elif conn['similarity'] > 0.4:
                interpretation = "Moderate Similarity"
            else:
                interpretation = "Low-Moderate Similarity"
            
            result += f"   Interpretation: {interpretation}\n"
            result += f"   Cosine similarity: {conn['similarity']:.6f}\n\n"
        
        if len(connections) > 20:
            result += f"... and {len(connections) - 20} additional connections found.\n\n"
        
        # Add detailed statistics
        high_sim = sum(1 for c in connections if c['similarity'] > 0.5)
        medium_sim = sum(1 for c in connections if 0.4 <= c['similarity'] <= 0.5)
        low_sim = sum(1 for c in connections if 0.3 <= c['similarity'] < 0.4)
        
        result += "SIMILARITY DISTRIBUTION:\n"
        result += f"  High similarity (>0.5): {high_sim} connections\n"
        result += f"  Medium similarity (0.4-0.5): {medium_sim} connections\n"
        result += f"  Low similarity (0.3-0.4): {low_sim} connections\n"
        result += f"  Average similarity: {sum(c['similarity'] for c in connections)/len(connections):.4f}\n"
        result += f"  Highest similarity found: {connections[0]['similarity']:.4f}\n"
        result += f"  Lowest similarity found: {connections[-1]['similarity']:.4f}\n"
        
        return result
        
    except Exception as e:
        return f"Error computing semantic similarities: {e}"
    
semantic_tool = Tool(
    name="find_semantic_connections",
    func=find_semantic_connections,
    description="Find semantic connections between papers using abstract content similarity analysis."
)


# ADD THIS TO tools.py (after semantic_tool definition, around line 350)

def hybrid_search_wrapper(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    min_year: int = None,
    max_year: int = None
) -> str:
    """
    Search papers using hybrid retrieval (BM25 + semantic embeddings)
    Combines keyword matching with semantic similarity for best results
    
    Args:
        query: Search query
        top_k: Number of results (default 10)
        alpha: BM25 weight (0.5=balanced, 0.7=more keyword, 0.3=more semantic)
        min_year: Optional minimum year filter
        max_year: Optional maximum year filter
    """
    if _memory_manager is None:
        return "Error: Memory system not initialized"
    
    try:
        # Build filters
        filters = {}
        if min_year:
            filters['min_year'] = min_year
        if max_year:
            filters['max_year'] = max_year
        
        # Perform hybrid search
        results = _memory_manager.hybrid_search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            filters=filters if filters else None
        )
        
        if not results:
            return f"No papers found matching '{query}'"
        
        # Format results
        output = f"Found {len(results)} papers using hybrid search (BM25 + semantic):\n\n"
        
        for paper in results:
            output += f"📄 {paper['title']}\n"
            output += f"   Authors: {paper.get('authors', 'N/A')}\n"
            output += f"   Year: {paper.get('year', 'N/A')}\n"
            output += f"   Source: {paper.get('source', 'N/A')}\n"
            
            # Show scoring breakdown
            output += f"   Relevance: {paper['hybrid_score']:.3f} "
            output += f"(Keyword: {paper['bm25_score']:.3f}, "
            output += f"Semantic: {paper['semantic_score']:.3f}) "
            output += f"[{paper['primary_match']} match]\n"
            
            # Abstract preview
            if paper.get('abstract'):
                abstract_preview = paper['abstract'][:200].replace('\n', ' ')
                output += f"   Abstract: {abstract_preview}...\n"
            
            output += "\n"
        
        return output
        
    except Exception as e:
        return f"Error in hybrid search: {str(e)}"


# Create the tool
class HybridSearchArgs(BaseModel):
    query: str = Field(..., description="Search query for finding papers")
    top_k: int = Field(10, description="Number of results to return (default 10)")
    alpha: float = Field(0.5, description="BM25 weight: 0.5=balanced, 0.7=more keyword, 0.3=more semantic")
    min_year: int = Field(None, description="Optional minimum year filter")
    max_year: int = Field(None, description="Optional maximum year filter")

# At the top of tools.py, add a function to get count
def get_paper_count():
    """Get current paper count from database"""
    try:
        import sqlite3
        conn = sqlite3.connect("papers.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except:
        return "your"  # Fallback to generic

# Then create description dynamically
paper_count = get_paper_count()
hybrid_description = f"""Search the LOCAL database of stored papers ({paper_count} papers) using advanced hybrid retrieval (BM25 + semantic similarity).
Use this when the user wants to:
- Find papers they already have/stored
- Search 'my papers', 'stored papers', 'saved papers'
- Look for specific topics in their collection
- Get the most relevant papers from their database
This searches ONLY the papers already in the local database, not external sources."""

hybrid_search_tool = StructuredTool.from_function(
    name="hybrid_paper_search",
    description=hybrid_description,
    func=hybrid_search_wrapper,
    args_schema=HybridSearchArgs,
)

def find_research_bridges(query: str = ""):
    """Find papers that actually bridge different research domains"""
    try:
        db = PaperDatabase()
        papers = db.get_stored_papers()
        
        if len(papers) < 2:
            return "Need at least 2 papers for cross-domain analysis."
        
        # Load embedding model
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            return f"Error loading embedding model: {e}"
        
        # Filter papers with good abstracts
        valid_papers = []
        abstracts = []
        
        for paper in papers:
            if paper.get('abstract') and len(paper['abstract'].strip()) > 50:
                abstracts.append(paper['abstract'])
                valid_papers.append({
                    'title': paper['title'],
                    'year': paper['year'],
                    'id': paper.get('id', 'unknown'),
                    'abstract': paper['abstract']
                })
        
        if len(valid_papers) < 2:
            return f"Need at least 2 papers with substantial abstracts. Found {len(valid_papers)} valid papers from {len(papers)} total."
        
        # Compute embeddings for abstracts
        embeddings = model.encode(abstracts)
        

        # Determine optimal number of clusters (domains) automatically
        n_clusters = min(5, max(2, len(valid_papers) // 3))
        
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        domain_labels = clustering.fit_predict(embeddings)
        
        # Assign domain labels to papers
        for i, paper in enumerate(valid_papers):
            paper['domain_cluster'] = int(domain_labels[i])
        
        # Identify domain themes by finding common keywords in each cluster
        domain_themes = {}
        for domain_id in range(n_clusters):
            domain_papers = [p for p in valid_papers if p['domain_cluster'] == domain_id]
            
            # Extract key terms from abstracts in this domain
            all_text = ' '.join([p['abstract'] for p in domain_papers])
            words = all_text.lower().split()
            
            # Simple keyword frequency for domain naming
            word_freq = {}
            for word in words:
                if len(word) > 4 and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords for this domain
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            domain_name = ', '.join([word for word, _ in top_words])
            
            domain_themes[domain_id] = {
                'name': domain_name,
                'papers': domain_papers,
                'count': len(domain_papers)
            }
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find cross-domain connections
        cross_domain_connections = []
        within_domain_connections = []
        
        for i in range(len(valid_papers)):
            for j in range(i+1, len(valid_papers)):
                similarity = similarity_matrix[i][j]
                
                if similarity > 0.3:
                    paper1 = valid_papers[i]
                    paper2 = valid_papers[j]
                    
                    connection = {
                        'paper1': paper1,
                        'paper2': paper2,
                        'similarity': float(similarity),
                        'is_cross_domain': paper1['domain_cluster'] != paper2['domain_cluster']
                    }
                    
                    if paper1['domain_cluster'] != paper2['domain_cluster']:
                        connection['bridge_type'] = f"{domain_themes[paper1['domain_cluster']]['name']} ↔ {domain_themes[paper2['domain_cluster']]['name']}"
                        cross_domain_connections.append(connection)
                    else:
                        within_domain_connections.append(connection)
        
        # ====================================================================
        # ONLY CHANGE: Shortened output formatting (everything above unchanged)
        # ====================================================================
        
        # Compact result formatting
        result = f"Bridge Analysis ({len(valid_papers)} papers, {n_clusters} domains):\n\n"
        
        # Show domains compactly
        result += "RESEARCH DOMAINS:\n"
        for domain_id, theme_info in domain_themes.items():
            result += f"• Domain {domain_id + 1}: {theme_info['name']} ({theme_info['count']} papers)\n"
        result += "\n"
        
        if cross_domain_connections:
            cross_domain_connections.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Show only top 5 to keep output manageable
            result += f"TOP 5 CROSS-DOMAIN BRIDGES (from {len(cross_domain_connections)} total):\n"
            for i, conn in enumerate(cross_domain_connections[:5], 1):
                # Truncate titles to 35 characters
                title1 = conn['paper1']['title'][:35] + "..."
                title2 = conn['paper2']['title'][:35] + "..."
                
                result += f"{i}. {title1} ({conn['paper1']['year']})\n"
                result += f"   ↔ {title2} ({conn['paper2']['year']})\n"
                result += f"   Similarity: {conn['similarity']:.3f}\n\n"
            
            # Add summary for remaining bridges
            if len(cross_domain_connections) > 5:
                strong_bridges = sum(1 for c in cross_domain_connections if c['similarity'] > 0.5)
                result += f"Additional bridges: {len(cross_domain_connections) - 5} more found "
                result += f"({strong_bridges} with similarity > 0.5)\n"
        else:
            result += "NO CROSS-DOMAIN BRIDGES FOUND:\n"
            result += "Papers are clustered within specific domains.\n\n"
            
            result += "WITHIN-DOMAIN CONNECTIONS:\n"
            domain_summary = {}
            for conn in within_domain_connections:
                domain = conn['paper1']['domain_cluster']
                domain_summary[domain] = domain_summary.get(domain, 0) + 1
            
            for domain_id, count in domain_summary.items():
                if domain_id in domain_themes:
                    theme = domain_themes[domain_id]['name']
                    result += f"• {theme}: {count} connections\n"
        
        return result
        
    except Exception as e:
        return f"Error in bridge analysis: {str(e)}"
    
# Create the bridge-finding tool
bridge_tool = Tool(
    name="find_research_bridges",
    func=find_research_bridges,
    description="Find papers that bridge different research domains using automatic domain clustering and cross-domain similarity analysis."
)

# ============ BIBTEX EXPORT TOOL ============
def export_to_bibtex_wrapper(
    filename: str = "references.bib",
    source_filter: str = None,
    min_year: int = None,
    max_year: int = None,
    title_keywords: str = None
) -> str:
    """
    Export papers from database to BibTeX format
    
    Args:
        filename: Output filename (default: references.bib)
        source_filter: Filter by source ('arxiv' or 'openalex')
        min_year: Minimum publication year
        max_year: Maximum publication year
        title_keywords: Filter by keywords in title (e.g., 'survey', 'review')
    
    Returns:
        Success message with file path
    
    Examples:
        - Export all: export_to_bibtex_wrapper("all.bib")
        - Export surveys: export_to_bibtex_wrapper("surveys.bib", title_keywords="survey")
        - Export recent: export_to_bibtex_wrapper("recent.bib", min_year=2023)
    """
    # If title keywords provided, do custom filtering
    if title_keywords:
        conn = sqlite3.connect("papers.db")
        cursor = conn.cursor()
        
        # Build query with title filter
        query = "SELECT id, title, authors, abstract, year, source, arxiv_id, doi FROM papers WHERE "
        conditions = []
        params = []
        
        # Add title keyword filter (search for ANY of the keywords)
        keywords = title_keywords.lower().split()
        title_conditions = []
        for keyword in keywords:
            title_conditions.append("LOWER(title) LIKE ?")
            params.append(f"%{keyword}%")
        
        conditions.append("(" + " OR ".join(title_conditions) + ")")
        
        # Add other filters
        if source_filter:
            conditions.append("source = ?")
            params.append(source_filter)
        if min_year:
            conditions.append("year >= ?")
            params.append(min_year)
        if max_year:
            conditions.append("year <= ?")
            params.append(max_year)
        
        query += " AND ".join(conditions)
        query += " ORDER BY year DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to papers list
        papers = []
        for row in rows:
            paper = {
                'id': row[0],
                'title': row[1],
                'authors': row[2],
                'abstract': row[3],
                'year': row[4],
                'source': row[5],
                'arxiv_id': row[6],
                'doi': row[7]
            }
            papers.append(paper)
        
        if not papers:
            return f"No papers found with keywords '{title_keywords}' in title."

        # Export the filtered papers to organized directory
        manager = get_output_manager()
        organized_filename = manager.get_bibtex_path(filename)
        return export_papers_to_bibtex(papers, organized_filename, mode="overwrite")
    
    else:
        # Use standard filtering (no title keywords)
        filters = {}
        
        if source_filter:
            filters['source'] = source_filter
        if min_year:
            filters['min_year'] = min_year
        if max_year:
            filters['max_year'] = max_year
        
        result = export_from_database(
            db_path="papers.db",
            filename=filename,
            filters=filters if filters else None,
            mode="overwrite"
        )
        
        return result


# Schema for BibTeX export parameters
class BibTeXExportArgs(BaseModel):
    filename: str = Field(..., description="Output .bib filename (e.g., 'references.bib', 'surveys.bib')")
    title_keywords: str = Field(None, description="Keywords to filter papers by title. Examples: 'survey', 'transformer', 'review'. Use this when user says 'export X papers' where X is the topic.")
    source_filter: str = Field(None, description="Filter by source: 'arxiv' or 'openalex'")
    min_year: int = Field(None, description="Minimum publication year (e.g., 2020)")
    max_year: int = Field(None, description="Maximum publication year (e.g., 2024)")

# Create the BibTeX export tool with proper schema
bibtex_export_tool = StructuredTool.from_function(
    name="export_bibtex",
    description=(
        "Export research papers to BibTeX format for citations. "
        "IMPORTANT: When user says 'export X papers', extract X and pass it to title_keywords. "
        "Examples: 'export survey papers' → title_keywords='survey', "
        "'export transformer papers' → title_keywords='transformer'. "
        "Can also filter by source (arxiv/openalex) and year range. "
        "Creates a .bib file suitable for LaTeX/reference managers."
    ),
    func=export_to_bibtex_wrapper,
    args_schema=BibTeXExportArgs,
    return_direct=False
)