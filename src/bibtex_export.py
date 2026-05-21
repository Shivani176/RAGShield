"""
BibTeX Export Module
Exports research papers to BibTeX format for citations
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

from output_manager import get_output_manager

# ============ HELPER FUNCTIONS ============

def sanitize_bibtex_field(text: str) -> str:
    """
    Clean text for BibTeX format
    - Remove curly braces that would break BibTeX
    - Escape special LaTeX characters
    - Handle None values
    """
    if text is None:
        return ""
    
    text = str(text)
    # Replace problematic characters
    text = text.replace('{', '').replace('}', '')
    text = text.replace('\\', '\\\\')
    text = text.replace('$', '\\$')
    text = text.replace('%', '\\%')
    text = text.replace('&', '\\&')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    
    return text.strip()


def format_authors_bibtex(authors: List[str]) -> str:
    """
    Format author list for BibTeX
    Converts list like ["John Doe", "Jane Smith"] to "John Doe and Jane Smith"
    """
    if not authors:
        return "Unknown"
    
    # Handle if authors is a JSON string
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except:
            return sanitize_bibtex_field(authors)
    
    # Join with ' and ' for BibTeX format
    return " and ".join([sanitize_bibtex_field(author) for author in authors])


def generate_bibtex_key(paper: Dict, index: int) -> str:
    """
    Generate a unique BibTeX citation key
    Format: FirstAuthorLastName2023keyword
    
    Example: Vaswani2017attention
    """
    # Get first author's last name
    authors = paper.get('authors', [])
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except:
            authors = [authors]
    
    if authors:
        first_author = authors[0].split()[-1]  # Get last name
    else:
        first_author = "Unknown"
    
    # Get year
    year = paper.get('year', 'YYYY')
    
    # Get first significant word from title
    title = paper.get('title', '')
    title_words = [w.lower() for w in title.split() if len(w) > 3 and w.isalnum()]
    keyword = title_words[0] if title_words else 'paper'
    
    # Clean up the key
    key = f"{first_author}{year}{keyword}"
    key = ''.join(c for c in key if c.isalnum())
    
    return key


# ============ CORE CONVERSION FUNCTIONS ============

def paper_to_bibtex(paper: Dict, index: int) -> str:
    """
    Convert a single paper dict to BibTeX entry
    
    Args:
        paper: Dictionary with keys: id, title, authors, abstract, year, source, arxiv_id, doi
        index: Paper index for fallback citation key
    
    Returns:
        BibTeX formatted string
    """
    # Generate citation key
    cite_key = generate_bibtex_key(paper, index)
    
    # Determine entry type (most papers are articles or preprints)
    source = paper.get('source', 'arxiv').lower()
    if source == 'arxiv':
        entry_type = 'article'
        journal = 'arXiv preprint'
    elif source == 'openalex':
        entry_type = 'article'
        journal = paper.get('journal', 'OpenAlex')
    else:
        entry_type = 'article'
        journal = source
    
    # Start building the BibTeX entry
    bibtex = f"@{entry_type}{{{cite_key},\n"
    
    # Required fields
    title = sanitize_bibtex_field(paper.get('title', 'Untitled'))
    bibtex += f"  title = {{{title}}},\n"
    
    authors = format_authors_bibtex(paper.get('authors', []))
    bibtex += f"  author = {{{authors}}},\n"
    
    year = paper.get('year', 'n.d.')
    bibtex += f"  year = {{{year}}},\n"
    
    # Optional fields
    if paper.get('abstract'):
        abstract = sanitize_bibtex_field(paper['abstract'][:500])  # Limit length
        bibtex += f"  abstract = {{{abstract}...}},\n"
    
    bibtex += f"  journal = {{{journal}}},\n"
    
    # Add DOI if available
    if paper.get('doi'):
        doi = sanitize_bibtex_field(paper['doi'])
        bibtex += f"  doi = {{{doi}}},\n"
    
    # Add arXiv ID if available
    if paper.get('arxiv_id'):
        arxiv_id = sanitize_bibtex_field(paper['arxiv_id'])
        bibtex += f"  eprint = {{{arxiv_id}}},\n"
        bibtex += f"  archivePrefix = {{arXiv}},\n"
    
    # Add URL (prefer DOI, then arXiv)
    if paper.get('doi'):
        url = f"https://doi.org/{paper['doi']}"
    elif paper.get('arxiv_id'):
        url = f"https://arxiv.org/abs/{paper['arxiv_id']}"
    elif paper.get('pdf_url'):
        url = paper['pdf_url']
    else:
        url = None
    
    if url:
        bibtex += f"  url = {{{url}}},\n"
    
    # Close the entry
    bibtex += "}\n"
    
    return bibtex


# ============ EXPORT FUNCTIONS ============

def export_papers_to_bibtex(
    papers: List[Dict],
    filename: str = "references.bib",
    mode: str = "overwrite"
) -> str:
    """
    Export a list of papers to BibTeX format
    
    Args:
        papers: List of paper dictionaries
        filename: Output filename (default: references.bib)
        mode: 'overwrite', 'append', or 'versioned'
    
    Returns:
        Success message with file path
    """
    if not papers:
        return "No papers to export."
    
    # Generate BibTeX entries
    bibtex_entries = []
    header = f"% BibTeX Bibliography\n% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"% Total entries: {len(papers)}\n\n"
    
    bibtex_entries.append(header)
    
    for i, paper in enumerate(papers, 1):
        bibtex = paper_to_bibtex(paper, i)
        bibtex_entries.append(bibtex)
    
    # Combine all entries
    full_bibtex = "\n".join(bibtex_entries)
    
    # Handle different save modes
    try:
        if mode == "append":
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"\n% Appended on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(full_bibtex)
            return f"✓ Appended {len(papers)} entries to {filename}"
        
        elif mode == "versioned":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1] if '.' in filename else 'bib'
            versioned_filename = f"{base}_{timestamp}.{ext}"
            
            with open(versioned_filename, 'w', encoding='utf-8') as f:
                f.write(full_bibtex)
            return f"✓ Created versioned file: {versioned_filename} ({len(papers)} entries)"
        
        else:  # overwrite (default)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_bibtex)
            return f"✓ Exported {len(papers)} papers to {filename}"
    
    except Exception as e:
        return f"✗ Error exporting to BibTeX: {str(e)}"


def export_from_database(
    db_path: str = "papers.db",
    filename: str = "references.bib",
    filters: Optional[Dict] = None,
    mode: str = "overwrite"
) -> str:
    """
    Export papers directly from SQLite database to BibTeX
    
    Args:
        db_path: Path to papers.db
        filename: Output BibTeX filename
        filters: Optional filters (e.g., {'source': 'arxiv', 'min_year': 2020})
        mode: 'overwrite', 'append', or 'versioned'
    
    Returns:
        Success message
    """
    manager = get_output_manager()
    filename = manager.get_bibtex_path(filename)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query based on filters
        query = "SELECT id, title, authors, abstract, year, source, arxiv_id, doi FROM papers"
        conditions = []
        params = []
        
        if filters:
            if 'source' in filters:
                conditions.append("source = ?")
                params.append(filters['source'])
            
            if 'min_year' in filters:
                conditions.append("year >= ?")
                params.append(filters['min_year'])
            
            if 'max_year' in filters:
                conditions.append("year <= ?")
                params.append(filters['max_year'])
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY year DESC, title"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert rows to paper dictionaries
        papers = []
        for row in rows:
            paper = {
                'id': row[0],
                'title': row[1],
                'authors': row[2],  # Will be JSON string
                'abstract': row[3],
                'year': row[4],
                'source': row[5],
                'arxiv_id': row[6],
                'doi': row[7]
            }
            papers.append(paper)
        
        if not papers:
            return "No papers found matching the filters."
        
        # Export to BibTeX
        return export_papers_to_bibtex(papers, filename, mode)
    
    except Exception as e:
        return f"✗ Error reading from database: {str(e)}"


# ============ COMMAND LINE INTERFACE ============
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("BibTeX Export Utility")
    print("=" * 70)
    
    # Quick test: Export all papers from database
    print("\n🧪 Testing: Exporting all papers from papers.db...\n")
    
    result = export_from_database(
        db_path="papers.db",
        filename="test_export.bib",
        mode="overwrite"
    )
    print(result)
    
    # Show first few lines of the file
    try:
        with open("test_export.bib", 'r', encoding='utf-8') as f:
            lines = f.readlines()[:25]
            print("\n" + "=" * 70)
            print("First 25 lines of exported file:")
            print("=" * 70)
            print("".join(lines))
            
            if len(lines) >= 25:
                print("\n... (file continues)")
    except Exception as e:
        print(f"\nCouldn't read file: {e}")
    
    print("\n" + "=" * 70)
    print("✓ BibTeX export module ready!")
    print("=" * 70)
    print("\nUsage in Python:")
    print("  from bibtex_export import export_papers_to_bibtex")
    print("  papers = [...]  # Your paper list")
    print("  export_papers_to_bibtex(papers, 'my_refs.bib')")
    print("\nOr export directly from database:")
    print("  from bibtex_export import export_from_database")
    print("  export_from_database(filename='refs.bib', filters={'min_year': 2020})")
    print("=" * 70)