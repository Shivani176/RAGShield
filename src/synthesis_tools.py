"""
Synthesis Tools - Integration wrapper for synthesis engine
"""

from synthesis_engine import SynthesisEngine
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field



# Import QA Layer for automatic validation
try:
    from qa_layer import QALayer
    _qa_layer = None
except ImportError:
    print("⚠️  QA Layer not available - reviews will not be validated")
    QALayer = None
    _qa_layer = None

def get_qa_layer():
    """Get or create global QA layer"""
    global _qa_layer
    if _qa_layer is None and QALayer is not None:
        _qa_layer = QALayer()
    return _qa_layer


# Global synthesis engine
_synthesis_engine = None
_memory_manager = None

def set_memory_manager(manager):
    """inject memory manager from main.py"""
    global _memory_manager
    _memory_manager = manager
    print("Synthesis memory manager connected")

def get_synthesis_engine():
    """get or create global synthesis engine"""
    global _synthesis_engine
    if _synthesis_engine is None:
        _synthesis_engine = SynthesisEngine()
    return _synthesis_engine    


def synthesize_literature_wrapper(
    query: str,
    max_papers: int = 10,
    include_gaps: bool = True
) -> str:
    """
    Generate a literature review from stored papers
    
    Args:
        query: Research question (e.g., "What are attention mechanisms in deep learning?")
        max_papers: Maximum papers to analyze (default 10)
        include_gaps: Whether to identify research gaps (default True)
    
    Returns:
        Formatted literature review with citations
    """
    if _memory_manager is None:
        return "Error: Memory system not initialized"
    
    try:
        # Step 1: Search for relevant papers using hybrid search
        print(f"\n🔍 Searching for papers about: {query}")
        papers = _memory_manager.hybrid_search(
            query=query,
            top_k=max_papers,
            alpha=0.5  # Balanced keyword + semantic
        )
        
        # CHECK 1: No papers found
        if not papers or len(papers) == 0:
            try:
                import sqlite3
                conn = sqlite3.connect("papers.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                total_papers = cursor.fetchone()[0]
                conn.close()
            except:
                total_papers = "your"
            
            return f"""📚 No papers found for query: '{query}'.

Your database has {total_papers} papers, but none match this topic.

**Suggestions:**
1. Try different keywords (e.g., broader terms)
2. Check what topics are available: "Check stored papers"
3. Search for new papers: "Find papers about {query}"

Example queries that work:
- "Generate literature review on transformers"
- "Generate literature review on quantum computing"
- "Generate literature review on deep learning"
"""
        
        print(f"📚 Found {len(papers)} relevant papers")
        
        # CHECK 2: Relevance threshold
        best_score = papers[0].get('hybrid_score', 0) if papers else 0
        RELEVANCE_THRESHOLD = 0.30
        
        if best_score < RELEVANCE_THRESHOLD:
            print(f"⚠️  Warning: Low relevance scores (best: {best_score:.3f})")
            return f"""⚠️  No relevant papers found for '{query}'

The search found {len(papers)} papers, but they don't seem directly related to your topic.

**What happened:**
- Best match score: {best_score:.3f} (threshold: {RELEVANCE_THRESHOLD})
- Papers found are only loosely related to '{query}'

**Suggestions:**
1. Try more specific or different keywords
2. Check if papers on this topic exist: "Check stored papers"
3. Search external sources: "Find papers about {query}"

Top paper found (low relevance):
- {papers[0]['title'][:80]}...
  (Score: {best_score:.3f})
"""
        
        # CHECK 3: Few papers warning
        if len(papers) < 3:
            print(f"⚠️  Warning: Only {len(papers)} papers found. Review will be limited.")
        
        if 0.30 <= best_score < 0.5:
            print(f"⚠️  Note: Medium relevance (score: {best_score:.3f})")
            print(f"📊 Papers may not be directly about '{query}'")
        
        # Step 2: Generate literature review
        engine = get_synthesis_engine()
        result = engine.generate_literature_review(
            papers=papers,
            query=query,
            max_papers=max_papers,
            include_gaps=include_gaps
        )
        
        if 'error' in result:
            return f"❌ Synthesis Error: {result['error']}"
        
        # Step 3: Format output
        output = f"# LITERATURE REVIEW: {query}\n\n"
        
        # Add warning if medium relevance
        if best_score < 0.5:
            output += f"⚠️  *Note: Medium relevance match (score: {best_score:.3f})*\n\n"
        
        output += result['review_text']
        
        # Step 4: AUTO-VALIDATE WITH QA LAYER
        try:
            qa = get_qa_layer()
            if qa is not None:
                # Run validation
                validation = qa.validate_review(
                    result['review_text'],
                    papers,
                    query
                )
                
                # Add QA section
                output += f"\n\n{'─'*70}\n"
                output += "📊 **QUALITY ASSURANCE**\n\n"
                
                # Show verdict
                if validation['overall_verdict'] == 'PASS':
                    output += "✅ **QUALITY VERIFIED**\n"
                else:
                    output += "⚠️  **QUALITY WARNING**\n"
                
                # Show key metrics
                output += f"- Overall Score: **{validation['quality_metrics']['overall_score']:.3f}/1.0**\n"
                output += f"- Grade: **{validation['quality_metrics']['grade']}**\n"
                output += f"- Citation Coverage: {validation['citation_verification']['coverage']*100:.0f}%\n"
                output += f"- Semantic Accuracy: {validation['semantic_accuracy']['average_similarity']:.3f}\n"
                
                # Show issues if any
                if validation['citation_verification'].get('issues'):
                    output += f"\n⚠️  Issues detected:\n"
                    for issue in validation['citation_verification']['issues']:
                        output += f"  • {issue}\n"
                
                # Show recommendation
                if validation['overall_verdict'] != 'PASS':
                    output += f"\n💡 Recommendation: {validation['recommendation']}\n"
                
                output += f"{'─'*70}\n"
        except Exception as e:
            print(f"Note: QA validation skipped: {e}")
        
        # Original statistics
        output += f"\n\n📊 **Analysis Statistics**\n\n"
        output += f"- Papers analyzed: **{result['paper_count']}**\n"
        output += f"- Papers cited: **{len(result['citations_used'])}** ({len(result['citations_used'])/result['paper_count']*100:.0f}% coverage)\n"
        output += f"- Review length: {len(result['review_text']):,} characters\n"
        output += f"- Relevance score: {best_score:.3f}\n"
        
        return output
        
    except Exception as e:
        return f"Error in synthesis: {str(e)}"

def quick_summary_wrapper(query: str, max_papers: int = 10) -> str:
    """
    Generate a quick summary (not full review) from papers
    
    Args:
        query: Research question
        max_papers: Maximum papers to use (default 10)
    
    Returns:
        Brief summary with citations
    """
    if _memory_manager is None:
        return "Error: Memory system not initialized"
    
    try:
        # Search for papers
        papers = _memory_manager.hybrid_search(
            query=query,
            top_k=max_papers,
            alpha=0.5
        )
        
        if not papers:
            return f"No papers found for: '{query}'"
        
        # Generate quick summary
        engine = get_synthesis_engine()
        summary = engine.quick_summary(papers, query, max_length=500)
        
        return f"Quick Summary ({len(papers)} papers):\n\n{summary}"
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"


# ============ TOOL SCHEMAS ============

class SynthesisArgs(BaseModel):
    query: str = Field(..., description="Research question or topic (e.g., 'What are transformer architectures?')")
    max_papers: int = Field(20, description="Maximum papers to analyze (default 20)")
    include_gaps: bool = Field(True, description="Whether to identify research gaps (default True)")


class QuickSummaryArgs(BaseModel):
    query: str = Field(..., description="Research question")
    max_papers: int = Field(10, description="Maximum papers (default 10)")


# ============ CREATE TOOLS ============

synthesis_tool = StructuredTool.from_function(
    name="generate_literature_review",
    description=(
        "Generate a structured literature review from stored papers with citation enforcement. "
        "Use this when user asks for: 'literature review', 'synthesize papers', 'what does research say about X', "
        "'summarize findings on X', 'write a review about X'. "
        "This analyzes multiple papers and generates: Introduction, Key Findings, Methodology, Research Gaps, Conclusion. "
        "Every claim is cited with [X] format. No hallucinations - only information from actual paper abstracts."
    ),
    func=synthesize_literature_wrapper,
    args_schema=SynthesisArgs,
    return_direct=True
)


quick_summary_tool = StructuredTool.from_function(
    name="quick_summary",
    description=(
        "Generate a brief summary (not full review) from papers. "
        "Use for quick answers like 'briefly tell me about X', 'quick summary of X', 'what is X in research'."
    ),
    func=quick_summary_wrapper,
    args_schema=QuickSummaryArgs,
    return_direct=False
)