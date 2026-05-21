"""
QA Layer - Quality Assurance for Literature Reviews
"""

import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class QALayer:
    """Quality assurance system for literature reviews"""
    
    def __init__(self):
        """Initialize QA Layer with embedding model for semantic checks"""
        print("Initializing QA Layer...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        print("✓ QA Layer initialized")
    
    # ============ CITATION VERIFICATION ============
    
    def verify_citations(self, review_text: str, papers: List[Dict]) -> Dict:
        """
        Verify all citations are valid and complete
        
        Args:
            review_text: Generated review with [X] citations
            papers: List of papers that were used for generation
            
        Returns:
            {
                'valid': bool,
                'total_citations': int,
                'total_papers': int,
                'coverage': float,
                'missing_citations': list,
                'invalid_citations': list,
                'duplicate_count': int,
                'issues': list
            }
        """
        # Extract all [X] citations from review
        citation_pattern = r'\[(\d+)\]'
        citations_found = re.findall(citation_pattern, review_text)
        citations_found = [int(c) for c in citations_found]
        
        # Expected citations: [1] through [N] where N = number of papers
        num_papers = len(papers)
        expected_citations = set(range(1, num_papers + 1))
        actual_citations = set(citations_found)
        
        # Find issues
        missing = expected_citations - actual_citations
        invalid = actual_citations - expected_citations
        
        # Count duplicates (total citations - unique citations)
        duplicate_count = len(citations_found) - len(actual_citations)
        
        # Calculate coverage
        coverage = len(actual_citations & expected_citations) / num_papers if num_papers > 0 else 0
        
        # Build issues list
        issues = []
        if missing:
            issues.append(f"Missing citations: {sorted(missing)}")
        if invalid:
            issues.append(f"Invalid citations (out of range): {sorted(invalid)}")
        if duplicate_count > 10:  # Some duplication is normal
            issues.append(f"High duplicate count: {duplicate_count} duplicates")
        
        # Determine if valid
        is_valid = len(missing) == 0 and len(invalid) == 0
        
        return {
            'valid': is_valid,
            'total_citations': len(citations_found),
            'unique_citations': len(actual_citations),
            'total_papers': num_papers,
            'coverage': coverage,
            'missing_citations': sorted(missing),
            'invalid_citations': sorted(invalid),
            'duplicate_count': duplicate_count,
            'issues': issues
        }
    
    def extract_claims_with_citations(self, review_text: str) -> List[Dict]:
        """
        Extract individual claims and their citations from review
        
        Args:
            review_text: Full review text
            
        Returns:
            List of {
                'claim': str,
                'citations': list of ints,
                'section': str
            }
        """
        claims = []
        
        # Improved sentence splitting (handles abbreviations better)
        # Split by period followed by space and capital letter, or explicit sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', review_text)
        
        # Track current section for better context
        current_section = "Unknown"
        section_pattern = r'^#+\s+(.+)|^(\d+)\.\s+([A-Z\s]+)$'
        
        for sentence in sentences:
            # Check if this is a section header
            section_match = re.match(section_pattern, sentence.strip())
            if section_match:
                current_section = section_match.group(1) or section_match.group(3) or "Unknown"
                continue
            
            # Find all citations in this sentence
            citations = re.findall(r'\[(\d+)\]', sentence)
            
            if citations and len(sentence.strip()) > 30:  # Minimum claim length
                # Remove citation markers for clean claim text
                clean_claim = re.sub(r'\[\d+\]', '', sentence).strip()
                
                # Remove markdown and special characters
                clean_claim = re.sub(r'[*_#]', '', clean_claim)
                
                # Only include substantive claims (not just headers or lists)
                if len(clean_claim) > 30 and not clean_claim.startswith('-'):
                    claims.append({
                        'claim': clean_claim,
                        'citations': [int(c) for c in citations],
                        'section': current_section
                    })
        
        return claims
    
    # ============ SEMANTIC ACCURACY VERIFICATION ============
    
    def verify_semantic_accuracy(self, review_text: str, papers: List[Dict], detailed: bool = False) -> Dict:
        """
        Verify that claims semantically match the papers they cite
        
        Args:
            review_text: Generated review
            papers: Source papers
            detailed: If True, return detailed claim-by-claim analysis
            
        Returns:
            {
                'average_similarity': float,
                'low_similarity_claims': list,
                'claim_scores': list,
                'passed': bool,
                'detailed_analysis': list (if detailed=True)
            }
        """
        # Extract claims with citations
        claims = self.extract_claims_with_citations(review_text)
        
        if not claims:
            return {
                'average_similarity': 0.0,
                'low_similarity_claims': [],
                'claim_scores': [],
                'passed': False,
                'error': 'No claims extracted'
            }
        
        claim_scores = []
        low_similarity_claims = []
        detailed_analysis = []
        
        for claim_info in claims:
            claim = claim_info['claim']
            citations = claim_info['citations']
            section = claim_info.get('section', 'Unknown')
            
            # Get abstracts of cited papers
            cited_papers_info = []
            for cite_num in citations:
                if 1 <= cite_num <= len(papers):
                    paper = papers[cite_num - 1]  # 0-indexed
                    abstract = paper.get('abstract', '')
                    if abstract:
                        cited_papers_info.append({
                            'num': cite_num,
                            'title': paper.get('title', 'Unknown'),
                            'abstract': abstract
                        })
            
            if not cited_papers_info:
                continue
            
            # Calculate semantic similarity
            claim_embedding = self.embedding_model.encode([claim])
            abstracts = [p['abstract'] for p in cited_papers_info]
            abstract_embeddings = self.embedding_model.encode(abstracts)
            
            # Get max similarity (best match among cited papers)
            similarities = cosine_similarity(claim_embedding, abstract_embeddings)[0]
            max_similarity = float(np.max(similarities))
            best_match_idx = int(np.argmax(similarities))
            
            claim_score = {
                'claim': claim[:100] + '...' if len(claim) > 100 else claim,
                'citations': citations,
                'similarity': max_similarity,
                'section': section
            }
            claim_scores.append(claim_score)
            
            # Flag low similarity (potential misattribution)
            if max_similarity < 0.3:
                low_similarity_claims.append({
                    'claim': claim[:150] + '...' if len(claim) > 150 else claim,
                    'citations': citations,
                    'similarity': max_similarity,
                    'section': section
                })
            
            # Detailed analysis
            if detailed:
                best_paper = cited_papers_info[best_match_idx]
                detailed_analysis.append({
                    'claim': claim,
                    'section': section,
                    'citations': citations,
                    'best_match': best_paper['num'],
                    'best_match_title': best_paper['title'],
                    'similarity': max_similarity,
                    'status': '✅ Good' if max_similarity >= 0.5 else ('⚠️ Moderate' if max_similarity >= 0.3 else '❌ Poor')
                })
        
        # Calculate average
        avg_similarity = np.mean([s['similarity'] for s in claim_scores]) if claim_scores else 0.0
        
        # Pass if average > 0.5 and less than 20% low-similarity claims
        passed = avg_similarity > 0.5 and len(low_similarity_claims) / len(claim_scores) < 0.2 if claim_scores else False
        
        result = {
            'average_similarity': float(avg_similarity),
            'low_similarity_claims': low_similarity_claims,
            'claim_scores': claim_scores,
            'total_claims': len(claim_scores),
            'passed': passed
        }
        
        if detailed:
            result['detailed_analysis'] = detailed_analysis
        
        return result
    
    # ============ QUALITY SCORING ============
    
    def score_quality(self, review_text: str, papers: List[Dict], citation_results: Dict) -> Dict:
        """
        Calculate overall quality metrics
        
        Args:
            review_text: Generated review
            papers: Source papers
            citation_results: Results from verify_citations()
            
        Returns:
            {
                'citation_density': float,
                'coverage': float,
                'length_score': float,
                'overall_score': float,
                'grade': str
            }
        """
        # Metric 1: Citation density (citations per 1000 characters)
        text_length = len(review_text)
        citation_count = citation_results['total_citations']
        citation_density = (citation_count / text_length * 1000) if text_length > 0 else 0
        
        # Normalize to 0-1 (assume 2-5 citations per 1000 chars is good)
        citation_density_score = min(citation_density / 3.5, 1.0)
        
        # Metric 2: Coverage (already calculated)
        coverage_score = citation_results['coverage']
        
        # Metric 3: Length score (5000-8000 chars is ideal)
        if 5000 <= text_length <= 8000:
            length_score = 1.0
        elif text_length < 5000:
            length_score = text_length / 5000
        else:  # > 8000
            length_score = max(0.7, 1.0 - (text_length - 8000) / 10000)
        
        # Overall score (weighted average)
        overall_score = (
            coverage_score * 0.5 +           # 50% weight on coverage
            citation_density_score * 0.3 +   # 30% weight on citation density
            length_score * 0.2               # 20% weight on length
        )
        
        # Assign grade
        if overall_score >= 0.9:
            grade = 'A (Excellent)'
        elif overall_score >= 0.8:
            grade = 'B (Good)'
        elif overall_score >= 0.7:
            grade = 'C (Acceptable)'
        elif overall_score >= 0.6:
            grade = 'D (Needs Improvement)'
        else:
            grade = 'F (Poor)'
        
        return {
            'citation_density': citation_density,
            'citation_density_score': citation_density_score,
            'coverage': coverage_score,
            'length': text_length,
            'length_score': length_score,
            'overall_score': overall_score,
            'grade': grade
        }
    
    # ============ COMPREHENSIVE VALIDATION ============
    
    def validate_review(self, review_text: str, papers: List[Dict], query: str = "") -> Dict:
        """
        Run all QA checks and return comprehensive report
        
        Args:
            review_text: Generated literature review
            papers: Source papers used
            query: Original search query
            
        Returns:
            Complete QA report with all metrics
        """
        print("\n" + "="*70)
        print("QA VALIDATION REPORT")
        print("="*70)
        
        if query:
            print(f"Query: {query}")
        print(f"Papers analyzed: {len(papers)}")
        print(f"Review length: {len(review_text):,} characters\n")
        
        # Run all checks
        citation_results = self.verify_citations(review_text, papers)
        semantic_results = self.verify_semantic_accuracy(review_text, papers)
        quality_results = self.score_quality(review_text, papers, citation_results)
        
        # Print results
        print("CITATION VERIFICATION:")
        if citation_results['valid']:
            print("  ✅ PASS - All citations valid")
        else:
            print("  ❌ FAIL - Citation issues detected")
        print(f"  • Total citations: {citation_results['total_citations']}")
        print(f"  • Coverage: {citation_results['coverage']*100:.1f}%")
        if citation_results['issues']:
            for issue in citation_results['issues']:
                print(f"  ⚠️  {issue}")
        
        print("\nSEMANTIC ACCURACY:")
        if semantic_results['passed']:
            print("  ✅ PASS - Claims match cited papers")
        else:
            print("  ⚠️  WARNING - Some claims may not match citations")
        print(f"  • Average similarity: {semantic_results['average_similarity']:.3f}")
        print(f"  • Total claims analyzed: {semantic_results.get('total_claims', 0)}")
        if semantic_results['low_similarity_claims']:
            print(f"  • Low similarity claims: {len(semantic_results['low_similarity_claims'])}")
        
        print("\nQUALITY SCORE:")
        print(f"  • Overall score: {quality_results['overall_score']:.3f}/1.0")
        print(f"  • Grade: {quality_results['grade']}")
        print(f"  • Citation density: {quality_results['citation_density']:.2f} per 1000 chars")
        print(f"  • Coverage: {quality_results['coverage']*100:.0f}%")
        print(f"  • Length: {quality_results['length']:,} characters")
        
        # Overall verdict
        print("\n" + "="*70)
        if citation_results['valid'] and semantic_results['passed'] and quality_results['overall_score'] >= 0.7:
            print("OVERALL VERDICT: ✅ APPROVED FOR USE")
        elif quality_results['overall_score'] >= 0.6:
            print("OVERALL VERDICT: ⚠️  ACCEPTABLE WITH MINOR ISSUES")
        else:
            print("OVERALL VERDICT: ❌ NEEDS IMPROVEMENT OR REGENERATION")
        print("="*70 + "\n")
        
        # Return comprehensive results
        return {
            'query': query,
            'citation_verification': citation_results,
            'semantic_accuracy': semantic_results,
            'quality_metrics': quality_results,
            'overall_verdict': 'PASS' if (citation_results['valid'] and quality_results['overall_score'] >= 0.7) else 'FAIL',
            'recommendation': self._get_recommendation(citation_results, semantic_results, quality_results)
        }
    
    def _get_recommendation(self, citation_results, semantic_results, quality_results):
        """Generate recommendation based on results"""
        if citation_results['valid'] and semantic_results['passed'] and quality_results['overall_score'] >= 0.8:
            return "Excellent quality - ready for use"
        elif quality_results['overall_score'] >= 0.7:
            return "Good quality - minor improvements possible"
        elif quality_results['overall_score'] >= 0.6:
            return "Acceptable quality - consider regeneration for critical use"
        else:
            return "Poor quality - regenerate with different parameters"
    
    def generate_detailed_report(self, review_text: str, papers: List[Dict], query: str = "") -> str:
        """
        Generate a detailed text report with claim-by-claim analysis
        
        Args:
            review_text: Generated review
            papers: Source papers
            query: Original query
            
        Returns:
            Formatted text report
        """
        # Run validation with detailed analysis
        citation_results = self.verify_citations(review_text, papers)
        semantic_results = self.verify_semantic_accuracy(review_text, papers, detailed=True)
        quality_results = self.score_quality(review_text, papers, citation_results)
        
        report = []
        report.append("=" * 70)
        report.append("DETAILED QA REPORT")
        report.append("=" * 70)
        report.append(f"Query: {query}")
        report.append(f"Generated: {len(review_text):,} characters")
        report.append(f"Papers: {len(papers)}")
        report.append("")
        
        # Citation verification
        report.append("CITATION VERIFICATION:")
        report.append(f"  Status: {'✅ PASS' if citation_results['valid'] else '❌ FAIL'}")
        report.append(f"  Total citations: {citation_results['total_citations']}")
        report.append(f"  Unique citations: {citation_results['unique_citations']}")
        report.append(f"  Coverage: {citation_results['coverage']*100:.1f}%")
        if citation_results['missing_citations']:
            report.append(f"  Missing: {citation_results['missing_citations']}")
        report.append("")
        
        # Semantic accuracy with details
        report.append("SEMANTIC ACCURACY:")
        report.append(f"  Status: {'✅ PASS' if semantic_results['passed'] else '⚠️  WARNING'}")
        report.append(f"  Average similarity: {semantic_results['average_similarity']:.3f}")
        report.append(f"  Claims analyzed: {semantic_results['total_claims']}")
        report.append(f"  Low similarity: {len(semantic_results['low_similarity_claims'])}")
        report.append("")
        
        # Show problematic claims
        if semantic_results.get('low_similarity_claims'):
            report.append("  PROBLEMATIC CLAIMS:")
            for i, claim in enumerate(semantic_results['low_similarity_claims'][:5], 1):
                report.append(f"    {i}. Similarity: {claim['similarity']:.3f}")
                report.append(f"       Citations: {claim['citations']}")
                report.append(f"       Claim: {claim['claim'][:100]}...")
                report.append("")
        
        # Quality metrics
        report.append("QUALITY METRICS:")
        report.append(f"  Overall score: {quality_results['overall_score']:.3f}/1.0")
        report.append(f"  Grade: {quality_results['grade']}")
        report.append(f"  Citation density: {quality_results['citation_density']:.2f}/1000 chars")
        report.append(f"  Length score: {quality_results['length_score']:.3f}")
        report.append("")
        
        # Recommendation
        recommendation = self._get_recommendation(citation_results, semantic_results, quality_results)
        report.append(f"RECOMMENDATION: {recommendation}")
        report.append("=" * 70)
        
        return "\n".join(report)


# ============ STANDALONE TESTING ============

if __name__ == "__main__":
    print("QA Layer Test")
    print("="*70)
    
    # Test with sample review
    sample_review = """
    # LITERATURE REVIEW: Test Topic
    
    This is a test review with citations [1] and [2]. The first paper discusses
    transformers [1], while the second explores BERT models [2]. Both papers
    contribute to understanding [1][2].
    """
    
    sample_papers = [
        {'title': 'Paper 1', 'abstract': 'This paper discusses transformer architectures.'},
        {'title': 'Paper 2', 'abstract': 'This paper explores BERT models and applications.'}
    ]
    
    qa = QALayer()
    results = qa.validate_review(sample_review, sample_papers, query="Test Topic")
    
    print("\nTest complete!")