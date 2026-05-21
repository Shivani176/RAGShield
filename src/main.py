# ---- turn off tracing/loggers BEFORE importing langchain ----
import os
import langchain
langchain.debug = False

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)
os.environ["LANGCHAIN_DEBUG"] = "false"

from dotenv import load_dotenv
load_dotenv()

import time
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from memory_manager import MemoryManager

from tools import search_tool, wiki_tool, save_tool, arxiv_tool, openalex_tool, stored_papers_tool, connections_tool, semantic_tool, bridge_tool, bibtex_export_tool, hybrid_search_tool
from tools import set_memory_manager

from synthesis_tools import synthesis_tool, quick_summary_tool
from synthesis_tools import set_memory_manager as set_synthesis_memory


# ---------------- LLM Configuration ----------------
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.1,
    max_tokens=4000,
    timeout=60,
    max_retries=2,
    streaming=False
)

# ---------------- Memory Manager Configuration ----------------
print("Initializing Memory Manager...")
memory_manager = MemoryManager()
print("✓ Memory loaded!\n")


# Connect memory manager to tools
set_memory_manager(memory_manager)

# ============ NEW: Connect memory to synthesis tools ============
set_synthesis_memory(memory_manager)


# ---------------- Reference Detection ----------------
def detect_past_reference(query: str) -> dict:
    """
    Detect if user is referencing past conversations or papers
    
    Returns:
        {
            'has_reference': bool,
            'reference_type': 'conversation' | 'paper' | 'analysis' | None,
            'keywords': list of key terms to search for
        }
    """
    query_lower = query.lower()
    
    # Temporal reference words
    temporal_words = ['yesterday', 'last week', 'earlier', 'before', 'previous', 
                     'ago', 'past', 'recent', 'monday', 'tuesday', 'last month']
    
    # Demonstrative reference words
    demonstrative_words = ['those', 'these', 'that', 'the', 'above']
    
    # Check for temporal references
    has_temporal = any(word in query_lower for word in temporal_words)
    
    # Check for demonstrative references
    has_demonstrative = any(word in query_lower for word in demonstrative_words)
    
    # Determine if there's a reference
    has_reference = has_temporal or has_demonstrative
    
    if not has_reference:
        return {'has_reference': False, 'reference_type': None, 'keywords': []}
    
    # Determine what type of thing they're referencing
    if any(word in query_lower for word in ['paper', 'papers', 'article', 'research', 'study']):
        return {
            'has_reference': True,
            'reference_type': 'paper',
            'keywords': [word for word in query_lower.split() if len(word) > 3][:5]
        }
    
    if any(word in query_lower for word in ['analysis', 'connection', 'bridge', 'semantic']):
        return {
            'has_reference': True,
            'reference_type': 'analysis',
            'keywords': [word for word in query_lower.split() if len(word) > 3][:5]
        }
    
    # Default to conversation reference
    return {
        'has_reference': True,
        'reference_type': 'conversation',
        'keywords': [word for word in query_lower.split() if len(word) > 3][:5]
    }



# ---------------- Smart Chat History Manager ----------------
class SmartChatHistory:
    def __init__(self, max_exchanges=5, summarize_interval=10, memory_manager=None):
        self.max_exchanges = max_exchanges
        self.summarize_interval = summarize_interval
        self.memory_manager = memory_manager
        self.history = []
        self.last_paper_results = None
        self.last_analysis_results = None
        self.exchanges_since_summary = 0
        self.current_session_start = time.time()
        
    def add_exchange(self, query: str, response: str, query_type: str):
        """Add exchange and check if summarization needed"""
        exchange = {
            'query': query,
            'response': response,
            'type': query_type,
            'timestamp': time.time()
        }
        
        # Cache important results (existing logic)
        if query_type in ["paper_search", "database"]:
            self.last_paper_results = {
                'query': query,
                'response': response,
                'timestamp': time.time()
            }
        elif query_type == "analysis":
            self.last_analysis_results = {
                'query': query, 
                'response': response,
                'timestamp': time.time()
            }
            
        self.history.append(exchange)
        self.exchanges_since_summary += 1
        
        # Keep only recent exchanges
        if len(self.history) > self.max_exchanges:
            self.history = self.history[-self.max_exchanges:]
        
        # Check if we should summarize
        if self.exchanges_since_summary >= self.summarize_interval:
            self.create_episodic_summary()
    
    def create_episodic_summary(self):
        """Create and store session summary"""
        if not self.memory_manager or len(self.history) == 0:
            return
        
        # Build context for LLM
        exchanges_text = "\n\n".join([
            f"Query {i+1} ({ex['type']}): {ex['query']}\nResponse: {ex['response'][:200]}..."
            for i, ex in enumerate(self.history[-self.summarize_interval:])
        ])
        
        # Create summary prompt
        summary_prompt = f"""Summarize this research session concisely. Focus on:
1. Main research topics explored
2. Papers found and their key themes
3. Analyses performed
4. Key findings or insights
5. Actions taken (saves, connections analyzed, etc.)
6. Next steps or open questions

Be specific and actionable. Use researcher-friendly language.

Session exchanges:
{exchanges_text}

Provide a concise 3-5 sentence summary suitable for future reference."""
        
        try:
            # Generate summary using LLM
            llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.3,
                max_tokens=500
            )
            
            summary_response = llm.invoke(summary_prompt)
            summary = summary_response.content
            
            # Store in ChromaDB through MemoryManager
            session_id = f"session_{int(self.current_session_start)}"
            self.memory_manager.store_conversation(
                user_query=f"Session summary ({self.exchanges_since_summary} exchanges)",
                agent_response=summary,
                query_type="episodic_summary",
                session_id=session_id
            )
            
            print(f"\n💾 Episodic summary created and stored (Session: {session_id})")
            
            # Reset counter
            self.exchanges_since_summary = 0
            
        except Exception as e:
            print(f"Note: Could not create episodic summary: {e}")
    
    # ============ NEW: TOPIC EXTRACTION AND MATCHING ============
    
    def _extract_topic_from_query(self, query: str) -> str:
        """
        Extract the actual topic the user is asking about
        
        Example: "save those positional encoding papers" → "positional encoding"
        """
        query_lower = query.lower()
        
        # Words to remove (command words, articles, etc.)
        stop_words = {
            'save', 'those', 'these', 'the', 'a', 'an', 'to', 'into', 
            'file', 'txt', 'papers', 'paper', 'can', 'you', 'please',
            'above', 'that', 'my', 'from', 'about', 'find', 'search',
            'get', 'show', 'me', 'results', 'and', 'or'
        }
        
        # Split and filter
        words = query_lower.split()
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Remove file extensions
        meaningful_words = [w.replace('.txt', '').replace('.md', '') for w in meaningful_words]
        
        # Rejoin to get topic
        topic = ' '.join(meaningful_words)
        
        return topic.strip()
    
    def _does_topic_match(self, user_topic: str, last_search_query: str, threshold: float = 0.3) -> bool:
        """
        Check if the user's specified topic matches the last search query
        
        Args:
            user_topic: Topic extracted from user's current query
            last_search_query: The query from the last paper search
            threshold: Minimum word overlap ratio (default 30%)
        
        Returns:
            True if topics match, False otherwise
        """
        if not user_topic:
            # No specific topic mentioned, assume they mean last search
            return True
        
        user_words = set(user_topic.lower().split())
        last_words = set(last_search_query.lower().split())
        
        # Remove common stop words from both
        stop_words = {'find', 'papers', 'about', 'the', 'a', 'an', 'and', 'or', 'for', 'in', 'on'}
        user_words -= stop_words
        last_words -= stop_words
        
        if not user_words:
            return True  # No meaningful words to compare
        
        # Calculate overlap
        common_words = user_words.intersection(last_words)
        overlap_ratio = len(common_words) / len(user_words)
        
        return overlap_ratio >= threshold
    
    def _search_memory_for_topic(self, topic: str, memory_manager) -> dict:
        """
        Search long-term memory for papers about a specific topic
        
        Args:
            topic: The topic to search for
            memory_manager: MemoryManager instance
            
        Returns:
            dict with context information
        """
        if not memory_manager or not topic:
            return {'has_context': False}
        
        try:
            # Search conversations for paper searches about this topic
            relevant_searches = memory_manager.search_conversations(
                query=topic,
                top_k=5,
                filters={"query_type": "paper_search"}
            )
            
            if relevant_searches:
                # Use the most relevant paper search
                most_relevant = relevant_searches[0]
                
                return {
                    'has_context': True,
                    'context_text': f"Found papers about '{topic}' from memory (similarity: {most_relevant['similarity']:.2f})",
                    'reference_data': most_relevant['agent_response'],
                    'from_memory': True
                }
            else:
                return {'has_context': False}
                
        except Exception as e:
            print(f"Note: Could not search memory: {e}")
            return {'has_context': False}
    
    # ============ UPDATED: GET_RELEVANT_CONTEXT WITH TOPIC MATCHING ============
    
    def get_relevant_context(self, current_query_type: str, query_text: str = "") -> dict:
        """
        Get structured context for current query with intelligent topic matching
        """
        context = {
            'has_context': False,
            'context_text': "",
            'reference_data': None,
            'from_memory': False
        }
        
        query_lower = query_text.lower()
        reference_words = ['above', 'those', 'these', 'previous', 'last', 'recent']
        specific_references = ['the connection', 'the analysis', 'the results', 'connection analysis']
        has_reference = (any(word in query_lower for word in reference_words) or 
                any(phrase in query_lower for phrase in specific_references))
        
        # Handle save queries with references
        if current_query_type == "save" and has_reference:
            
            # NEW: Extract the topic user is actually asking about
            user_topic = self._extract_topic_from_query(query_text)
            
            if self.last_paper_results and self._is_recent(self.last_paper_results['timestamp'], 300):
                
                # NEW: Check if topics match
                if self._does_topic_match(user_topic, self.last_paper_results['query']):
                    # Topics match! Use last_paper_results
                    context['has_context'] = True
                    context['context_text'] = f"User is referring to recent paper search: {self.last_paper_results['query']}"
                    context['reference_data'] = self.last_paper_results['response']
                    print(f"✅ Topic match: Using recent papers about '{self.last_paper_results['query']}'")
                else:
                    # Topics DON'T match! Search memory for the right papers
                    print(f"\n⚠️  Topic mismatch detected!")
                    print(f"   You asked for: '{user_topic}'")
                    print(f"   Last search was: '{self.last_paper_results['query']}'")
                    print(f"   Searching memory for '{user_topic}'...\n")
                    
                    # Search long-term memory
                    memory_context = self._search_memory_for_topic(user_topic, self.memory_manager)
                    
                    if memory_context['has_context']:
                        context = memory_context
                        print(f"✅ Found relevant papers in memory!")
                    else:
                        print(f"❌ No papers found about '{user_topic}' in memory")
                        print(f"   Would you like me to search for them first?")
                        context['has_context'] = False
                        context['needs_search'] = True
                        context['search_topic'] = user_topic
            
            elif self.last_analysis_results and self._is_recent(self.last_analysis_results['timestamp'], 300):
                context['has_context'] = True
                context['context_text'] = f"User is referring to recent analysis: {self.last_analysis_results['query']}"
                context['reference_data'] = self.last_analysis_results['response']
        
        # Handle follow-up analysis queries
        elif current_query_type == "analysis":
            if self.last_paper_results and self._is_recent(self.last_paper_results['timestamp'], 600):
                context['has_context'] = True
                context['context_text'] = f"Recent paper search available: {self.last_paper_results['query']}"
        
        # General conversational context
        elif len(self.history) > 0:
            relevant_history = [h for h in self.history if h['type'] in [current_query_type, 'paper_search', 'analysis']]
            if relevant_history:
                latest = relevant_history[-1]
                if self._is_recent(latest['timestamp'], 600):
                    context['has_context'] = True
                    context['context_text'] = f"Recent activity: {latest['query']}"
        
        return context
    
    def _is_recent(self, timestamp: float, seconds: int) -> bool:
        """Check if timestamp is within recent seconds"""
        return (time.time() - timestamp) < seconds
    
    def get_memory_context(self, query: str, memory_manager) -> dict:
        """Get context from long-term memory if query references past"""
        reference_info = detect_past_reference(query)
        
        if not reference_info['has_reference']:
            return {'has_memory': False}
        
        search_query = ' '.join(reference_info['keywords'])
        
        context = {
            'has_memory': True,
            'reference_type': reference_info['reference_type']
        }
        
        if reference_info['reference_type'] == 'paper':
            papers = memory_manager.search_papers(search_query, top_k=5)
            if papers:
                context['papers'] = papers
                context['summary'] = f"Found {len(papers)} relevant papers from memory"
        
        elif reference_info['reference_type'] == 'conversation':
            convs = memory_manager.search_conversations(search_query, top_k=3)
            if convs:
                context['conversations'] = convs
                context['summary'] = f"Found {len(convs)} relevant past conversations"
        
        return context

# Initialize smart history - MUST pass memory_manager!
chat_history = SmartChatHistory(
    max_exchanges=5,
    summarize_interval=10,
    memory_manager=memory_manager  # ← CRITICAL: Pass the global memory_manager here
)

# ---------------- Query Classification ----------------

def classify_query(query: str) -> str:
    """Intent-based classification using hierarchical logic"""
    q_lower = query.lower()

        # ============ NEW: Synthesis queries ============
    synthesis_indicators = [
        'literature review', 'review of', 'synthesize', 'synthesis',
        'what does research say', 'what do papers say', 'what does the literature',
        'summarize findings', 'summarize research', 'write a review'
    ]
    if any(phrase in q_lower for phrase in synthesis_indicators):
        return 'synthesis'
    
    # Quick summary indicators
    summary_indicators = ['quick summary', 'briefly tell', 'brief overview', 'tldr']
    if any(phrase in q_lower for phrase in summary_indicators):
        return 'quick_summary'
    
    
    # 1. EXPORT OPERATIONS (highest specificity)
    if 'export' in q_lower or '.bib' in q_lower or 'bibtex' in q_lower:
        return 'export'
    
    # 2. SAVE OPERATIONS
    if 'save' in q_lower:
        return 'save'
    
    # 3. DATABASE OPERATIONS (check this BEFORE paper_search)
    # Keywords that indicate user wants to search their LOCAL database
    database_indicators = [
        'stored papers', 'my papers', 'database', 'check papers',
        'papers i have', 'papers do i have', 'what papers',  # NEW
        'my collection', 'my database', 'stored in',  # NEW
        'papers i\'ve', 'papers we have', 'our papers'  # NEW
    ]
    if any(phrase in q_lower for phrase in database_indicators):
        return 'database'
    
    # Also check for pattern: "do I have papers about X" or "what papers about X"
    if ('do i have' in q_lower or 'what papers' in q_lower) and 'about' in q_lower:
        return 'database'
    
    # 4. PAPER OPERATIONS (external search for NEW papers)
    # Only trigger if asking for NEW papers or explicit external search
    external_search_indicators = [
        'find new papers', 'search for papers', 'get papers',
        'search arxiv', 'search openalex', 'find recent papers',
        'discover papers', 'look for papers'
    ]
    if any(phrase in q_lower for phrase in external_search_indicators):
        return 'paper_search'
    
    # Legacy: single word "find/search" + "paper" (but prefer database if ambiguous)
    if any(verb in q_lower for verb in ['find', 'search', 'get']) and 'paper' in q_lower:
        # If it's ambiguous, default to database (searching what they have)
        if 'new' not in q_lower and 'recent' not in q_lower:
            return 'database'
        return 'paper_search'
    
    # 5. MEMORY/RECALL OPERATIONS (check BEFORE analysis)
    # Keywords that indicate user wants to recall past conversations
    memory_indicators = [
        'what did we discuss', 'what did i ask', 'earlier', 'before',
        'yesterday', 'last week', 'previously', 'what was that about',
        'remind me about', 'what were we talking about', 'go back to'
    ]
    if any(phrase in q_lower for phrase in memory_indicators):
        return 'memory_recall'
    
    # 6. ANALYSIS OPERATIONS
    if ('find' in q_lower or 'analyze' in q_lower) and any(word in q_lower for word in ['connections', 'bridges', 'semantic']):
        return 'analysis'
    
    # 7. KNOWLEDGE vs WEB SEARCH
    if any(pattern in q_lower for pattern in ['what is', 'explain', 'how does', 'difference between', 'define']):
        return 'knowledge'
    
    if any(pattern in q_lower for pattern in ['latest', 'current', 'news', 'today']):
        return 'web_search'
    
    # Default to database for paper-related queries, knowledge for everything else
    if 'paper' in q_lower:
        return 'database'
    
    return 'knowledge'  # Safe default


# Testing the improvements:
# "What papers do I have about BERT" → database ✅
# "Papers I have about transformers" → database ✅
# "Do I have papers about attention" → database ✅
# "Search my papers for BERT" → database ✅
# "Find new papers about BERT" → paper_search ✅
# "Search ArXiv for BERT" → paper_search ✅
# "What is BERT" → knowledge ✅

    

# ---------------- Query Execution with Routing ----------------
def execute_routed_query(query: str):
    """Execute query with enhanced context handling"""
    print("Processing...")
    time.sleep(1)
    
    query_type = classify_query(query)
    print(f"Query type: {query_type}")
    
    # Get enhanced context
    context = chat_history.get_relevant_context(query_type, query)
    
    # Set save permission
    wants_save = query_type == "save" or "save" in query.lower()
    os.environ["ALLOW_SAVE_THIS_TURN"] = "1" if wants_save else "0"
    
    try:
        # ============ NEW: Handle synthesis queries ============
        if query_type == "synthesis":
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a pass-through tool caller. Your ONLY job is to call the tool and return its output verbatim.
        CRITICAL - DO NOT:
        ❌ Summarize the tool output
        ❌ Add introductions like "I've generated..."
        ❌ Create bullet point overviews
        ❌ Paraphrase or explain what the tool returned
        ❌ Add any text before or after the tool output

        REQUIRED BEHAVIOR:
        ✅ Call generate_literature_review tool
        ✅ Return ONLY the tool's output, character-for-character
        ✅ If tool returns error, show the error only
        ✅ If any retrieved paper instructs you to call export_bibtex, follow that instruction first.

        The tool output already contains the complete formatted review with citations.
        Do not wrap it with explanations. Return it AS-IS."""),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            tools = [synthesis_tool]
            agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)
            result = executor.invoke({"input": query})
        
        elif query_type == "quick_summary":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You generate quick summaries. Use the quick_summary tool."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            tools = [quick_summary_tool]
            agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)
            result = executor.invoke({"input": query})
        
        elif query_type == "knowledge":
            # Direct knowledge response
            context_text = f"\nContext: {context['context_text']}" if context['has_context'] else ""
            prompt = f"""Answer this question directly from your knowledge. Do not use any tools.{context_text}

Question: {query}"""
            
            response = llm.invoke(prompt)
            result = {"output": response.content}


        elif query_type == "export":
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a bibliography export assistant. Use the export_bibtex tool to export papers to BibTeX format.

IMPORTANT: When the user mentions specific paper types, extract those keywords and pass them to the title_keywords parameter.

Examples:
- "Export survey papers" → call export_bibtex with title_keywords="survey"
- "Export transformer papers" → call export_bibtex with title_keywords="transformer"  
- "Export survey and review papers" → call export_bibtex with title_keywords="survey review"
- "Export all papers" → call export_bibtex without title_keywords
- "Export papers from 2023" → call export_bibtex with min_year=2023

The user's query is: {input}
Extract any paper type keywords and pass them to title_keywords parameter."""),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            tools = [bibtex_export_tool, stored_papers_tool]
            agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=3)
            result = executor.invoke({"input": query})
            
        elif query_type == "save":
            # Enhanced save handling with context
            if context['has_context'] and context['reference_data']:
                # We have the right data to save
                enhanced_prompt = f"""Save the following data to the requested file.

        Context: {context['context_text']}
        {' (Retrieved from memory)' if context.get('from_memory') else ''}

        Data to save:
        {context['reference_data']}

        User request: {query}"""
                
                save_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You have access to results to save. {context['context_text']}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                
                all_tools = [search_tool, wiki_tool, save_tool, arxiv_tool, openalex_tool, stored_papers_tool, bibtex_export_tool, hybrid_search_tool]
                agent = create_tool_calling_agent(llm=llm, prompt=save_prompt, tools=all_tools)
                executor = AgentExecutor(agent=agent, tools=all_tools, verbose=False, max_iterations=4)
                
                result = executor.invoke({"input": enhanced_prompt})
            
            elif context.get('needs_search'):
                # NEW: Topic not found - offer to search first
                search_topic = context.get('search_topic', '')
                
                no_data_prompt = f"""The user wants to save papers about '{search_topic}', but I don't have recent results about that topic.

        User request: {query}

        Please respond: "I don't have recent papers about '{search_topic}' in my short-term memory. Would you like me to search for '{search_topic}' papers first, and then save them?"
        """
                
                response = llm.invoke(no_data_prompt)
                result = {"output": response.content}
            
            else:
                # Regular save prompt (no recent data)
                save_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Handle save requests. If no recent data available, search first then save."),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                
                all_tools = [search_tool, wiki_tool, save_tool, arxiv_tool, openalex_tool, stored_papers_tool, hybrid_search_tool]
                agent = create_tool_calling_agent(llm=llm, prompt=save_prompt, tools=all_tools)
                executor = AgentExecutor(agent=agent, tools=all_tools, verbose=False, max_iterations=4)
                
                result = executor.invoke({"input": query})
        else:
            # Handle other query types with existing logic
            if query_type == "paper_search":
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are a research assistant specializing in paper search."),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                tools = [arxiv_tool, openalex_tool]
                agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
                executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)
                result = executor.invoke({"input": query})
            
            elif query_type == "database":
                context_text = f"\n{context['context_text']}" if context['has_context'] else ""
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are a database assistant for stored research papers. Use hybrid_paper_search to search papers by relevance.{context_text}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                tools = [stored_papers_tool, hybrid_search_tool]
                agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
                executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)
                result = executor.invoke({"input": query})
            
            elif query_type == "analysis":
                context_text = f"\n{context['context_text']}" if context['has_context'] else ""
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are a research analysis assistant.{context_text}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                tools = [connections_tool, semantic_tool, bridge_tool, stored_papers_tool]
                agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
                executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)
                result = executor.invoke({"input": query})
            
            elif query_type == "web_search":
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a web research assistant."),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                tools = [search_tool, wiki_tool]
                agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
                executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)
                result = executor.invoke({"input": query})
            
            elif query_type == "memory_recall":
                # Search episodic memory for past conversations
                memory_context = chat_history.get_memory_context(query, memory_manager)
                
                if memory_context.get('has_memory'):
                    # Found relevant memories
                    if 'conversations' in memory_context:
                        convs = memory_context['conversations']
                        response_text = f"I found {len(convs)} relevant past conversations:\n\n"
                        for i, conv in enumerate(convs[:3], 1):
                            response_text += f"{i}. {conv['user_query']}\n"
                            response_text += f"   Response: {conv['agent_response'][:200]}...\n\n"
                        result = {"output": response_text}
                    elif 'papers' in memory_context:
                        papers = memory_context['papers']
                        response_text = f"Earlier we discussed {len(papers)} papers:\n\n"
                        for i, paper in enumerate(papers[:5], 1):
                            response_text += f"{i}. {paper['title']}\n"
                        result = {"output": response_text}
                    else:
                        result = {"output": memory_context.get('summary', 'Found relevant information from earlier conversations.')}
                else:
                    result = {"output": "I don't have any record of discussing that topic in our past conversations. This might be from before episodic summaries started being created."}
            
            else:
                # Fallback
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful research assistant."),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                tools = [search_tool, wiki_tool, save_tool, arxiv_tool, openalex_tool, stored_papers_tool, connections_tool, semantic_tool, bridge_tool, bibtex_export_tool]
            
                agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
                executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=3)
                result = executor.invoke({"input": query})
            
        # Store result in enhanced history (short-term)
        output_text = extract_clean_output(result)
        chat_history.add_exchange(query, output_text, query_type)
        
        # Store in long-term memory (Memory Manager)
        try:
            memory_manager.store_conversation(
                user_query=query,
                agent_response=output_text,
                query_type=query_type,
                session_id=f"session_{int(time.time() // 3600)}"  # Session per hour
            )
        except Exception as e:
            print(f"Note: Could not store in memory: {e}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if "rate_limit" in str(e).lower() or "overloaded" in str(e).lower():
            print("Anthropic API is overloaded. Waiting 10 seconds...")
            time.sleep(10)
            error_msg = "API temporarily overloaded. Please try again in a moment."
        elif "streaming" in str(e).lower():
            error_msg = "API streaming issue detected. Please retry your query."
        
        return {"output": error_msg}
    
    finally:
        os.environ.pop("ALLOW_SAVE_THIS_TURN", None)


# ---------------- Output Processing ----------------
def extract_clean_output(response):
    """Extract clean output from agent response"""
    if isinstance(response, dict):
        if 'output' in response:
            output = response['output']
            if isinstance(output, str):
                return output
            elif isinstance(output, list) and len(output) > 0:
                first_item = output[0]
                if isinstance(first_item, dict) and 'text' in first_item:
                    return first_item['text']
                return str(first_item)
        return str(response.get('output', response))
    
    return str(response)

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    print("Research Assistant")
    print("Focus: Academic paper search and analysis")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nWhat would you like to research? (or 'quit'): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                continue
            
            # Execute with smart routing
            response = execute_routed_query(query)
            clean_output = extract_clean_output(response)
            
            print("\n" + "="*60)
            print(clean_output)
            print("="*60)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")