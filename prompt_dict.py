Prompts = {
    
    "summarize_details":\
"""You are a helpful assistant that summarizes a part of a novel.
Goals:
- Include only salient details: main characters, key relationships, core plot points, and any rules/regulations if present.
- No fabrication.

STRICT REQUIREMENTS:
1) **Explicit Anti-Repetition Protocol**: 
   - Before writing each point, check if it semantically overlaps with any existing point
   - If semantic overlap >70%, merge into a single comprehensive point
   - Use varied vocabulary and syntactic structures for similar concepts
2) **Incremental Information Filter**:
   - Each point must introduce distinct narrative value
   - Flag and remove points that only rephrase previous content
   - Focus on information advancement, not reinforcement
3) **Content Density Enforcement**:
   - Remove all transitional phrases (e.g., "the text shows", "it is mentioned")
   - Direct substantive statements only
   - Maximum information-to-word ratio

CONTENT SCANNING CHECKLIST:
✓ Character introductions and developments
✓ Plot progression elements  
✓ Relationship dynamics changes
✓ Rule/system establishments
✓ NEW information not previously covered

Here is the content:
Content: {content}

Now, please generate a dense, non-repetitive summary.
Summary: """,


    "summarize_summary":\
"""You are a helpful assistant that further summarizes multiple summaries of a novel.
Goals:
- Include only salient details: main characters, key relationships, core plot points, and any rules/regulations if present.
- Base the consolidation on the question if provided. No fabrication.

STRICT PROCESSING PIPELINE:
1) **Cross-Summary Deduplication**:
   - Create semantic clusters across all input summaries
   - Select the most comprehensive version from each cluster
   - Discard redundant variations
2) **Abstraction Hierarchy Building**:
   - Group specific instances under general categories
   - Replace multiple examples with pattern descriptions
   - Elevate concrete events to conceptual developments
3) **Information Compression**:
   - Remove illustrative examples while keeping core facts
   - Combine related character actions into behavioral patterns
   - Condense sequential events into overarching narratives

CONSOLIDATION FILTERS:
• Merge character-specific actions into role-based descriptions
• Replace repeated event patterns with "multiple instances" phrasing  
• Elevate dialog examples to relationship dynamics

Here are the summaries:
Summary: {summary}

Now, please apply consolidation and abstraction to create a compressed summary.
Summary: """,


    "QA_prompt_answer":\
"""You are a helpful assistant, you are given a question, please answer the question based on the given evidences. The answer should be a short sentence that supported by the given edidences and matches the requirements of the question. You should not assume any information beyond the evidence. You should only output the answer. 

Question: {question}
Evidence: {evidence}


Answer: """,

   "QA_prompt_options":\
"""You are a precise QA assistant that selects options based strictly on evidence.

EVIDENCE PROCESSING:
- Analyze all provided evidence fragments
- Extract factual information relevant to the question
- Ignore unsupported information

DECISION CRITERIA:
- Only choose option with direct evidence support
- If multiple options seem plausible, select the one with strongest evidence backing
- If no clear evidence, do not guess

Question: {question}
Evidence: {evidence}


Output only the letter option (A/B/C/D):
Answer: """,
}