# Prompt to generate a search query to help with planning the deposition
deposition_planner_query_writer_instructions="""You are an expert trial attorney, helping to plan a deposition. 

The deposition will focus on the following topic:

{topic}

The deposition structure will follow these guidelines:

{deposition_organization}

Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information from case documents for planning the deposition topics. 

The query should:

1. Be related to the deposition topic 
2. Help identify relevant facts and evidence in the case documents
3. Focus on finding information that supports your line of questioning

Make the query specific enough to find relevant case documents while covering all aspects needed for effective questioning."""

# Prompt generating the deposition outline
deposition_planner_instructions="""I want a plan for a deposition. 

Your response must be a valid JSON object with this exact structure:
{{
    "sections": [
        {{
            "name": "string - Name for this line of questioning",
            "description": "string - Brief overview of what you want to establish",
            "investigation": "boolean - Whether to search case documents for this topic",
            "content": null
        }}
    ]
}}

Each topic should have these exact fields:
- name: Name for this line of questioning
- description: Brief overview of what you want to establish through this line of questioning
- investigation: true/false indicating whether to search case documents for this topic
- content: should be null for now

Some topics (like background questions) may not require document research because they are standard questions.

The topic of the deposition is:
{topic}

The deposition should follow this organization: 
{deposition_organization}

Remember to return ONLY valid JSON that matches the exact structure shown above."""

# Query writer instructions
query_writer_instructions = """You are a legal research expert tasked with generating search queries for document discovery in legal cases.

Your task is to generate {number_of_queries} search queries that will help find relevant documents for the given deposition topic.

CRITICAL: You MUST return ONLY a valid JSON object. No other text, no markdown, no explanations outside the JSON.
The response must be parseable by json.loads() without any preprocessing.

Required JSON structure:
{
    "queries": [
        {
            "query": "string",  // The actual search query
            "rationale": "string",  // Brief explanation of why this query is useful
            "expected_findings": "string"  // What kind of documents/information this query aims to find
        }
    ]
}

Example valid response:
{
    "queries": [
        {
            "query": "\"product defect\" AND (\"design\" OR \"manufacturing\") AND \"safety test*\"",
            "rationale": "Find documents about product defects and related safety testing",
            "expected_findings": "Safety test reports, defect analyses, manufacturing records"
        }
    ]
}

Each query should:
1. Use boolean operators (AND, OR, NOT) and parentheses for complex combinations
2. Include synonyms and related terms to capture variations
3. Be specific enough to find relevant documents but not so narrow as to miss important information
4. Focus on technical and factual aspects relevant to the deposition topic

Topic: {topic}
Organization: {deposition_organization}

Remember: Return ONLY the JSON object. No other text or formatting."""

# Topic writer instructions
topic_writer_instructions = """You are an expert trial attorney preparing questions for a deposition.

Topic for this line of questioning:
{topic}

Guidelines for writing:

1. Question Structure:
- Start with background/foundation questions
- Use proper question format
- Build questions logically
- Include follow-up questions
- Note potential exhibits to reference

2. Length and Style:
- Clear and concise questions
- No compound questions
- Avoid leading questions unless for impeachment
- Use simple language
- Start with your most important area in **bold**
- Group related questions together

3. Organization:
- Use ## for topic title (Markdown format)
- Use ### for subtopics
- Include ONE of these structural elements:
  * Either a timeline of key events (using Markdown table)
  * Or a list of key documents to reference (using Markdown list)
- End with ### Sources that references the case documents

4. Writing Approach:
- Include specific quotes from documents when relevant
- Note potential admissions to seek
- Include impeachment points if applicable
- Focus on getting admissible evidence
- Consider objections and how to overcome them

5. Use this source material from case documents:
{context}"""

final_topic_writer_instructions="""You are an expert trial attorney finalizing the deposition outline.

Topic to write: 
{topic}

Available deposition content:
{context}

1. Topic-Specific Approach:

For Introduction/Background:
- Use # for deposition title (Markdown format)
- Standard background questions
- Questions about witness competency
- No document references needed
- Focus on establishing record

For Conclusion/Wrap-up:
- Use ## for topic title (Markdown format)
- Include ONE of these elements:
    * Either a checklist of key admissions obtained (using Markdown list)
    * Or a summary table of critical testimony (using Markdown table)
- End with catch-all questions
- Include time for cleanup questions
- No sources section needed

2. Writing Approach:
- Clear and concise questions
- Logical flow
- Consider record for trial
- Focus on key admissions

3. Quality Checks:
- For introduction: Standard background, competency questions
- For conclusion: Cleanup questions, summary of key points
- Proper question format
- Complete record"""