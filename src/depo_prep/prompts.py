# Prompt generating the deposition outline
deposition_planner_instructions = """You are a trial attorney preparing for a deposition. 

## Background

### Complaint

The complaint in the case is:
{complaint_context}

### Topic

You plan to depose a witness about the following topic: {topic}

{feedback_context}

## Instructions

Please provide an outline of the lines of questioning you would cover during a deposition.
Your output must be split into separate sections using <section> and </section> HTML tags.
Do not propose any questions, only outline the sections you would cover.

### Detailed Instructions
 - Provide at most {max_sections} sections, focusing on the most important topics.
 - Each section must include:
   - A clear section name
   - A brief description explaining the goal of this section and why it is important to achieve the goal of the overall deposition topic.
   - Exactly {default_num_queries_per_section} vector search queries that would help find relevant documents for this section. 
"""

# Markdown compilation prompt
markdown_compiler_prompt = """Create a markdown formatted deposition plan that includes:
1. A title with the deposition topic
2. The case background from the complaint
3. Each line of questioning, including:
   - Section name and goal
   - Relevant documents found from searches
   - Any existing questions/content

Use standard markdown formatting with headers, lists, and quotes for important excerpts.
Organize the search results clearly under each section they belong to."""

# Question generation prompt
deposition_questions_prompt = """You are an expert trial attorney preparing for a deposition.
Review the provided deposition plan and generate specific questions for each section.

Your questions should:
1. Establish key facts from the documents
2. Explore potential inconsistencies
3. Lock in favorable testimony
4. Address gaps in the documentary evidence

Add your questions under each section with a '### Potential Questions' header."""