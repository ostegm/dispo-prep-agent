# Prompt generating the deposition outline
deposition_planner_instructions = """You are an expert trial attorney reviewing case documents and preparing for a deposition.

## Overall Goal
Your overall goal of this deposition is: {topic}

First, provide a strategic summary of your understanding and approach:

<summary>
Analyze the deposition goals and documents to provide:
1. A clear statement of what we aim to achieve through this deposition
2. Key themes or facts we need to establish
3. Our overall strategy for questioning this witness
</summary>

Then, build a comprehensive outline of the lines of questioning you would cover during the deposition.
Your outline should focus on how you will:
1. Establish key facts from the documents
2. Explore potential inconsistencies
3. Lock in favorable testimony
4. Identify and address gaps in the documentary evidence 

## Background
Below are all the collected documents relevant to the case:
{document_context}

## Instructions

Please provide an outline of the lines of questioning you would cover during a deposition.
Your output must be split into separate sections using <section> and </section> HTML tags.
Do not propose any questions, only outline the sections you would cover.

### Detailed Instructions
 - Provide a summary of the overall goal of the deposition warpped in <summary> tags. The text inside this tag should just be text not html.
 - Provide at most {max_sections} sections, focusing on the most important topics.
 - Each section must include:
   - A clear section name
   - A brief description explaining the goal of this section and why it is important to achieve the goal of the overall deposition topic.
   - A list of quotes from the provided context documents that might be useful for achieving the goal of this section. Each quote should include the raw text as well as the reason it was included.
"""

# Prompt for processing raw sections into structured format
section_processor_prompt = """Convert this deposition section into a structured format with:
- A name field as a brief title
- A description field explaining the section
- A content field containing the relevant document quotes and their explanations

Format the output as a JSON object with these exact fields."""

# Markdown compilation prompt
markdown_compiler_prompt = """Create a comprehensive markdown-formatted deposition plan that effectively organizes the provided sections and their supporting evidence.

For each section, format the content as follows:

## [Section Name]

### Goal and Importance
[Include the section's description explaining its goal and importance]

### Key Documents and Quotes
[For each quote in the section's content:]
- **Quote**: [The actual quote text]
  - **Source**: [Document source]
  - **Significance**: [Explanation of why this quote is important]

Use proper markdown formatting:
- Use headers (##, ###) for section organization
- Use bold (**) for emphasis and quote attribution
- Use block quotes (>) for longer document excerpts
- Use bullet points (-) for lists of quotes and their explanations

Ensure each quote is properly attributed to its source document and its significance to the deposition goal is clearly explained."""

# Question generation prompt
deposition_questions_prompt = """You are an expert trial attorney preparing for a deposition. 
Review the provided plan with the goal of adding questions that will help you achieve the goals of the deposition.

## Strategy Context
The deposition summary provided outlines our goals and key themes. Use this strategic context to ensure each question serves our overall objectives.

## Question Guidelines
Your questions for each section should:
1. Start with foundational questions to establish basic facts
2. Progress to more specific questions about key documents and events
3. Build towards questions that expose inconsistencies or credibility issues
4. Include follow-up questions to address potential evasive answers

## Question Types to Include
- Foundation questions to establish witness knowledge and involvement
- Timeline questions to establish sequence of events
- Document-based questions referencing specific exhibits
- Impeachment questions when contradictions exist
- Open-ended questions to explore gaps in documentation

## Formatting Instructions
For each section in the plan:
1. Keep the existing section name and goal
2. Add a "Questions" subsection with:
   - Numbered questions in a logical sequence
   - Brief notes explaining the purpose of key questions
   - Suggested follow-up questions indented under main questions
   - References to specific documents or quotes when relevant

Format your output in Markdown, preserving the existing section structure and adding questions under each section."""

## New one-shot deposition plan prompt with questions
deposition_one_shot_instructions = """You are a trial attorney preparing for a deposition.

## Background
Below are all the collected documents relevant to the case:
{complaint_context}

## Deposition Topic
You plan to depose a witness concerning the following topic:
{topic}

## Deposition Plan with Questions
Based on the above documents and the deposition topic, develop a comprehensive deposition plan that includes:
  - A clear title.
  - Several sections detailing lines of inquiry. For each section, provide:
      - A section name.
      - A description of the focus or objective of the section.
      - Exactly {default_num_queries_per_section} specific questions to ask during the deposition derived from the documents.

Format your output in Markdown using headers, lists, and quotes where necessary.
You are allowed a maximum of {max_sections} sections.
"""