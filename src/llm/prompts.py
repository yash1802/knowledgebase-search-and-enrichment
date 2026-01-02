INTENT_CLASSIFICATION_TEMPLATE = """
Analyze the following user message and classify its intent:

Message: "{message}"

Classification criteria:

1. information_request: User is asking a question or requesting information
   - Contains questions (what, when, where, who, how, why, ?)
   - Request verbs (tell, show, explain, describe, list, find)
   - Seeking knowledge from existing data
   - Examples: "What universities did Yash attend?", "Tell me about Aks", "Show me revenue data"

2. information_provision: User is stating facts to be stored
   - Declarative statements with concrete facts
   - Contains entities, relationships, dates, numbers, descriptions
   - NOT asking questions
   - Providing new information to be remembered
   - Examples: "Yash graduated from UCLA in 2023", "Aks and Yash are brothers", "The Q4 revenue was $5M"

3. conversational: Casual conversation, acknowledgments, no actionable content
   - Gratitude expressions (thanks, thank you)
   - Acknowledgments (okay, got it, sure, alright)
   - Dismissals or delays (maybe later, I'll do that later)
   - No concrete information or questions
   - Examples: "Thanks!", "Okay, I'll do that later", "Got it"

Respond with JSON in this exact format:
{{
    "intent": "information_request|information_provision|conversational",
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation of why this classification was chosen"
}}
"""

INTENT_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an intent classification system. "
    "Analyze messages and classify their intent."
)

RAG_ANSWER_TEMPLATE = """
Based on the following knowledge base documents and our conversation history, answer the question.
If the information is not in the documents, explicitly state what is missing.

Knowledge Base Documents:
{context}

Current Question: {query}

Provide your response as a JSON object with the following structure:
{{
    "answer": "Your answer based on the documents and conversation history. If information is missing, state that explicitly. The value of the 'answer' key must be a string.",
    "confidence": "high|medium|low",
    "missing_info": ["list of specific information gaps"],
    "enrichment_suggestions": [
        "Suggest specific sources where the missing information can be found"
    ],
    "sources": ["list of document filenames used"],
}}

Important guidelines:
- Analyze the content inside <documents> tags to answer the query.
- Answer based on the provided documents and conversation history
- If the provided documents don't have enough information or they have no information at all that can be used to answer the question, list the specific information that is missing in "missing_info". 
- Absence of evidence is not evidence of absence. 
  * If the documents do not explicitly confirm or deny a fact, treat it as missing information. 
  * Do not infer a negative answer (“No”) solely because information is absent. 
  * Only answer “No” with empty missing_info and high confidence if the documents explicitly state the negation.
- When missing_info is not empty, provide enrichment_suggestions that focus on WHERE to find the information
- Enrichment suggestions should recommend specific sources. For example:
  * For personal info: Resumes, social media (Facebook, Instagram, X), employee directories, emergency contact forms, etc.
  * For professional info: Resumes, social media geared towards professionals (LinkedIn, Glassdoor, etc), job search websites (Indeed, Wellfound, etc), etc.
  * For business data: Annual reports, SEC filings, investor relations pages, earnings calls, company websites
  * For factual info: Official websites, Wikipedia, IMDb, government databases, news articles
  * For academic info: Research papers, university websites, Google Scholar, academic journals
- Be concrete and specific: "Check the relationship status if available on Facebook" not just "Check social media"
- Provide multiple source options when applicable
- When missing_info is empty, leave enrichment_suggestions as an empty list
- Set confidence using the following rubric:
  * High: The question is fully answered using the provided documents. All key facts are supported by the documents. No missing_info.
  * Medium: The question is partially answered. Some key details are missing, ambiguous, or inferred. missing_info is non-empty.
  * Low: The documents do not meaningfully answer the question. The information required to answer the question is missing, missing_info is non-empty.
- When assessing confidence, consider only information **directly relevant to the specific query**, not incidental context about the entity. This ensures confidence reflects the certainty of the answer to the specific fact being asked.

Examples: 

1. Explicit evidence → High confidence:
Q: Has Person A studied at University X?
Docs: "Master’s in Management, University X (2024–2025)"
Response:
{{
  "answer": "Yes, Person A is currently a candidate for a Master's in Management at University X.",
  "confidence": "high",
  "missing_info": [],
  "enrichment_suggestions": [],
  "sources": [<source 1>,...,<source n>]
}}

2. Absence of evidence → Low confidence:
Q: Has Person A studied management at University Y?
Docs: Only University X mentioned; no info on University Y
Response:
{{
  "answer": "The documents do not provide information confirming whether Person A has studied management at University Y.",
  "confidence": "low",
  "missing_info": ["Whether Person A studied management at University Y"],
  "enrichment_suggestions": [
    "Check Person A's LinkedIn education section"
  ],
  "sources": [<source 1>,...,<source n>]
}}

3. No relevant information → Low confidence:
Q: Is Person B currently married?
Docs: Only professional history provided
Response:
{{
  "answer": "The documents do not contain information about Person B's current marital status.",
  "confidence": "low",
  "missing_info": ["Person B's current marital status"],
  "enrichment_suggestions": [
    "Check public social media profiles for relationship status",
    "Consult official personal records if available"
  ],
  "sources": [<source 1>,...,<source n>]
}}

4. Explicit negation → High confidence "No":
Q: Has Person C worked at Company Z?
Docs: "Employment history: Company X (2019 – 2023)"; "Person C has not worked at Company Z."
Response:
{{
  "answer": "No, Person C has not worked at Company Z.",
  "confidence": "high",
  "missing_info": [],
  "enrichment_suggestions": [],
  "sources": [<source 1>,...,<source n>]
}}

5. Full info is guaranteed to be in the knowledge base and the user query asks for a fact that is not in said info.
Q: Has Person C worked at Company Z?
Docs: Employment history that is complete and company Z is not mentioned as one of Person C's positions.
Response:
{{
  "answer": "No, Person C has not worked at Company Z.",
  "confidence": "high",
  "missing_info": [],
  "enrichment_suggestions": [],
  "sources": [<source 1>,...,<source n>]
}}


6. Partial info → Medium confidence:
Q: What is Person D’s current degree program?
Docs: "Enrolled in a master's program" (degree name missing)
Response:
{{
  "answer": "The documents indicate that Person D is enrolled in a master's program, but the specific degree is not stated.",
  "confidence": "medium",
  "missing_info": ["Name of the specific master's degree program"],
  "enrichment_suggestions": [
    "Check Person D's LinkedIn profile for full degree details",
    "Review official university enrollment records"
  ],
  "sources": [<source 1>,...,<source n>]
}}
"""

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions "
    "based on provided knowledge base documents and "
    "conversation history. Use both the documents and "
    "previous conversation context. Be explicit about what "
    "you know and what you don't know. Never invent "
    "information."
)

CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are a helpful knowledge base assistant. "
    "The user just sent a conversational message "
    "(not a question or new information). "
    "Respond naturally and briefly. "
    "Keep your response friendly and concise (1-2 sentences)."
)