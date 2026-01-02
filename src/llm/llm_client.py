import json
from openai import OpenAI
from config.settings import OPENAI_API_KEY, LLM_MODEL


class LLMClient:

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL

    def generate_answer(self,query,context_chunks,chat_history = None):
        
        context = self._build_context(context_chunks)
        prompt = self._create_prompt_with_context(query, context)

        messages = []
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant that answers questions "
                       "based on provided knowledge base documents and "
                       "conversation history. Use both the documents and "
                       "previous conversation context. Be explicit about what "
                       "you know and what you don't know. Never invent "
                       "information."
        })

        if chat_history:
            messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            normalized = self._normalize_response(result)
            
            # Enhance sources with manual_input content
            normalized["sources"] = self._enhance_sources(
                sources=normalized["sources"],
                chunks=context_chunks
            )
            
            return normalized

        except json.JSONDecodeError as e:
            return {
                "answer": "Error parsing LLM response.",
                "confidence": "low",
                "missing_info": ["Unable to process response"],
                "enrichment_suggestions": [],
                "sources": [],
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": "low",
                "missing_info": ["System error occurred"],
                "enrichment_suggestions": [],
                "sources": [],
            }

    def _build_context(self, chunks):
        if not chunks:
            return "No relevant documents found."

        # Group chunks by filename (preserving order of first appearance)
        docs_dict = {}
        doc_order = []
        
        for chunk in chunks:
            filename = chunk.get("filename", "Unknown")
            text = chunk.get("text", "")
            
            if filename not in docs_dict:
                docs_dict[filename] = []
                doc_order.append(filename)
            docs_dict[filename].append(text)
        
        # Build context with unique document headers
        context_parts = []
        for i, filename in enumerate(doc_order, 1):
            chunk_texts = docs_dict[filename]
            # Combine all chunks from this document
            combined_text = "\n\n".join(chunk_texts)
            context_parts.append(
                f"[Document {i}: {filename}]\n{combined_text}\n"
            )
        
        return "\n".join(context_parts)

    def _enhance_sources(self,sources,chunks):
        """
        Enhance manual_input sources with their content.
        """
        from config.settings import MANUAL_INFO_FILENAME
        
        # Build a mapping of filename -> text for manual_input files
        manual_input_map = {}
        for chunk in chunks:
            filename = chunk.get("filename", "")
            # Check if it's a manual input file (new or old format)
            if filename == MANUAL_INFO_FILENAME or filename.startswith("manual_input_"):
                text = chunk.get("text", "").strip()
                if filename not in manual_input_map:
                    manual_input_map[filename] = text
                else:
                    # Concatenate multiple chunks from same file
                    manual_input_map[filename] += " " + text
        
        enhanced = []
        for source in sources:
            if source in manual_input_map:
                content = manual_input_map[source]
                # Truncate if too long
                max_length = 150
                if len(content) > max_length:
                    content = content[:max_length] + "..."
                enhanced.append(f"{source} - {content}")
            else:
                # Not a manual input, keep as-is
                enhanced.append(source)
        
        return enhanced

    def _create_prompt_with_context(self, query, context):
        
        return f"""Based on the following knowledge base documents and our conversation history, answer the question.
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

Important:
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
"""

    def _normalize_response(self, response):
        normalized = {
            "answer": response.get("answer", "No answer provided"),
            "confidence": response.get("confidence", "low"),
            "missing_info": response.get("missing_info", []),
            "enrichment_suggestions": response.get(
                "enrichment_suggestions", []
            ),
            "sources": response.get("sources", []),
        }

        # Ensure lists are lists
        if not isinstance(normalized["missing_info"], list):
            normalized["missing_info"] = []
        if not isinstance(normalized["enrichment_suggestions"], list):
            normalized["enrichment_suggestions"] = []
        if not isinstance(normalized["sources"], list):
            normalized["sources"] = []

        return normalized

    def generate_conversational_response(self, message):
        """
        Generate a brief response for conversational (non-informational) messages.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful knowledge base assistant. "
                            "The user just sent a conversational message "
                            "(not a question or new information). "
                            "Respond naturally and briefly. "
                            "Keep your response friendly and concise (1-2 sentences)."
                        )
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # Fallback to simple responses on error
            return self._get_conversational_fallback(message)


    def _get_conversational_fallback(self, message):
        """
        Provide fallback responses when LLM call fails.
        """
        message_lower = message.lower().strip()
        
        if any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! Let me know if you need anything else."
        
        if any(word in message_lower for word in ["okay", "ok", "got it", "sure"]):
            return "Great! Feel free to ask if you have any questions."
        
        if "later" in message_lower:
            return "Sounds good! I'm here whenever you're ready."
        
        return "I'm here to help! You can ask me questions or provide information to add to my knowledge base."

    

    def classify_intent(self, message):
        """
        Classify user message intent.
        """
        prompt = self._create_classification_prompt(message)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intent classification system. "
                                 "Analyze messages and classify their intent."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            return result.get("intent", "information_request")

        except Exception as e:
            return {
                "intent": "information_request",
                "confidence": "low",
                "reasoning": f"Classification error: {str(e)}"
            }


    def _create_classification_prompt(self, message):
        return f"""Analyze the following user message and classify its intent:

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
}}"""

        