import json
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from config.settings import OPENAI_API_KEY, LLM_MODEL
from src.llm.prompts import INTENT_CLASSIFICATION_TEMPLATE, RAG_ANSWER_TEMPLATE, RAG_SYSTEM_PROMPT, INTENT_CLASSIFICATION_SYSTEM_PROMPT, CONVERSATIONAL_SYSTEM_PROMPT


class LLMClient:

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError))
    )
    def _chat_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def generate_answer(self, query, context_chunks, chat_history=None):
        
        context = self._build_context(context_chunks)
        prompt = self._create_prompt_with_context(query, context)

        messages = []
        messages.append({
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        })

        if chat_history:
            messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            response = self._chat_completion(
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
            return "<documents></documents>"

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
        
        # Building context with XML tags
        context_parts = ["<documents>"]
        for i, filename in enumerate(doc_order, 1):
            chunk_texts = docs_dict[filename]
            combined_text = "\n\n".join(chunk_texts)
            
            context_parts.append(
                f'  <document index="{i}" source="{filename}">\n{combined_text}\n  </document>'
            )
        context_parts.append("</documents>")
        
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
        return RAG_ANSWER_TEMPLATE.format(
            context=context,
            query=query
        )

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
            response = self._chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": CONVERSATIONAL_SYSTEM_PROMPT
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
            response = self._chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": INTENT_CLASSIFICATION_SYSTEM_PROMPT
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
        return INTENT_CLASSIFICATION_TEMPLATE.format(message=message)