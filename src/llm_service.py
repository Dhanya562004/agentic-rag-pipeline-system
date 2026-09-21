"""
LLM Service for RAG Pipeline
Handles text generation using Together AI or OpenAI
"""

import os
import traceback
from typing import List, Dict, Any
from dotenv import load_dotenv

try:
    import together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv()

class LLMService:
    """
    Service for handling LLM interactions
    """
    
    def __init__(self, provider: str = "together"):
        """
        Initialize LLM service
        
        Args:
            provider: 'together' or 'openai'
        """
        self.provider = provider
        
        if provider == "together":
            if not TOGETHER_AVAILABLE:
                raise ImportError("Together AI library not available. Please install: pip install together")
            self.api_key = os.getenv("TOGETHER_API_KEY")
            if not self.api_key:
                raise ValueError("TOGETHER_API_KEY not found in environment variables")
            self.client = together.Together(api_key=self.api_key)
            self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not available. Please install: pip install openai")
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=self.api_key)
            self.model = "gpt-3.5-turbo"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_response(self, query: str, context: str, max_tokens: int = 1000) -> str:
        """
        Generate a response using the LLM
        
        Args:
            query: User query
            context: Retrieved context
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            prompt = self._create_prompt(query, context)
            
            if self.provider == "together":
                return self._generate_together(prompt, max_tokens)
            elif self.provider == "openai":
                return self._generate_openai(prompt, max_tokens)
            else:
                return f"Error: Unsupported provider {self.provider}"
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            print(traceback.format_exc())
            return f"Error generating response: {str(e)}"
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a direct and concise assistant that answers questions about France based on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question directly using ONLY the provided context
2. Be extremely concise and to-the-point
3. Use simple, clear language with no fluff
4. DO NOT include any introduction or conclusion phrases
5. DO NOT add pleasantries, signatures, or offers for further assistance
6. If the context doesn't contain relevant information, just say "I don't have information about that in the provided context."
7. Start your answer immediately with the relevant facts

Answer:"""
        
        return prompt
    
    def _generate_together(self, prompt: str, max_tokens: int) -> str:
        """
        Generate response using Together AI
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.5,
                frequency_penalty=1.0,
                presence_penalty=0.8,
                stop=[
                    "Question:", "Context:", "\n\nQuestion:", "\n\nContext:", 
                    "Best regards", "Have a great day", "Is there anything else", 
                    "Please let me know", "Note:", "Hope this helps", "Thank you", 
                    "I hope this", "Let me know", "In conclusion", "To summarize",
                    "In summary", "Feel free", "As requested", "As mentioned",
                    "\n\n", "\nQuestion:"
                ]
            )
            
            if response and response.choices:
                # Remove any trailing pleasantries or common closing phrases
                text = response.choices[0].text.strip()
                # Remove common closers that might have slipped through
                closers = ["Hope this helps", "Thank you", "I hope this", "Let me know", "In conclusion"]
                for closer in closers:
                    if text.endswith(closer):
                        text = text[:-(len(closer))].strip()
                return text
            else:
                return "Error: Invalid response format from Together AI"
        except Exception as e:
            print(f"Error calling Together AI: {str(e)}")
            print(traceback.format_exc())
            return f"Error calling Together AI: {str(e)}"
    
    def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        """
        Generate response using OpenAI
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a direct and concise assistant that answers questions about France based on provided context. Answer in under 200 words, with no introductions or conclusions. Never repeat yourself and avoid any pleasantries or unnecessary remarks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.5,
                presence_penalty=0.8,
                frequency_penalty=1.0
            )
            
            if response and response.choices:
                return response.choices[0].message.content.strip()
            else:
                return "Error: Invalid response format from OpenAI"
        except Exception as e:
            print(f"Error calling OpenAI: {str(e)}")
            print(traceback.format_exc())
            return f"Error calling OpenAI: {str(e)}"
    
    def generate_general_response(self, query: str, max_tokens: int = 500) -> str:
        """
        Generate a general LLM response for queries that do not require RAG context
        """
        try:
            prompt = f"Answer the following question directly, concisely, and accurately:\n\nQuestion: {query}\n\nAnswer:"
            if self.provider == "together":
                return self._generate_together(prompt, max_tokens)
            elif self.provider == "openai":
                return self._generate_openai(prompt, max_tokens)
            else:
                return f"Error: Unsupported provider {self.provider}"
        except Exception as e:
            print(f"Error in generate_general_response: {str(e)}")
            return f"Error generating general response: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Model information
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_configured": bool(self.api_key)
        }

