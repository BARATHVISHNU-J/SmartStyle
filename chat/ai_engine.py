import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from django.conf import settings

logger = logging.getLogger(__name__)

class FashionRAG:
    def __init__(self):
        self.embedder = None
        self.index = None
        self.documents = []
        self.load_knowledge_base()

    def load_embedder(self):
        """Lazy load embedder to avoid startup issues"""
        if self.embedder is None:
            try:
                # Try using a simple word embedding approach
                import numpy as np
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import TruncatedSVD
                
                # Use TF-IDF with SVD for better semantic similarity
                self.vectorizer = TfidfVectorizer(
                    stop_words='english',
                    max_features=1000,
                    ngram_range=(1, 2)
                )
                self.svd = TruncatedSVD(n_components=100, random_state=42)
                self.embedder = 'tfidf_svd'
                logger.info("Loaded TF-IDF with SVD embedder for better semantic similarity")
            except Exception as e:
                logger.error(f"Failed to load TF-IDF SVD embedder: {e}")
                # Fallback to basic TF-IDF
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    self.vectorizer = TfidfVectorizer(stop_words='english')
                    self.embedder = 'tfidf'
                    logger.info("Fallback to basic TF-IDF embedder")
                except Exception as e2:
                    logger.error(f"Failed to load TF-IDF embedder: {e2}")
                    self.embedder = None
        return self.embedder

    def load_knowledge_base(self):
        """Load fashion knowledge base and create embeddings"""
        try:
            # Sample fashion knowledge base
            self.documents = [
                "Casual summer outfit: Light cotton t-shirt, denim shorts, sneakers, sunglasses",
                "Formal business attire: Crisp white shirt, tailored trousers, leather shoes, minimal accessories",
                "Winter fashion: Wool coat, thermal sweaters, boots, scarf, gloves",
                "Athletic wear: Moisture-wicking t-shirt, leggings, running shoes, sports bra",
                "Evening dress: Little black dress, heels, statement jewelry, clutch bag",
                "Beach vacation: Swimsuit, cover-up, sandals, wide-brimmed hat, sunglasses",
                "Office casual: Blouse, chinos, loafers, simple necklace",
                "Date night: Tailored blazer, silk blouse, pencil skirt, heels",
                "Spring fashion: Floral dresses, light jackets, ballet flats, pastel colors",
                "Fall trends: Layered looks, plaid shirts, ankle boots, earthy tones",
                "Color matching: Navy blue pairs well with white, gray, and beige",
                "Accessories: Belts, watches, earrings, scarves enhance any outfit",
                "Budget brands: H&M, Zara, Uniqlo, Target for affordable fashion",
                "Premium brands: Gucci, Prada, Chanel, Louis Vuitton for luxury",
                "Mid-range brands: Banana Republic, J.Crew, Ann Taylor, Nordstrom",
                "Seasonal trends: Sustainable fashion, oversized silhouettes, bold patterns",
                "Gender neutral: Unisex hoodies, straight-leg jeans, sneakers, minimalism",
                "Vintage style: High-waisted jeans, graphic tees, platform shoes, retro accessories"
            ]

            # Create embeddings based on embedder type
            embedder = self.load_embedder()
            if embedder == 'tfidf_svd':
                # Fit TF-IDF and SVD
                tfidf_matrix = self.vectorizer.fit_transform(self.documents)
                self.embeddings = self.svd.fit_transform(tfidf_matrix)
                logger.info(f"Loaded {len(self.documents)} fashion documents with TF-IDF SVD embeddings")
            elif embedder == 'tfidf':
                self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
                logger.info(f"Loaded {len(self.documents)} fashion documents into TF-IDF system")
            else:
                logger.warning("Embedder not available, RAG will be disabled")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.tfidf_matrix = None
            self.embeddings = None

    def _create_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using transformers directly"""
        try:
            import torch
            import torch.nn.functional as F
            
            embeddings = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    # Use mean pooling of the last hidden state
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embedding = sum_embeddings / sum_mask
                    embeddings.append(embedding[0].numpy())
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error creating transformer embeddings: {e}")
            return np.array([])

    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant fashion documents for a query"""
        embedder = self.load_embedder()

        if embedder == 'tfidf_svd' and hasattr(self, 'embeddings') and self.embeddings is not None:
            docs = self._retrieve_tfidf_svd_docs(query, top_k)
        elif embedder == 'tfidf' and hasattr(self, 'tfidf_matrix') and self.tfidf_matrix is not None:
            docs = self._retrieve_tfidf_docs(query, top_k)
        else:
            docs = []

        # Filter out irrelevant results by checking similarity threshold
        if docs:
            # Only return docs that are actually relevant to fashion queries
            fashion_keywords = ['outfit', 'wear', 'fashion', 'style', 'clothing', 'dress', 'shirt', 'pants', 'shoes', 'accessories', 'casual', 'formal', 'business', 'athletic', 'summer', 'winter', 'spring', 'fall', 'color', 'brand', 'trend']
            filtered_docs = []
            query_lower = query.lower()
            for doc in docs:
                doc_lower = doc.lower()
                # Check if query contains fashion keywords or doc is highly relevant
                if any(keyword in query_lower for keyword in fashion_keywords) or any(keyword in doc_lower for keyword in fashion_keywords):
                    filtered_docs.append(doc)
            return filtered_docs[:top_k]

        return docs

    def _retrieve_tfidf_svd_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve documents using TF-IDF SVD embeddings"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Transform query using the same TF-IDF and SVD
            query_tfidf = self.vectorizer.transform([query])
            query_embedding = self.svd.transform(query_tfidf)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            relevant_docs = [self.documents[i] for i in top_indices if similarities[i] > 0.1]
            return relevant_docs
        except Exception as e:
            logger.error(f"Error retrieving TF-IDF SVD documents: {e}")
            return []

    def _retrieve_transformer_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve documents using transformer embeddings"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create query embedding
            query_embedding = self._create_transformer_embeddings([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            relevant_docs = [self.documents[i] for i in top_indices if similarities[i] > 0.1]
            return relevant_docs
        except Exception as e:
            logger.error(f"Error retrieving transformer documents: {e}")
            return []

    def _retrieve_sentence_transformer_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve documents using sentence transformer embeddings"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Create query embedding
            query_embedding = self.embedder_model.encode([query])

            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            relevant_docs = [self.documents[i] for i in top_indices if similarities[i] > 0.1]
            return relevant_docs
        except Exception as e:
            logger.error(f"Error retrieving sentence transformer documents: {e}")
            return []

    def _retrieve_tfidf_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve documents using TF-IDF"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            relevant_docs = [self.documents[i] for i in top_indices if similarities[i] > 0.1]
            return relevant_docs
        except Exception as e:
            logger.error(f"Error retrieving TF-IDF documents: {e}")
            return []

class LocalGenerator:
    def __init__(self):
        self.model_name = None  # Disable local model for efficiency
        self.tokenizer = None
        self.model = None
        # self.load_model()  # Commented out to avoid loading

    def load_model(self):
        """Load local DialoGPT model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("Loaded local DialoGPT model")
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            self.model = None

    def generate_response(self, context: str, max_length: int = 100) -> Optional[str]:
        """Generate response using local model"""
        if not self.model or not self.tokenizer:
            return None

        try:
            inputs = self.tokenizer.encode(context + self.tokenizer.eos_token, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(context):].strip()
        except Exception as e:
            logger.error(f"Error generating local response: {e}")
            return None

class GeminiGenerator:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini with model: {self.model_name}")
        else:
            self.model = None
            logger.warning("Gemini API key not found")

    def generate_response(self, prompt: str, context: str = "") -> Optional[str]:
        """Generate response using Gemini API"""
        if not self.model:
            return None

        try:
            # Parse context to get gender preference
            context_dict = json.loads(context) if context else {}
            user_gender = context_dict.get('gender', '').lower()

            # Create gender-specific prompt
            gender_instruction = ""
            if user_gender == 'men':
                gender_instruction = "Provide fashion advice specifically for MEN. Focus on men's clothing, accessories, and styling tips."
            elif user_gender == 'woman':
                gender_instruction = "Provide fashion advice specifically for WOMAN. Focus on woman's clothing, accessories, and styling tips."
            elif user_gender == 'both':
                gender_instruction = "Provide fashion advice for BOTH men and women. Structure your response with separate sections: **For Men:** and **For Women:**"

            full_prompt = f"""
You are SmartStyle, an AI fashion advisor chatbot. Provide personalized fashion recommendations based on user preferences.

{gender_instruction}

Context: {context}

User query: {prompt}

IMPORTANT: Format your response as follows:
- Use bullet points (•) for main suggestions
- Use **bold text** for key items, brands, and important tips
- Keep responses concise and scannable
- Structure with clear sections when appropriate
- Include specific outfit suggestions, styling tips, and brand recommendations
- Make it visually appealing and easy to read
- Use positive, professional language that engages users

Example format:
• **Outfit Suggestion**: [specific items]
• **Key Accessories**: [important accessories]
• **Styling Tips**: [practical advice]
• **Brand Recommendations**: [specific brands]

For gender-specific advice when "both" is selected:
**For Men:**
• [men's suggestions]

**For Women:**
• [women's suggestions]
"""
            # Add generation config for better reliability
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 1024,
            }

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            if response and hasattr(response, 'text'):
                return response.text.strip()
            else:
                logger.warning("Empty or invalid response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return None

class FashionAI:
    def __init__(self):
        self.rag = FashionRAG()
        self.local_gen = LocalGenerator()
        self.gemini_gen = GeminiGenerator()

    def generate_response(self, user_message: str, context: str = "", timeout: int = 5) -> str:
        """Generate fashion advice response with fallback strategy"""
        start_time = time.time()

        # Parse context for name and gender
        context_dict = json.loads(context) if context else {}
        user_name = context_dict.get('name', '')
        user_gender = context_dict.get('gender', '')

        # Check for simple greetings
        message_lower = user_message.lower().strip()
        if message_lower in ['hi', 'hello', 'hey', 'hi!', 'hello!', 'hey!', 'hiya', 'sup', 'yo']:
            greeting = f"Hi {user_name}!" if user_name else "Hi there!"
            return f"{greeting} I'm SmartStyle, your fashion assistant. I can help you with outfit suggestions, style advice, and fashion tips. What would you like to know?"

        # Try RAG first (if available)
        relevant_docs = self.rag.retrieve_relevant_docs(user_message)
        rag_context = "\n".join(relevant_docs) if relevant_docs else ""

        # Direct to Gemini (primary method for efficiency)
        gemini_response = self.gemini_gen.generate_response(user_message, f"{rag_context}\n{context}")
        if gemini_response:
            # Personalize response with name if available
            if user_name:
                gemini_response = gemini_response.replace("Hello!", f"Hello {user_name}!")
                gemini_response = gemini_response.replace("Hi!", f"Hi {user_name}!")
                gemini_response = gemini_response.replace("Hey!", f"Hey {user_name}!")
            return gemini_response

        # Fallback to RAG-based response if Gemini fails
        if relevant_docs:
            # Create a more conversational response using RAG
            response_parts = []
            if user_name:
                response_parts.append(f"Hello {user_name}!")
            else:
                response_parts.append("Hello!")

            response_parts.append("Based on fashion trends, here are some suggestions:")

            for doc in relevant_docs[:2]:
                # Format the document as a bullet point
                response_parts.append(f"• {doc}")

            response_parts.append("\nWhat specific fashion advice are you looking for?")
            return " ".join(response_parts)

        # Final fallback
        greeting = f"Hello {user_name}!" if user_name else "Hello!"
        return f"{greeting} I'd love to help with your fashion questions! Could you tell me more about what you're looking for - occasion, style preferences, or specific items?"

    def update_context(self, session_context: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Update conversation context with user preferences"""
        # Simple context tracking - extract preferences from messages
        message_lower = user_message.lower()

        if 'male' in message_lower or 'man' in message_lower:
            session_context['gender'] = 'male'
        elif 'female' in message_lower or 'woman' in message_lower:
            session_context['gender'] = 'female'

        if 'casual' in message_lower:
            session_context['style'] = 'casual'
        elif 'formal' in message_lower or 'business' in message_lower:
            session_context['style'] = 'formal'
        elif 'athletic' in message_lower or 'sport' in message_lower:
            session_context['style'] = 'athletic'

        if 'budget' in message_lower or 'cheap' in message_lower:
            session_context['budget'] = 'budget'
        elif 'premium' in message_lower or 'luxury' in message_lower:
            session_context['budget'] = 'premium'

        return session_context
