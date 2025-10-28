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
from gensim.models import Word2Vec
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk

logger = logging.getLogger(__name__)

class FashionWordEmbeddings:
    """Handles fashion-specific word embeddings and disambiguation"""
    def __init__(self):
        self.word2vec_model = None
        self.initialize_nltk()
    
    def initialize_nltk(self):
        """Initialize NLTK resources"""
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.error(f"Error downloading NLTK resources: {e}")

    def train_word2vec(self, documents):
        """Train Word2Vec model on fashion-specific corpus"""
        try:
            # Tokenize documents
            tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
            
            # Train Word2Vec model
            self.word2vec_model = Word2Vec(
                sentences=tokenized_docs,
                vector_size=100,
                window=5,
                min_count=1,
                workers=4
            )
            logger.info("Successfully trained Word2Vec model")
        except Exception as e:
            logger.error(f"Error training Word2Vec model: {e}")

    def disambiguate_word(self, word, context):
        """Perform word sense disambiguation"""
        try:
            # Get word synsets
            synsets = wordnet.synsets(word)
            if not synsets:
                return word

            # Tokenize context
            context_words = word_tokenize(context)
            
            # Find best sense based on context overlap
            max_overlap = 0
            best_sense = synsets[0]
            
            for sense in synsets:
                definition = sense.definition()
                overlap = len(set(context_words) & set(word_tokenize(definition)))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sense = sense
            
            return best_sense.name().split('.')[0]
        except Exception as e:
            logger.error(f"Error in word disambiguation: {e}")
            return word

    def get_similar_fashion_terms(self, word, topn=5):
        """Get similar fashion-related terms using Word2Vec"""
        try:
            if self.word2vec_model and word in self.word2vec_model.wv:
                similar_terms = self.word2vec_model.wv.most_similar(word, topn=topn)
                return [(term, score) for term, score in similar_terms]
        except Exception as e:
            logger.error(f"Error finding similar terms: {e}")
        return []

class FashionRAG:
    def __init__(self):
        self.embedder = None
        self.index = None
        self.documents = []
        self.document_metadata = []
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
        """Load fashion knowledge base from CSV file and create embeddings"""
        try:
            import pandas as pd
            csv_path = os.path.join(settings.BASE_DIR, 'outfit_makeup_rag_dataset.csv')
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found at {csv_path}")
                return

            # Load fashion knowledge base from CSV
            df = pd.read_csv(csv_path)
            self.documents = df['text'].tolist()
            self.document_metadata = df[['id', 'category']].to_dict('records')
            logger.info(f"Loaded {len(self.documents)} documents from CSV file")

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
                gender_instruction = "Provide fashion advice specifically for WOMEN. Focus on women's clothing, accessories, and styling tips."
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
        """Generate comprehensive fashion advice response with enhanced NLP processing"""
        from .debug_logger import backend_logger
        start_time = time.time()

        # Parse context for name and gender
        context_dict = json.loads(context) if context else {}
        user_name = context_dict.get('name', '')

        # Determine gender from user message if not explicitly set in context
        user_gender = context_dict.get('gender', '')
        if not user_gender:
            message_lower = user_message.lower()
            if 'men' in message_lower or 'man' in message_lower or 'male' in message_lower:
                user_gender = 'men'
            elif 'women' in message_lower or 'woman' in message_lower or 'female' in message_lower:
                user_gender = 'woman'
            else:
                user_gender = 'both'  # Default to both if no gender specified

        # Extract session_id from context for logging
        session_id = context_dict.get('session_id', 'unknown')

        # Enhanced greeting detection with NLP
        message_lower = user_message.lower().strip()
        greeting_patterns = [
            'hi', 'hello', 'hey', 'hi!', 'hello!', 'hey!', 'hiya', 'sup', 'yo',
            'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings'
        ]

        # Enhanced farewell detection
        farewell_patterns = [
            'thank you', 'thanks', 'thank you!', 'thanks!', 'bye', 'bye bye', 'goodbye',
            'good bye', 'see you', 'see you later', 'farewell', 'take care', 'bye!',
            'goodbye!', 'see you!', 'take care!'
        ]

        if any(pattern in message_lower for pattern in farewell_patterns):
            # Handle thank you and goodbye messages
            if 'thank' in message_lower:
                farewell_msg = f"You're welcome {user_name}!" if user_name else "You're welcome!"
                response = f"{farewell_msg} I'm glad I could help with your fashion questions. Feel free to come back anytime for more style advice!"
            else:
                farewell_msg = f"Goodbye {user_name}!" if user_name else "Goodbye!"
                response = f"{farewell_msg} It was great chatting about fashion with you. Have a stylish day!"

            # Log AI generation for farewells
            backend_logger.log_ai_generation(
                session_id,
                user_message,
                context,
                response,
                {
                    'model': 'enhanced_farewell_handler',
                    'generation_time': time.time() - start_time,
                    'tokens_used': len(response.split()),
                    'gender_instruction': '',
                }
            )
            return response

        if any(pattern in message_lower for pattern in greeting_patterns):
            # Always provide greeting without asking for gender
            greeting = f"Hi {user_name}!" if user_name else "Hi!"
            response = f"{greeting} I'm SmartStyle, your fashion advisor. What fashion advice can I help you with today?"

            # Log AI generation for greetings
            backend_logger.log_ai_generation(
                session_id,
                user_message,
                context,
                response,
                {
                    'model': 'enhanced_greeting_handler',
                    'generation_time': time.time() - start_time,
                    'tokens_used': len(response.split()),
                    'gender_instruction': '',
                }
            )
            return response

        # Enhanced fashion query processing with NLP categorization
        fashion_categories = {
            'casual': ['casual', 'everyday', 'comfortable', 'relaxed', 'weekend'],
            'business': ['business', 'work', 'office', 'professional', 'corporate'],
            'party': ['party', 'celebration', 'night out', 'club', 'event'],
            'date': ['date', 'romantic', 'dinner', 'special occasion'],
            'swim': ['swim', 'beach', 'pool', 'vacation', 'summer'],
            'formal': ['formal', 'wedding', 'ceremony', 'elegant', 'dressy'],
            'athletic': ['athletic', 'workout', 'gym', 'sports', 'active', 'trekking', 'hiking', 'outdoor', 'adventure'],
            'makeup': ['makeup', 'beauty', 'cosmetics', 'face', 'skincare'],
            'accessories': ['accessories', 'jewelry', 'bags', 'shoes', 'hats'],
            'color': ['color', 'colors', 'matching', 'palette', 'combination']
        }

        # Determine query category for better context
        query_category = 'general'
        message_words = set(message_lower.split())
        for category, keywords in fashion_categories.items():
            if any(keyword in message_words for keyword in keywords):
                query_category = category
                break

        # Enhanced RAG retrieval with category-specific filtering
        rag_start = time.time()
        relevant_docs = self.rag.retrieve_relevant_docs(user_message)

        # Filter and prioritize documents based on category
        filtered_docs = []
        if relevant_docs:
            for doc in relevant_docs:
                doc_lower = doc.lower()
                # Boost relevance for category-specific content
                if query_category != 'general':
                    category_keywords = fashion_categories[query_category]
                    if any(keyword in doc_lower for keyword in category_keywords):
                        filtered_docs.insert(0, doc)  # Add to front for priority
                    else:
                        filtered_docs.append(doc)
                else:
                    filtered_docs = relevant_docs[:5]  # Limit to top 5 for general queries

        rag_time = time.time() - rag_start
        rag_context = "\n".join(filtered_docs[:3]) if filtered_docs else ""  # Use top 3 most relevant

        # Check if we have sufficient relevant knowledge for the query
        has_sufficient_knowledge = len(filtered_docs) >= 2 or (len(filtered_docs) > 0 and query_category == 'general')

        # Log enhanced RAG retrieval
        backend_logger.log_rag_retrieval(
            session_id,
            user_message,
            filtered_docs,
            {
                'method': 'enhanced_tfidf_svd',
                'processing_time': rag_time,
                'threshold': 0.1,
                'category_filter': query_category,
                'docs_filtered': len(filtered_docs)
            }
        )

        # Enhanced Gemini generation with structured prompts
        gen_start = time.time()

        # Create category-specific system prompt
        category_prompts = {
            'casual': "Focus on comfortable, everyday wear that balances style and practicality. Suggest versatile pieces that work for multiple occasions.",
            'business': "Emphasize professional attire that maintains authority while being comfortable. Consider corporate dress codes and modern business casual trends.",
            'party': "Suggest bold, eye-catching outfits perfect for celebrations. Include accessories and styling tips for making a statement.",
            'date': "Recommend romantic, flattering outfits that boost confidence. Consider the venue, time of day, and creating the right impression.",
            'swim': "Focus on swimwear, cover-ups, and beach-ready outfits. Include sun protection and resort wear suggestions.",
            'formal': "Suggest elegant, sophisticated attire for special occasions. Include complete outfit coordination and accessory recommendations.",
            'athletic': "Recommend performance wear that's both functional and stylish. Include layering options and post-workout comfort. For outdoor activities like trekking and hiking, prioritize durability, weather protection, and comfort over fashion.",
            'makeup': "Provide detailed makeup looks with product recommendations. Include step-by-step application and color coordination with outfits.",
            'accessories': "Focus on complementary accessories that elevate any outfit. Include jewelry, bags, shoes, and styling tips.",
            'color': "Provide expert color matching advice. Explain color theory, seasonal palettes, and how to create cohesive looks."
        }

        system_instruction = category_prompts.get(query_category, "Provide comprehensive fashion advice with specific recommendations, styling tips, and practical suggestions.")

        # Enhanced prompt with structure and formatting requirements
        enhanced_prompt = f"""
You are SmartStyle, an expert fashion consultant. Provide detailed, structured fashion advice.

**Category:** {query_category.title()}
**System Instruction:** {system_instruction}

**User Query:** {user_message}

**Available Fashion Knowledge:** {rag_context}

**Response Requirements:**
- Provide 3-5 specific outfit/item recommendations with detailed descriptions
- Include color suggestions and fabric recommendations
- Add styling tips and accessory suggestions
- Mention current trends where relevant
- End with a friendly, contextual closing message
- Use bullet points and bold formatting for clarity
- Keep response engaging and conversational

**Gender Context:** {f"Focus on {user_gender} fashion" if user_gender and user_gender != 'both' else "Provide unisex or gender-neutral suggestions"}

**IMPORTANT:** If the query is about women's fashion (woman/women), provide DISTINCT recommendations that are specifically tailored for women. Avoid suggesting the same outfits as for men. Women and men have different fashion needs, body types, and style preferences. For women, focus on feminine cuts, colors, fabrics, and accessories that complement female forms and styles.

**GENDER HANDLING:** If the user query mentions "men" or "women", provide fashion advice specifically for that gender. If no gender is mentioned, provide suggestions for both men and women with separate sections.

**FALLBACK:** If our knowledge base doesn't contain relevant information for the specific query, clearly state that we don't have specific recommendations for that topic and suggest searching online or consulting fashion experts. Do not provide generic responses when the query is specific.

Format your response professionally but conversationally.
"""

        # Only generate response if we have sufficient knowledge or it's a general query
        gemini_response = None
        if has_sufficient_knowledge:
            gemini_response = self.gemini_gen.generate_response(enhanced_prompt, context)
        gen_time = time.time() - gen_start

        if gemini_response:
            # Enhanced personalization
            if user_name:
                # Replace generic greetings with personalized ones
                gemini_response = gemini_response.replace("Hello!", f"Hello {user_name}!")
                gemini_response = gemini_response.replace("Hi!", f"Hi {user_name}!")
                gemini_response = gemini_response.replace("Hey!", f"Hey {user_name}!")

                # Add personalized touches
                if "recommend" in gemini_response.lower():
                    gemini_response = gemini_response.replace("I recommend", f"For you, {user_name}, I recommend")

            # Add emoji and formatting enhancements
            gemini_response = self._enhance_response_formatting(gemini_response, query_category)

        # Log enhanced AI generation
            backend_logger.log_ai_generation(
                session_id,
                user_message,
                f"{rag_context}\n{context}",
                gemini_response,
                {
                    'model': 'gemini-2.0-flash-enhanced',
                    'generation_time': gen_time,
                    'tokens_used': len(gemini_response.split()),
                    'gender_instruction': 'gender-specific' if user_gender else 'general',
                    'category': query_category,
                    'structured_prompt': True
                }
            )

            # Add friendly closing if not present
            gemini_response = self._add_friendly_closing(gemini_response, query_category)

            # Update RAG with successful response for future learning
            self._update_rag_with_response(user_message, gemini_response, query_category)

            return gemini_response

        # Enhanced fallback with category-specific responses
        if filtered_docs and has_sufficient_knowledge:
            response_parts = []
            if user_name:
                response_parts.append(f"Hello {user_name}!")
            else:
                response_parts.append("Hello!")

            category_responses = {
                'casual': "For comfortable everyday wear, here are some great options:",
                'business': "For professional settings, consider these polished looks:",
                'party': "For celebrations and special events, these outfits will make you shine:",
                'date': "For romantic occasions, these flattering styles work perfectly:",
                'swim': "For beach days and pool time, these swimwear options are ideal:",
                'formal': "For formal events, these elegant pieces are perfect:",
                'athletic': "For workouts and active lifestyles, these performance pieces deliver:",
                'makeup': "For beauty and makeup looks, here are some expert recommendations:",
                'accessories': "To complete your look, these accessories are must-haves:",
                'color': "For color coordination, here are some expert tips:"
            }

            response_parts.append(category_responses.get(query_category, "Based on current fashion trends, here are some suggestions:"))

            for i, doc in enumerate(filtered_docs[:3], 1):
                response_parts.append(f"**Option {i}:** {doc}")

            response_parts.append(f"\nWhat specific aspects of {query_category} fashion interest you most?")
            fallback_response = "\n\n".join(response_parts)

            # Log enhanced fallback
            backend_logger.log_ai_generation(
                session_id,
                user_message,
                rag_context,
                fallback_response,
                {
                    'model': 'enhanced_rag_fallback',
                    'generation_time': time.time() - start_time,
                    'tokens_used': len(fallback_response.split()),
                    'gender_instruction': '',
                    'category': query_category
                }
            )
            return fallback_response

        # Final enhanced fallback
        category_suggestions = {
            'casual': "comfortable everyday outfits",
            'business': "professional work attire - you'll look great in your interview!",
            'party': "celebration and evening wear - have an amazing time!",
            'date': "romantic date night looks - good luck on your date!",
            'swim': "beach and swimwear - enjoy your vacation!",
            'formal': "elegant formal wear - you'll be the center of attention!",
            'athletic': "active and workout wear - crush that workout!",
            'makeup': "beauty and makeup tips - you'll look stunning!",
            'accessories': "complementary accessories - perfect finishing touches!",
            'color': "color coordination advice - you'll look put together!"
        }

        # Check if we have relevant docs for the query
        has_relevant_docs = len(filtered_docs) > 0 and any(
            any(keyword in doc.lower() for keyword in fashion_categories.get(query_category, []))
            for doc in filtered_docs
        )

        if not has_sufficient_knowledge and query_category != 'general':
            # No relevant knowledge for specific query
            final_response = f"I don't have specific recommendations for {query_category} fashion in my current knowledge base. I recommend searching online fashion resources or consulting with fashion experts for the most current trends and advice."
        else:
            suggestion = category_suggestions.get(query_category, "fashion advice and outfit suggestions")
            final_response = f"I'd love to help you with {suggestion}"

        # Log final enhanced fallback
        backend_logger.log_ai_generation(
            session_id,
            user_message,
            context,
            final_response,
            {
                'model': 'enhanced_final_fallback',
                'generation_time': time.time() - start_time,
                'tokens_used': len(final_response.split()),
                'gender_instruction': '',
                'category': query_category
            }
        )
        return final_response

    def _enhance_response_formatting(self, response: str, category: str) -> str:
        """Add emoji and formatting enhancements based on category"""
        category_emojis = {
            'casual': '👕',
            'business': '💼',
            'party': '🎉',
            'date': '💕',
            'swim': '🏖️',
            'formal': '👗',
            'athletic': '🏃‍♀️',
            'makeup': '💄',
            'accessories': '🛍️',
            'color': '🎨'
        }

        emoji = category_emojis.get(category, '✨')

        # Add emoji to bullet points and key sections
        enhanced = response.replace('•', f'{emoji} •')

        # Add category-specific emoji at the beginning if not present
        if not enhanced.startswith(emoji):
            enhanced = f"{emoji} {enhanced}"

        return enhanced

    def _add_friendly_closing(self, response: str, category: str) -> str:
        """Add friendly, contextual closing messages based on fashion category"""
        # Don't add closing if response already has one
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in ['good luck', 'have fun', 'enjoy', 'you\'ll look', 'best wishes', 'hope you']):
            return response

        category_closings = {
            'business': "You'll look professional and confident!",
            'formal': "You'll make a great impression!",
            'party': "Have an amazing time at your event!",
            'date': "Good luck on your date!",
            'casual': "You'll look comfortable and stylish!",
            'athletic': "You'll perform great in your activities!",
            'swim': "Enjoy your time by the water!",
            'makeup': "You'll look fabulous!",
            'accessories': "Those will complete your look perfectly!",
            'color': "Those colors will look amazing on you!"
        }

        closing = category_closings.get(category, "Hope this helps with your fashion choices!")
        return f"{response}\n\n{closing}"

    def _update_rag_with_response(self, user_query: str, ai_response: str, category: str):
        """Update RAG knowledge base with successful AI responses for continuous learning"""
        try:
            # Create a knowledge entry combining the query and response
            knowledge_entry = f"Query: {user_query}\nCategory: {category.title()}\nResponse: {ai_response}"

            # Add to RAG system if it has an update method
            if hasattr(self.rag, 'add_document'):
                self.rag.add_document(knowledge_entry, metadata={'category': category, 'type': 'ai_generated'})

        except Exception as e:
            # Log but don't fail if RAG update fails
            print(f"Warning: Failed to update RAG with response: {e}")
            pass

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
