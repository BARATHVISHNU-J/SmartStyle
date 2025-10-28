# SmartStyle Chatbot - Project Operations Documentation

## Overview
SmartStyle is a Django-based AI fashion advisor chatbot that uses Google's Gemini AI with Retrieval-Augmented Generation (RAG) for personalized fashion recommendations. This document provides comprehensive technical documentation of all operations, methods, and processes in the project.

## Table of Contents
1. [AI Engine Operations (`chat/ai_engine.py`)](#ai-engine-operations)
2. [Django Views Operations (`chat/views.py`)](#django-views-operations)
3. [Database Models (`chat/models.py`)](#database-models)
4. [Data Population Operations (`populate_data.py`)](#data-population-operations)
5. [Backend Processing Pipeline](#backend-processing-pipeline)

---

## AI Engine Operations (`chat/ai_engine.py`)

### Class: FashionWordEmbeddings
**Purpose**: Handles fashion-specific word embeddings and semantic processing for enhanced fashion terminology understanding.

#### Method: `__init__()`
- **Purpose**: Initialize the word embeddings handler
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Initializes word2vec_model to None
  2. Calls initialize_nltk() to download required NLTK resources
- **Dependencies**: nltk library

#### Method: `initialize_nltk()`
- **Purpose**: Download and setup NLTK resources for natural language processing
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Downloads wordnet corpus
  2. Downloads punkt tokenizer
  3. Downloads averaged_perceptron_tagger
  4. Logs errors if download fails
- **Dependencies**: nltk

#### Method: `train_word2vec(documents)`
- **Purpose**: Train Word2Vec model on fashion-specific corpus for semantic similarity
- **Input Parameters**:
  - `documents`: List[str] - List of text documents for training
- **Output**: None
- **Functionality**:
  1. Tokenize each document using word_tokenize
  2. Train Word2Vec model with vector_size=100, window=5, min_count=1, workers=4
  3. Log successful training
- **Dependencies**: gensim.Word2Vec, nltk.tokenize

#### Method: `disambiguate_word(word, context)`
- **Purpose**: Perform word sense disambiguation using WordNet
- **Input Parameters**:
  - `word`: str - Word to disambiguate
  - `context`: str - Surrounding context text
- **Output**: str - Disambiguated word
- **Functionality**:
  1. Get all synsets for the word from WordNet
  2. Tokenize context words
  3. Calculate overlap between each synset definition and context
  4. Return the synset with maximum overlap
- **Dependencies**: nltk.corpus.wordnet, nltk.tokenize

#### Method: `get_similar_fashion_terms(word, topn=5)`
- **Purpose**: Find similar fashion-related terms using Word2Vec
- **Input Parameters**:
  - `word`: str - Input word
  - `topn`: int - Number of similar terms to return (default: 5)
- **Output**: List[Tuple[str, float]] - List of (term, similarity_score) tuples
- **Functionality**:
  1. Check if word2vec_model exists and word is in vocabulary
  2. Use most_similar() to find similar terms
  3. Return list of similar terms with scores
- **Dependencies**: gensim.models.Word2Vec

### Class: FashionRAG
**Purpose**: Implements Retrieval-Augmented Generation system for fashion knowledge using vector similarity search.

#### Method: `__init__()`
- **Purpose**: Initialize the RAG system
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Initialize embedder, index, documents, document_metadata to None/empty
  2. Call load_knowledge_base() to load fashion data
- **Dependencies**: None

#### Method: `load_embedder()`
- **Purpose**: Lazy load the embedding model (TF-IDF with SVD)
- **Input Parameters**: None
- **Output**: str - Embedder type ('tfidf_svd', 'tfidf', or None)
- **Functionality**:
  1. If embedder not loaded, try to initialize TF-IDF with SVD
  2. Fallback to basic TF-IDF if SVD fails
  3. Return embedder type or None if both fail
- **Dependencies**: sklearn.feature_extraction.text.TfidfVectorizer, sklearn.decomposition.TruncatedSVD

#### Method: `load_knowledge_base()`
- **Purpose**: Load fashion knowledge base from CSV file and create embeddings
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Load CSV file 'outfit_makeup_rag_dataset.csv'
  2. Extract documents and metadata
  3. Create embeddings based on embedder type
  4. Log loading status and errors
- **Dependencies**: pandas, sklearn

#### Method: `retrieve_relevant_docs(query, top_k=3)`
- **Purpose**: Retrieve relevant fashion documents for a query using vector similarity
- **Input Parameters**:
  - `query`: str - Search query
  - `top_k`: int - Number of documents to retrieve (default: 3)
- **Output**: List[str] - List of relevant document texts
- **Functionality**:
  1. Determine embedder type and call appropriate retrieval method
  2. Filter results based on fashion keywords and similarity threshold
  3. Return filtered relevant documents
- **Dependencies**: sklearn.metrics.pairwise.cosine_similarity

#### Method: `_retrieve_tfidf_svd_docs(query, top_k=3)`
- **Purpose**: Retrieve documents using TF-IDF with SVD embeddings
- **Input Parameters**:
  - `query`: str - Search query
  - `top_k`: int - Number of documents to retrieve
- **Output**: List[str] - Relevant documents
- **Functionality**:
  1. Transform query using TF-IDF vectorizer
  2. Transform to SVD space
  3. Calculate cosine similarities
  4. Return top-k documents above similarity threshold
- **Dependencies**: sklearn

#### Method: `_retrieve_tfidf_docs(query, top_k=3)`
- **Purpose**: Retrieve documents using basic TF-IDF
- **Input Parameters**:
  - `query`: str - Search query
  - `top_k`: int - Number of documents to retrieve
- **Output**: List[str] - Relevant documents
- **Functionality**:
  1. Transform query using TF-IDF vectorizer
  2. Calculate cosine similarities with document matrix
  3. Return top-k documents above similarity threshold
- **Dependencies**: sklearn

### Class: LocalGenerator
**Purpose**: Local language model generator (currently disabled for efficiency)

#### Method: `__init__()`
- **Purpose**: Initialize local generator
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Set model_name to None (disabled)
  2. Initialize tokenizer and model to None
- **Dependencies**: None

#### Method: `load_model()`
- **Purpose**: Load local DialoGPT model (currently disabled)
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Load tokenizer and model from pretrained DialoGPT
  2. Set model to evaluation mode
- **Dependencies**: transformers

#### Method: `generate_response(context, max_length=100)`
- **Purpose**: Generate response using local model
- **Input Parameters**:
  - `context`: str - Conversation context
  - `max_length`: int - Maximum response length
- **Output**: Optional[str] - Generated response or None
- **Functionality**:
  1. Encode context with tokenizer
  2. Generate response using model with sampling parameters
  3. Decode and return response (without context)
- **Dependencies**: transformers, torch

### Class: GeminiGenerator
**Purpose**: Google Gemini AI response generator for fashion advice

#### Method: `__init__()`
- **Purpose**: Initialize Gemini API client
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Get API key and model name from Django settings
  2. Configure genai with API key
  3. Initialize GenerativeModel
- **Dependencies**: google.generativeai, django.conf.settings

#### Method: `generate_response(prompt, context="")`
- **Purpose**: Generate fashion advice response using Gemini AI
- **Input Parameters**:
  - `prompt`: str - User query
  - `context`: str - Conversation context (JSON string)
- **Output**: Optional[str] - Generated response or None
- **Functionality**:
  1. Parse context JSON to extract gender preference
  2. Create gender-specific prompt instructions
  3. Build full prompt with formatting guidelines
  4. Generate response with specified configuration
  5. Return cleaned response text
- **Dependencies**: google.generativeai, json

### Class: FashionAI
**Purpose**: Main AI orchestrator combining RAG and generation components

#### Method: `__init__()`
- **Purpose**: Initialize the fashion AI system
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Initialize FashionRAG instance
  2. Initialize LocalGenerator instance
  3. Initialize GeminiGenerator instance
- **Dependencies**: All above classes

#### Method: `generate_response(user_message, context="", timeout=5)`
- **Purpose**: Generate comprehensive fashion advice response
- **Input Parameters**:
  - `user_message`: str - User's input message
  - `context`: str - Session context (JSON string)
  - `timeout`: int - Timeout in seconds (unused)
- **Output**: str - AI response
- **Functionality**:
  1. Parse context for user name and gender
  2. Handle simple greetings with personalization
  3. Retrieve relevant documents using RAG
  4. Generate response using Gemini with RAG context
  5. Personalize response with user name
  6. Fallback to RAG-based response if Gemini fails
  7. Final fallback to generic greeting
- **Dependencies**: json, time

#### Method: `update_context(session_context, user_message)`
- **Purpose**: Extract and update user preferences from messages
- **Input Parameters**:
  - `session_context`: Dict[str, Any] - Current session context
  - `user_message`: str - User's message
- **Output**: Dict[str, Any] - Updated context
- **Functionality**:
  1. Analyze message for gender keywords
  2. Extract style preferences (casual, formal, athletic)
  3. Extract budget preferences
  4. Update context dictionary with extracted information
- **Dependencies**: None

---

## Django Views Operations (`chat/views.py`)

### Function: `chat_view(request)`
- **Purpose**: Render the main chat interface
- **Input Parameters**:
  - `request`: HttpRequest - Django request object
- **Output**: HttpResponse - Rendered chat template
- **Functionality**:
  1. Return rendered 'chat/chat.html' template
- **Dependencies**: django.shortcuts.render

### Function: `chat_api(request)`
- **Purpose**: Handle chat API requests and orchestrate AI responses
- **Input Parameters**:
  - `request`: HttpRequest - POST request with JSON body
- **Output**: JsonResponse - API response with AI reply and session data
- **Functionality**:
  1. Parse JSON request body for message and session_id
  2. Get or create ChatSession and UserPreference objects
  3. Build context from session and preferences
  4. Generate AI response using FashionAI
  5. Update session context and save message history
  6. Update user preferences if new information extracted
  7. Return JSON response with AI reply and updated context
- **Dependencies**: django.http.JsonResponse, .models, .ai_engine, json, uuid

### Function: `update_preferences(request)`
- **Purpose**: Update user preferences via API
- **Input Parameters**:
  - `request`: HttpRequest - POST request with preference data
- **Output**: JsonResponse - Success confirmation with updated preferences
- **Functionality**:
  1. Parse JSON request body
  2. Get or create UserPreference object
  3. Update preference fields (name, gender, style, budget, occasions)
  4. Save preferences to database
  5. Return success response with updated preferences
- **Dependencies**: django.http.JsonResponse, .models, json

---

## Database Models (`chat/models.py`)

### Model: FashionItem
**Purpose**: Store detailed information about fashion items
**Fields**:
- `name`: CharField - Item name
- `category`: CharField - Category (tops, bottoms, shoes, etc.)
- `description`: TextField - Detailed description
- `season`: CharField - Applicable seasons
- `occasion`: CharField - Suitable occasions
- `gender`: CharField - Target gender
- `price_range`: CharField - Price category
- `brands`: JSONField - List of recommended brands
- `colors`: JSONField - List of available colors
- `created_at`: DateTimeField - Creation timestamp

### Model: Trend
**Purpose**: Store current fashion trends and seasonal information
**Fields**:
- `title`: CharField - Trend title
- `description`: TextField - Trend description
- `season`: CharField - Applicable season
- `year`: IntegerField - Trend year
- `categories`: JSONField - Affected categories
- `created_at`: DateTimeField - Creation timestamp

### Model: UserPreference
**Purpose**: Store user profile and fashion preferences
**Fields**:
- `session_id`: CharField (unique) - Session identifier
- `name`: CharField - User's name
- `gender`: CharField - User's gender
- `style_preferences`: JSONField - List of preferred styles
- `color_preferences`: JSONField - List of preferred colors
- `budget`: CharField - Budget category
- `occasions`: JSONField - Preferred occasions
- `created_at/updated_at`: DateTimeField - Timestamps

### Model: ChatSession
**Purpose**: Manage conversation sessions and message history
**Fields**:
- `session_id`: CharField (unique) - Session identifier
- `messages`: JSONField - List of message dictionaries
- `context`: JSONField - Conversation context data
- `created_at/updated_at`: DateTimeField - Timestamps

---

## Data Population Operations (`populate_data.py`)

### Function: `populate_fashion_data()`
- **Purpose**: Populate database with initial fashion data
- **Input Parameters**: None
- **Output**: None
- **Functionality**:
  1. Clear existing FashionItem and Trend data
  2. Define fashion items data structure with all fields
  3. Define trends data structure
  4. Create FashionItem objects from data
  5. Create Trend objects from data
  6. Print success message with counts
- **Dependencies**: django, .models

---

## Backend Processing Pipeline

### User Input Processing Flow

1. **Browser Request**:
   - User sends message via JavaScript POST to `/api/chat/`
   - Request includes: message, session_id (optional)

2. **Django View Processing** (`chat_api`):
   - Parse JSON request body
   - Validate message presence
   - Get/create ChatSession and UserPreference objects
   - Build context dictionary from session and preferences
   - Call `ai_engine.generate_response()`

3. **AI Engine Processing** (`FashionAI.generate_response`):
   - Parse context JSON for user info
   - Check for greeting patterns
   - Retrieve relevant documents via RAG
   - Generate response using Gemini AI
   - Personalize response with user name
   - Update session context

4. **Response Formation**:
   - Save message to session history
   - Update user preferences if new info extracted
   - Return JSON response with AI reply and context

### Key Processing Components:

- **Context Building**: Combines user preferences, conversation history, and session data
- **RAG Retrieval**: Uses TF-IDF/SVD to find relevant fashion knowledge
- **AI Generation**: Leverages Gemini for personalized fashion advice
- **Preference Learning**: Extracts and stores user preferences from conversations
- **Session Management**: Maintains conversation continuity across requests

### Data Flow:
```
User Input → Django View → AI Engine → RAG Retrieval → Gemini Generation → Response → Database Update
```

---

## Dependencies Summary

### Core Dependencies:
- **Django**: Web framework and ORM
- **Google Generative AI**: Primary AI engine
- **Sentence Transformers**: Text embeddings (alternative)
- **FAISS**: Vector similarity search (alternative)
- **Scikit-learn**: TF-IDF and similarity calculations
- **NLTK**: Natural language processing
- **Gensim**: Word2Vec embeddings
- **Transformers**: Local model support
- **PyTorch**: Machine learning framework
- **Pandas**: Data processing for CSV loading

### Development Dependencies:
- **Django REST Framework**: API development
- **Python-dotenv**: Environment variable management

---

## Configuration Requirements

### Environment Variables:
- `GEMINI_API_KEY`: Google Gemini API key (required)
- `GEMINI_MODEL`: Gemini model version (default: gemini-2.0-flash)

### Django Settings:
- Database configuration
- Static files settings
- Allowed hosts for production
- Debug mode settings

---

## Error Handling

The system implements comprehensive error handling:
- API key validation for Gemini
- Fallback mechanisms (RAG → Local → Generic responses)
- Database operation error catching
- NLTK resource download error handling
- Embedding model loading fallbacks

---

## Performance Considerations

- **Lazy Loading**: Embedding models loaded on first use
- **Caching**: Session and preference data cached in database
- **Fallback Strategy**: Multiple generation methods for reliability
- **Similarity Thresholds**: Filter irrelevant RAG results
- **Context Limiting**: Restrict conversation history to recent messages

---