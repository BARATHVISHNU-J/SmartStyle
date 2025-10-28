# SmartStyle - AI Fashion Advisor Chatbot

SmartStyle is an intelligent fashion advisor chatbot built with Django that provides personalized fashion recommendations using AI. The application combines Google's Gemini AI with a Retrieval-Augmented Generation (RAG) system to deliver context-aware fashion advice based on user preferences, trends, and fashion knowledge.

## 🌟 Features

- **AI-Powered Fashion Advice**: Leverages Google's Gemini AI for intelligent fashion recommendations
- **Personalized Recommendations**: Tracks user preferences (gender, style, budget, occasions) for tailored suggestions
- **Fashion Knowledge Base**: Built-in RAG system with fashion trends, outfit combinations, and brand information
- **Session Management**: Maintains conversation context across chat sessions
- **Responsive Web Interface**: Modern, clean UI with dark/light theme support and improved contrast
- **Enhanced Logging**: Detailed backend processing logs with clear separators for better debugging
- **User Safety Disclaimer**: Prominent warning about AI limitations and the need to verify fashion advice
- **REST API**: Full API support for chat interactions and preference management
- **Database Models**: Comprehensive fashion item and trend data storage

## 🛠 Tech Stack

### Backend
- **Django 4.2.7**: Web framework
- **Django REST Framework**: API development
- **Google Gemini AI**: Primary AI engine for fashion advice
- **Sentence Transformers**: For semantic search in RAG system
- **FAISS**: Vector similarity search for fashion knowledge retrieval
- **SQLite**: Database (easily configurable for PostgreSQL/MySQL)

### Frontend
- **HTML5/CSS3**: Responsive web interface
- **JavaScript**: Dynamic chat functionality
- **Font Awesome**: Icons and UI elements

### Key Dependencies
- `google-generativeai`: Gemini AI integration
- `sentence-transformers`: Text embeddings for RAG
- `faiss-cpu`: Vector database for similarity search
- `transformers`: NLP processing
- `torch`: Machine learning framework
- `python-dotenv`: Environment variable management

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git
- Google Gemini API key

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd smartstyle
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 5. Database Setup
```bash
python manage.py migrate
```

### 6. Populate Initial Data
```bash
python populate_data.py
```

### 7. Run the Development Server
```bash
python manage.py runserver
```

## 📖 Usage

### Web Interface
1. Open the application in your browser
2. Start chatting with SmartStyle about fashion preferences
3. The AI will learn your style preferences over time
4. Toggle between light/dark themes using the moon/sun icon
5. **Important**: Always verify AI fashion advice before use (see disclaimer at bottom)

### Example Conversations
- "What should I wear to a business meeting?"
- "Suggest casual outfits for summer"
- "I'm looking for formal wear on a budget"
- "What colors go well with navy blue?"

## 🗄 Database Models

### FashionItem
Stores fashion items with detailed attributes:
- Name, category, description
- Season, occasion, gender
- Price range, brands, colors

### Trend
Current fashion trends and seasonal information:
- Title, description, season, year
- Affected categories

### UserPreference
User profile and preferences:
- Session ID, name, gender
- Style preferences, color preferences
- Budget range, preferred occasions

### ChatSession
Conversation management:
- Session ID, messages history
- Context data, timestamps

## 🔧 API Documentation

### Chat API
**Endpoint**: `POST /api/chat/`

Send chat messages and receive AI responses.

**Request Body**:
```json
{
  "message": "What should I wear to a wedding?",
  "session_id": "optional-session-id"
}
```

**Response**:
```json
{
  "response": "For a wedding, I recommend...",
  "session_id": "session-uuid",
  "context": {
    "gender": "female",
    "style": "formal",
    "budget": "mid-range"
  }
}
```

### Update Preferences API
**Endpoint**: `POST /api/preferences/`

Update user preferences for better recommendations.

**Request Body**:
```json
{
  "session_id": "required-session-id",
  "name": "John Doe",
  "gender": "male",
  "style": ["casual", "business"],
  "budget": "mid-range",
  "occasions": ["work", "casual"]
}
```

## ⚙ Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `GEMINI_MODEL`: Gemini model version (default: gemini-2.0-flash)

### Django Settings
Key settings in `smartstyle/settings.py`:
- `DEBUG`: Set to `False` in production
- `ALLOWED_HOSTS`: Configure for production deployment
- `DATABASES`: Configure database connection
- `STATIC_ROOT`: Static files directory for production

## 🏗 Project Structure

```
smartstyle/
├── chat/                    # Main Django app
│   ├── ai_engine.py        # AI and RAG implementation
│   ├── models.py           # Database models
│   ├── views.py            # API endpoints and views
│   ├── urls.py             # URL routing
│   ├── templates/chat/     # HTML templates
│   ├── static/chat/        # CSS, JS, images
│   ├── debug_logger.py     # Backend processing logger
│   └── migrations/         # Database migrations
├── smartstyle/             # Django project settings
├── models/                 # Pre-trained ML models
├── staticfiles/            # Collected static files
├── backend_logs/           # Backend processing logs directory
├── db.sqlite3              # SQLite database
├── manage.py               # Django management script
├── populate_data.py        # Data seeding script
├── backend_logs_viewer.py  # Logs analysis and viewing utility
├── PROJECT_OPERATIONS.md   # Comprehensive technical documentation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

### Development Guidelines
- Follow Django best practices
- Write clear, documented code
- Test API endpoints thoroughly
- Update documentation for new features
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for powering the fashion recommendations
- Sentence Transformers library for semantic search
- Django community for the excellent web framework
- Font Awesome for UI icons

## ⚠️ Important Notice

**AI Limitations**: SmartStyle uses advanced AI technology to provide fashion recommendations, but AI systems can make mistakes or provide outdated information. Always verify fashion advice with trusted sources and consider your personal circumstances before making purchasing decisions.

## 📞 Support

For questions, issues, or contributions, please:
1. Check existing issues on GitHub
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Happy styling with SmartStyle! 👗👔**
