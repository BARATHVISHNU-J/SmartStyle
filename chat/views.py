from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import uuid
import time
from .models import ChatSession, UserPreference
from .ai_engine import FashionAI
from .debug_logger import backend_logger
from backend_logs_cleanup import cleanup_on_startup

# Initialize AI engine
ai_engine = FashionAI()

# Run log cleanup on startup
cleanup_on_startup()

def chat_view(request):
    """Main chat interface"""
    return render(request, 'chat/chat.html')

@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    """API endpoint for chat interactions"""
    start_time = time.time()
    session_id = None

    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))

        if not user_message:
            return JsonResponse({'error': 'Message is required'}, status=400)

        # Log input tokenization
        backend_logger.log_input_tokenization(session_id, user_message)

        # Get or create chat session
        session, created = ChatSession.objects.get_or_create(
            session_id=session_id,
            defaults={'messages': [], 'context': {}}
        )

        # Get user preferences
        preferences, pref_created = UserPreference.objects.get_or_create(
            session_id=session_id,
            defaults={'name': '', 'gender': '', 'style_preferences': [], 'color_preferences': [],
                     'budget': '', 'occasions': []}
        )

        # Build context from session and preferences
        context = {
            'name': preferences.name,
            'gender': preferences.gender,
            'style': preferences.style_preferences,
            'budget': preferences.budget,
            'occasions': preferences.occasions,
            'conversation_history': session.messages[-5:] if session.messages else []  # Last 5 messages
        }

        # Force gender from preferences if available
        if preferences.gender:
            context['gender'] = preferences.gender

        # Log context building
        backend_logger.log_context_building(session_id, context)

        context_str = json.dumps(context)

        # Generate AI response
        ai_response = ai_engine.generate_response(user_message, context_str)

        # Update context with new information
        updated_context = ai_engine.update_context(session.context, user_message)

        # Log preference extraction
        backend_logger.log_preference_extraction(session_id, user_message, updated_context)

        # Save message to session
        new_message = {
            'user': user_message,
            'bot': ai_response,
            'timestamp': str(session.updated_at)
        }
        session.messages.append(new_message)
        session.context = updated_context
        session.save()

        # Update preferences if new info extracted
        if 'gender' in updated_context and not preferences.gender:
            preferences.gender = updated_context['gender']
            preferences.save()

        if 'style' in updated_context and updated_context['style'] not in preferences.style_preferences:
            preferences.style_preferences.append(updated_context['style'])
            preferences.save()

        if 'budget' in updated_context and not preferences.budget:
            preferences.budget = updated_context['budget']
            preferences.save()

        # Log response formation
        total_time = time.time() - start_time
        processing_summary = {
            'total_time': total_time,
            'steps': ['tokenization', 'context_building', 'ai_generation', 'preference_extraction', 'response_formation'],
            'fallback': False,
            'personalized': bool(preferences.name)
        }
        backend_logger.log_response_formation(session_id, ai_response, processing_summary)

        return JsonResponse({
            'response': ai_response,
            'session_id': session_id,
            'context': updated_context
        })

    except Exception as e:
        if session_id:
            backend_logger.log_error(session_id, 'CHAT_API', str(e))
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def update_preferences(request):
    """Update user preferences"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')

        if not session_id:
            return JsonResponse({'error': 'Session ID is required'}, status=400)

        preferences, created = UserPreference.objects.get_or_create(
            session_id=session_id,
            defaults={'gender': '', 'style_preferences': [], 'color_preferences': [],
                     'budget': '', 'occasions': []}
        )

        # Update preferences
        if 'name' in data:
            preferences.name = data['name']
        if 'gender' in data:
            preferences.gender = data['gender']
        if 'style' in data:
            preferences.style_preferences = data['style']
        if 'budget' in data:
            preferences.budget = data['budget']
        if 'occasions' in data:
            preferences.occasions = data['occasions']

        preferences.save()

        return JsonResponse({'success': True, 'preferences': {
            'name': preferences.name,
            'gender': preferences.gender,
            'style_preferences': preferences.style_preferences,
            'budget': preferences.budget,
            'occasions': preferences.occasions
        }})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
