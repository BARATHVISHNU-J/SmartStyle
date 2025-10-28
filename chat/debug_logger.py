import json
import logging
import os
import atexit
from datetime import datetime, timedelta
from typing import Dict, Any, List
from django.conf import settings

class BackendProcessingLogger:
    """Logs detailed backend processing steps for user inputs"""

    def __init__(self):
        self.log_directory = os.path.join(settings.BASE_DIR, 'backend_logs')
        os.makedirs(self.log_directory, exist_ok=True)

        # Single file for all sessions
        self.sessions_file = os.path.join(self.log_directory, 'all_sessions.jsonl')

        # Setup logging
        self.logger = logging.getLogger('backend_processing')
        self.logger.setLevel(logging.DEBUG)

        # File handler for detailed logs
        log_file = os.path.join(self.log_directory, f'processing_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Register cleanup on app shutdown
        atexit.register(self._cleanup_on_shutdown)

    def _cleanup_on_shutdown(self):
        """Clean up all log files when the web app shuts down"""
        try:
            # Remove all individual session files
            for filename in os.listdir(self.log_directory):
                if filename.startswith('session_') and filename.endswith('.jsonl'):
                    file_path = os.path.join(self.log_directory, filename)
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up session log: {filename}")
                    except Exception as e:
                        print(f"Failed to remove {filename}: {e}")

            # Clear contents of ALL processing log files on shutdown (don't delete the file)
            for filename in os.listdir(self.log_directory):
                if filename.startswith('processing_') and filename.endswith('.log'):
                    file_path = os.path.join(self.log_directory, filename)
                    try:
                        # Clear the file contents instead of deleting the file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write('')  # Write empty string to clear contents
                        print(f"Cleared contents of processing log: {filename}")
                    except Exception as e:
                        print(f"Failed to clear contents of {filename}: {e}")

            print("Log cleanup completed on app shutdown")

        except Exception as e:
            print(f"Error during log cleanup: {e}")

    def log_processing_step(self, session_id: str, step_name: str, data: Dict[str, Any]):
        """Log a specific processing step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'step': step_name,
            'data': data
        }

        # Add separator line for better log readability
        separator = "=" * 80
        self.logger.info(separator)
        self.logger.info(f"STEP: {step_name} - SESSION: {session_id}")
        self.logger.info(separator)
        self.logger.debug(f"DATA: {json.dumps(log_entry, indent=2)}")

        # Also save to individual session file
        self._save_session_log(session_id, log_entry)

    def _save_session_log(self, session_id: str, log_entry: Dict[str, Any]):
        """Save log entry to single sessions file"""
        # Save to single file for all sessions
        with open(self.sessions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_input_tokenization(self, session_id: str, user_message: str):
        """Log input tokenization process"""
        import nltk
        from nltk.tokenize import word_tokenize, sent_tokenize

        try:
            nltk.download('punkt', quiet=True)

            tokens = word_tokenize(user_message.lower())
            sentences = sent_tokenize(user_message)

            tokenization_data = {
                'original_message': user_message,
                'word_tokens': tokens,
                'sentence_tokens': sentences,
                'token_count': len(tokens),
                'sentence_count': len(sentences),
                'character_count': len(user_message)
            }

            self.log_processing_step(session_id, 'INPUT_TOKENIZATION', tokenization_data)

        except Exception as e:
            self.logger.error(f"Tokenization error: {e}")
            tokenization_data = {
                'original_message': user_message,
                'error': str(e),
                'basic_split': user_message.split()
            }
            self.log_processing_step(session_id, 'INPUT_TOKENIZATION_ERROR', tokenization_data)

    def log_context_building(self, session_id: str, context_data: Dict[str, Any]):
        """Log context building process"""
        context_summary = {
            'user_name': context_data.get('name', ''),
            'user_gender': context_data.get('gender', ''),
            'style_preferences': context_data.get('style', []),
            'budget': context_data.get('budget', ''),
            'occasions': context_data.get('occasions', []),
            'conversation_history_length': len(context_data.get('conversation_history', [])),
            'context_keys': list(context_data.keys())
        }

        self.log_processing_step(session_id, 'CONTEXT_BUILDING', context_summary)

    def log_rag_retrieval(self, session_id: str, query: str, retrieved_docs: List[str], rag_metadata: Dict[str, Any]):
        """Log RAG retrieval process"""
        rag_data = {
            'query': query,
            'retrieved_docs_count': len(retrieved_docs),
            'retrieved_docs_preview': [doc[:200] + '...' if len(doc) > 200 else doc for doc in retrieved_docs],
            'rag_method': rag_metadata.get('method', 'unknown'),
            'similarity_threshold': rag_metadata.get('threshold', 0.0),
            'processing_time': rag_metadata.get('processing_time', 0.0)
        }

        self.log_processing_step(session_id, 'RAG_RETRIEVAL', rag_data)

    def log_ai_generation(self, session_id: str, prompt: str, context: str, response: str, generation_metadata: Dict[str, Any]):
        """Log AI generation process"""
        generation_data = {
            'prompt_length': len(prompt),
            'context_length': len(context),
            'response_length': len(response),
            'model_used': generation_metadata.get('model', 'unknown'),
            'generation_time': generation_metadata.get('generation_time', 0.0),
            'tokens_used': generation_metadata.get('tokens_used', 0),
            'gender_instruction': generation_metadata.get('gender_instruction', ''),
            'response_preview': response[:300] + '...' if len(response) > 300 else response
        }

        self.log_processing_step(session_id, 'AI_GENERATION', generation_data)

    def log_preference_extraction(self, session_id: str, user_message: str, extracted_prefs: Dict[str, Any]):
        """Log preference extraction process"""
        extraction_data = {
            'user_message': user_message,
            'extracted_gender': extracted_prefs.get('gender', ''),
            'extracted_style': extracted_prefs.get('style', ''),
            'extracted_budget': extracted_prefs.get('budget', ''),
            'extraction_method': 'keyword_matching',
            'confidence_score': extracted_prefs.get('confidence', 0.0)
        }

        self.log_processing_step(session_id, 'PREFERENCE_EXTRACTION', extraction_data)

    def log_response_formation(self, session_id: str, final_response: str, processing_summary: Dict[str, Any]):
        """Log final response formation"""
        response_data = {
            'response_length': len(final_response),
            'response_preview': final_response[:300] + '...' if len(final_response) > 300 else final_response,
            'total_processing_time': processing_summary.get('total_time', 0.0),
            'steps_completed': processing_summary.get('steps', []),
            'fallback_used': processing_summary.get('fallback', False),
            'personalization_applied': processing_summary.get('personalized', False)
        }

        self.log_processing_step(session_id, 'RESPONSE_FORMATION', response_data)

    def log_error(self, session_id: str, error_stage: str, error_message: str, error_data: Dict[str, Any] = None):
        """Log processing errors"""
        error_log = {
            'stage': error_stage,
            'error_message': error_message,
            'error_data': error_data or {},
            'timestamp': datetime.now().isoformat()
        }

        self.logger.error(f"ERROR in {error_stage}: {error_message}")
        self.log_processing_step(session_id, 'ERROR', error_log)

    def get_session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all logs for a specific session from the single sessions file"""
        if not os.path.exists(self.sessions_file):
            return []

        logs = []
        with open(self.sessions_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    if log_entry.get('session_id') == session_id:
                        logs.append(log_entry)
                except json.JSONDecodeError:
                    continue

        return logs

    def get_processing_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of processing steps for a session"""
        logs = self.get_session_logs(session_id)

        if not logs:
            return {'error': 'No logs found for session'}

        summary = {
            'session_id': session_id,
            'total_steps': len(logs),
            'steps_breakdown': {},
            'processing_times': {},
            'errors': [],
            'start_time': logs[0]['timestamp'] if logs else None,
            'end_time': logs[-1]['timestamp'] if logs else None
        }

        for log in logs:
            step = log['step']
            summary['steps_breakdown'][step] = summary['steps_breakdown'].get(step, 0) + 1

            if step == 'ERROR':
                summary['errors'].append(log['data'])

            if 'processing_time' in log['data']:
                summary['processing_times'][step] = log['data']['processing_time']

        return summary

# Global logger instance
backend_logger = BackendProcessingLogger()
