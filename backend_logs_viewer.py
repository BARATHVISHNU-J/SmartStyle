#!/usr/bin/env python
"""
Backend Processing Logs Viewer
A utility script to view and analyze backend processing logs for the SmartStyle chatbot.
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Add Django project to path
sys.path.append(os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartstyle.settings')

import django
django.setup()

from chat.debug_logger import backend_logger

class BackendLogsViewer:
    """Viewer for backend processing logs"""

    def __init__(self, logs_directory: str = 'backend_logs'):
        self.logs_directory = logs_directory

    def list_sessions(self) -> List[str]:
        """List all session IDs that have logs"""
        if not os.path.exists(self.logs_directory):
            return []

        sessions = []
        for filename in os.listdir(self.logs_directory):
            if filename.startswith('session_') and filename.endswith('.jsonl'):
                session_id = filename.replace('session_', '').replace('.jsonl', '')
                sessions.append(session_id)

        return sorted(sessions)

    def get_session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a specific session"""
        return backend_logger.get_session_logs(session_id)

    def get_processing_summary(self, session_id: str) -> Dict[str, Any]:
        """Get processing summary for a session"""
        return backend_logger.get_processing_summary(session_id)

    def display_session_summary(self, session_id: str):
        """Display a formatted summary of session processing"""
        summary = self.get_processing_summary(session_id)

        if 'error' in summary:
            print(f"No logs found for session {session_id}")
            return

        print(f"\n{'='*60}")
        print(f"SESSION PROCESSING SUMMARY: {session_id}")
        print(f"{'='*60}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Start Time: {summary['start_time']}")
        print(f"End Time: {summary['end_time']}")
        print(f"Errors: {len(summary['errors'])}")

        print(f"\nStep Breakdown:")
        for step, count in summary['steps_breakdown'].items():
            print(f"  {step}: {count}")

        if summary['processing_times']:
            print(f"\nProcessing Times:")
            for step, time_taken in summary['processing_times'].items():
                print(".3f")

        if summary['errors']:
            print(f"\nErrors:")
            for error in summary['errors']:
                print(f"  - {error['stage']}: {error['error_message']}")

    def display_detailed_logs(self, session_id: str, step_filter: str = None):
        """Display detailed logs for a session"""
        logs = self.get_session_logs(session_id)

        if not logs:
            print(f"No logs found for session {session_id}")
            return

        print(f"\n{'='*80}")
        print(f"DETAILED LOGS FOR SESSION: {session_id}")
        print(f"{'='*80}")

        for log in logs:
            if step_filter and log['step'] != step_filter:
                continue

            print(f"\n[{log['timestamp']}] STEP: {log['step']}")
            print("-" * 40)

            data = log['data']

            if log['step'] == 'INPUT_TOKENIZATION':
                print(f"Original Message: {data['original_message']}")
                print(f"Word Tokens: {data['word_tokens']}")
                print(f"Sentence Tokens: {data['sentence_tokens']}")
                print(f"Token Count: {data['token_count']}")
                print(f"Character Count: {data['character_count']}")

            elif log['step'] == 'CONTEXT_BUILDING':
                print(f"User Name: {data['user_name']}")
                print(f"User Gender: {data['user_gender']}")
                print(f"Style Preferences: {data['style_preferences']}")
                print(f"Budget: {data['budget']}")
                print(f"Occasions: {data['occasions']}")
                print(f"Conversation History Length: {data['conversation_history_length']}")

            elif log['step'] == 'RAG_RETRIEVAL':
                print(f"Query: {data['query']}")
                print(f"Retrieved Docs Count: {data['retrieved_docs_count']}")
                print(f"RAG Method: {data['rag_method']}")
                print(".3f")
                print("Retrieved Documents Preview:")
                for i, doc in enumerate(data['retrieved_docs_preview'], 1):
                    print(f"  {i}. {doc}")

            elif log['step'] == 'AI_GENERATION':
                print(f"Prompt Length: {data['prompt_length']}")
                print(f"Context Length: {data['context_length']}")
                print(f"Response Length: {data['response_length']}")
                print(f"Model Used: {data['model_used']}")
                print(".3f")
                print(f"Tokens Used: {data['tokens_used']}")
                print(f"Gender Instruction: {data['gender_instruction']}")
                print(f"Response Preview: {data['response_preview']}")

            elif log['step'] == 'PREFERENCE_EXTRACTION':
                print(f"User Message: {data['user_message']}")
                print(f"Extracted Gender: {data['extracted_gender']}")
                print(f"Extracted Style: {data['extracted_style']}")
                print(f"Extracted Budget: {data['extracted_budget']}")
                print(f"Extraction Method: {data['extraction_method']}")

            elif log['step'] == 'RESPONSE_FORMATION':
                print(f"Response Length: {data['response_length']}")
                print(".3f")
                print(f"Steps Completed: {data['steps_completed']}")
                print(f"Fallback Used: {data['fallback_used']}")
                print(f"Personalization Applied: {data['personalization_applied']}")
                print(f"Response Preview: {data['response_preview']}")

            elif log['step'] == 'ERROR':
                print(f"Error Stage: {data['stage']}")
                print(f"Error Message: {data['error_message']}")

    def export_session_logs(self, session_id: str, output_file: str = None):
        """Export session logs to a JSON file"""
        logs = self.get_session_logs(session_id)

        if not logs:
            print(f"No logs found for session {session_id}")
            return

        if output_file is None:
            output_file = f'session_{session_id}_logs.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': session_id,
                'exported_at': datetime.now().isoformat(),
                'logs': logs
            }, f, indent=2, ensure_ascii=False)

        print(f"Logs exported to {output_file}")

    def analyze_all_sessions(self):
        """Analyze all sessions and provide aggregate statistics"""
        sessions = self.list_sessions()

        if not sessions:
            print("No sessions found.")
            return

        total_sessions = len(sessions)
        total_steps = 0
        step_counts = defaultdict(int)
        error_sessions = 0
        avg_processing_times = defaultdict(list)

        for session_id in sessions:
            summary = self.get_processing_summary(session_id)

            if 'error' not in summary:
                total_steps += summary['total_steps']
                error_sessions += len(summary['errors'])

                for step, count in summary['steps_breakdown'].items():
                    step_counts[step] += count

                for step, time_taken in summary['processing_times'].items():
                    avg_processing_times[step].append(time_taken)

        print(f"\n{'='*60}")
        print("AGGREGATE SESSIONS ANALYSIS")
        print(f"{'='*60}")
        print(f"Total Sessions: {total_sessions}")
        print(f"Total Processing Steps: {total_steps}")
        print(".2f")
        print(f"Sessions with Errors: {error_sessions}")

        print(f"\nStep Distribution:")
        for step, count in sorted(step_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_steps) * 100
            print(".1f")

        print(f"\nAverage Processing Times:")
        for step, times in avg_processing_times.items():
            if times:
                avg_time = sum(times) / len(times)
                print(".3f")

def main():
    """Main function for command-line usage"""
    viewer = BackendLogsViewer()

    if len(sys.argv) < 2:
        print("Usage: python backend_logs_viewer.py <command> [args...]")
        print("\nCommands:")
        print("  list                          - List all sessions")
        print("  summary <session_id>          - Show session summary")
        print("  logs <session_id> [step]      - Show detailed logs (optional step filter)")
        print("  export <session_id> [file]    - Export session logs to file")
        print("  analyze                       - Analyze all sessions")
        return

    command = sys.argv[1]

    if command == 'list':
        sessions = viewer.list_sessions()
        if sessions:
            print("Available Sessions:")
            for session in sessions:
                print(f"  {session}")
        else:
            print("No sessions found.")

    elif command == 'summary' and len(sys.argv) >= 3:
        session_id = sys.argv[2]
        viewer.display_session_summary(session_id)

    elif command == 'logs' and len(sys.argv) >= 3:
        session_id = sys.argv[2]
        step_filter = sys.argv[3] if len(sys.argv) > 3 else None
        viewer.display_detailed_logs(session_id, step_filter)

    elif command == 'export' and len(sys.argv) >= 3:
        session_id = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        viewer.export_session_logs(session_id, output_file)

    elif command == 'analyze':
        viewer.analyze_all_sessions()

    else:
        print("Invalid command or missing arguments.")

if __name__ == '__main__':
    main()
