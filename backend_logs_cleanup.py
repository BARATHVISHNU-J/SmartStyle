#!/usr/bin/env python3
"""
Backend Logs Cleanup Script
Automatically cleans up old log files to conserve disk space.
"""

import os
import time
import glob
from datetime import datetime, timedelta

def cleanup_old_logs(logs_dir="backend_logs", days_to_keep=7):
    """
    Remove log files older than specified days.

    Args:
        logs_dir (str): Directory containing log files
        days_to_keep (int): Number of days of logs to keep
    """
    if not os.path.exists(logs_dir):
        print(f"Logs directory {logs_dir} does not exist.")
        return

    # Calculate cutoff time
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    files_removed = 0
    space_saved = 0

    # Find all log files
    log_patterns = [
        os.path.join(logs_dir, "*.log"),
        os.path.join(logs_dir, "*.jsonl"),
        os.path.join(logs_dir, "session_*.jsonl")
    ]

    for pattern in log_patterns:
        for log_file in glob.glob(pattern):
            try:
                # Check file modification time
                file_mtime = os.path.getmtime(log_file)

                if file_mtime < cutoff_time:
                    # Get file size before deletion
                    file_size = os.path.getsize(log_file)

                    # Remove the file
                    os.remove(log_file)
                    files_removed += 1
                    space_saved += file_size

                    print(f"Removed: {log_file}")

            except OSError as e:
                print(f"Error removing {log_file}: {e}")

    # Print summary
    if files_removed > 0:
        space_mb = space_saved / (1024 * 1024)
        print(f"\nCleanup completed:")
        print(f"- Files removed: {files_removed}")
        print(f"- Space saved: {space_mb:.2f} MB")
    else:
        print("No old log files found to clean up.")

def cleanup_on_startup(logs_dir="backend_logs"):
    """
    Clean up logs when the application starts.
    This is called automatically when the Django server starts.
    """
    print("Running automatic log cleanup...")
    cleanup_old_logs(logs_dir)

if __name__ == "__main__":
    # Run cleanup when script is executed directly
    cleanup_old_logs()
