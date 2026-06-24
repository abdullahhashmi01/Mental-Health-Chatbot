"""
src/exception.py
-----------------
Custom exception class that wraps the original error with
file name and line number info, for clearer debugging/logging
across the whole pipeline (industry-standard pattern).
"""

import sys


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return (
        f"Error occurred in script [{file_name}] "
        f"at line number [{line_number}] "
        f"with message: [{str(error)}]"
    )


class ChatbotException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
