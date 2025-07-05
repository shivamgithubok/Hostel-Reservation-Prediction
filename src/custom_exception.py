import traceback # for debugging purposes
import sys
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        #we need to inheret from our exception class
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message,error_detail)

# for not making custom exception again and again will use @staticmethod 
@staticmethod
def get_detailed_error_message(error_message, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in script: [{file_name}] at line number: [{line_number}] with error message: [{error_message}]"

def __str__(self):
    return self.error_message