from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def divide_numbers(a, b):
    try:
        result =  a / b
        logger.info("dividing to numbers")
        return result
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise CustomException("An error occurred",sys)
    
if __name__ == "__main__":
    try:
        logger.info("Starting the division operation")
        result = divide_numbers(10, 0)
        logger.info(f"Result: {result}")
    except CustomException as ce:
        logger.error(f"CustomException: {ce}")
