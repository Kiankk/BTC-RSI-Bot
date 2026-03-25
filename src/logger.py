import logging
import sys

def setup_logger(name="QuantEngine"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers (Console and File)
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler('trading_engine.log')
    
    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger
