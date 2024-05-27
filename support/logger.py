import logging
from logging.handlers import TimedRotatingFileHandler

def init_logger(name):
    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG

    # 创建一个handler，用于每天换一个日志文件
    file_handler = TimedRotatingFileHandler(f'{name}.log', when="midnight", interval=1, backupCount=7)
    file_handler.setLevel(logging.DEBUG)  # 设置handler的日志级别
    file_handler.suffix = "%Y-%m-%d.log"  # 设置文件名后缀，按天进行区分

    # 创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 设置handler的日志级别

    # 创建一个formatter，设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 给handler添加formatter
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger