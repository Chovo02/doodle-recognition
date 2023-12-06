from loguru import logger
import pendulum
import os
import sys


class Log:
    def __init__(self, save_file: bool = True, dir_path: str = "logs"):
        self.save_file = save_file
        self.dir_path = dir_path
        self.__level_list = [
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]
        self.__date = pendulum.now()
        self.__add_log_file(level="DEBUG")
        logger.remove(0)
        logger.add(sys.stdout, level="SUCCESS")

    def __add_log_file(self, level: str = "INFO"):
        if self.save_file:
            if not os.path.exists(self.dir_path):
                os.mkdir(self.dir_path)
            file_name = f"{self.__date.year}-{self.__date.month}-{self.__date.day}-{self.__date.hour}-{self.__date.minute}-{self.__date.second}.log"
            logger.add(
                f"{self.dir_path}/{file_name}",
                format="{time} {level} {message}",
                level=level,
            )

    def write_log(self, message: str, level: str = "INFO"):
        if level not in self.__level_list:
            raise ValueError(
                f"Invalid level: {level}. Level must be one of the following values: {self.__level_list}"
            )
        logger.log(level, message)
