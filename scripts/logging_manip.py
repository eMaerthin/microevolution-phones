import logging
from os.path import (abspath, basename, join)


def change_file_logger_path(new_base_dir):
    logger = logging.getLogger()

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            rel_filename = join(new_base_dir,
                                basename(handler.baseFilename))
            new_logger_base_filename = abspath(rel_filename)
            handler.close()
            handler.baseFilename = new_logger_base_filename
