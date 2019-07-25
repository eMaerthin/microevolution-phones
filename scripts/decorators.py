import functools
import hashlib
import logging
from os.path import isfile
import time
logger = logging.getLogger()


def timeit(func):
    """
    Useful decorator to measure evaluation time easily
    :param func: function to be measured
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        logger.info(f'Evaluation time of {func.__name__}: {te-ts} sec')
        return result

    return timed


class _CheckIfDone:
    """https://stackoverflow.com/questions/7492068/python-class-decorator-arguments"""
    def __init__(self, function, check_path, validator=None, ignore_done=False):
        self.function = function
        self.check_path = check_path
        self.validator = validator
        self.ignore_done = ignore_done
        functools.update_wrapper(self, function)  # this is class equivalent of @functools.wraps

    def done_path(self):
        return f'{self.check_path}.done'

    def md5_signature(self):
        hash_md5 = hashlib.md5()
        with open(self.check_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def __call__(self, *args, **kwargs):
        if not self.check_path:
            self.check_path = kwargs.get('check_path')
            if not self.check_path:
                logger.warning(f'check_path not provided')

        if self.check_path and isfile(self.done_path()) and not self.ignore_done:
            with open(self.done_path(), 'r') as f:
                reference_md5 = f.read()
            if reference_md5 == self.md5_signature():
                logger.info(f'skipping evaluating this method'
                            f' because {self.done_path()} already exists'
                            ' - hence ignoring return value!')
                return None
            logger.warning(f'invalid md5 hash found from {self.done_path()}'
                           f' the method that is responsible for generating it'
                           f' the result has to be re-run')

        ret_val = self.function(*args, **kwargs)
        if not self.validator or self.validator(ret_val):
            with open(self.done_path(), 'w') as f:
                f.write(self.md5_signature())
        return ret_val


def check_if_already_done(check_path=None, validator=None, ignore_done=False):
    """
    This logic assumes check_path is provided or given in kwargs of the function being wrapped around
    :param check_path: if it is None then wrapper function parameter is expected to have such a keyword param
    :param validator:
    :param ignore_done:
    :return:
    """
    def wrapper(function):
        return _CheckIfDone(function, check_path, validator, ignore_done)
    return wrapper
