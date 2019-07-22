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


class CheckIfDone:
    def __init__(self, function, check_path, validator=None,
                 ignore_done=False):
        self.function = function
        self.check_path = check_path
        self.validator = validator
        self.ignore_done = ignore_done
        functools.update_wrapper(self, function)

    def done_path(self):
        return f'{self.check_path}.done'

    def md5_signature(self):
        hash_md5 = hashlib.md5()
        with open(self.check_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def __call__(self, *args, **kwargs):
        skip = False
        if isfile(self.done_path()) and not self.ignore_done:
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
        if self.validator is None or self.validator(ret_val):
            if self.ignore_done and isfile(self.done_path()):

            with open(self.done_path(), 'w') as f:
                f.write(self.md5_signature())
        return ret_val

"""
value = None
            if not isfile(done(check_path)) or ignore_already_done:
                value = func(*args, **kwargs)
                task_done = True
                if ret_value_validator:
                    task_done = ret_value_validator(value)
                if task_done:
                    with open(done(check_path), 'w') as f:
                        f.write('OK')
            logger.info(f'skipping evaluating this method'
                        f' because {done(check_path)} already exists'
                        ' - hence ignoring return value!')
            return value
            """

def check_if_already_done(check_path,
                          ret_value_validator=None,
                          ignore_already_done=False):
    """
        Useful decorator to speed up evaluation of the chain.
        Main purpose is to not calculate values of the same
        nodes of the chain more than once.
        :param check_path: path to be checked if
        we already know the result of the method
        :param ret_value_validator: validates if
        function terminates properly - translates
        outcome of the function to either True or False.
        :param ignore_already_done: if set, will
        run the decorated method even if it was already
        done

    TODO:
    1. decorator does not need an extra parameter -
    they can read the result_path from *args, also all check_if_already_done decorated methods
    can be converted into members of a class that has automatically inserted check_if_already_
    done + require result_path param
    2. those decorated methods should have better implementation of the what-if-already-done
    meaning that they should already return required value if any

    :return:
    """
    def done(path):
        return f'{path}.done'

    def decorator_if_already_done(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = None
            if not isfile(done(check_path)) or ignore_already_done:
                value = func(*args, **kwargs)
                task_done = True
                if ret_value_validator:
                    task_done = ret_value_validator(value)
                if task_done:
                    with open(done(check_path), 'w') as f:
                        f.write('OK')
            logger.info(f'skipping evaluating this method'
                        f' because {done(check_path)} already exists'
                        ' - hence ignoring return value!')
            return value

        return wrapper

    return decorator_if_already_done
