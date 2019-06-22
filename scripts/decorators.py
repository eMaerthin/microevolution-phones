from functools import wraps
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
        @wraps(func)
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
