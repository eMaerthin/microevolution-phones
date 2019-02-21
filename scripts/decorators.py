from functools import wraps
from os.path import isfile


def check_if_already_done(check_path, verbose=0, ret_value_validator=None, ignore_already_done=False):
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
            elif verbose > 0:
                print(f'skipping evaluating this method because \
                      {done(check_path)} already exists')
                print(f'Hence ignoring return value!')
            return value

        return wrapper

    return decorator_if_already_done
