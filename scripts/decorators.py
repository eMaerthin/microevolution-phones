from functools import wraps

from os.path import isfile

def done(path):
    return f'{path}.done'

def check_if_already_done(check_path, verbose=0):
    def decorator_if_already_done(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            value = None
            if not isfile(done(check_path)):
                value = func(*args, **kwargs)
                with open(done(check_path), 'w') as f:
                    f.write('OK')
            elif verbose > 0:
                print(f'skipping evaluating this method because \
                      {done(check_path)} already exists')
                print(f'Hence ignoring return value!')
            return value
        return wrapper
    return decorator_if_already_done
