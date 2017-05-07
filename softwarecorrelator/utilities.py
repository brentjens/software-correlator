import os

class working_dir(object):
    '''
    Context manager for use in a "with" statement. Executes body of with
    statement in the specified sub directory, returning to the current
    working directory upon exit.
    
    **Example**
    
    >>> initial_dir = os.getcwd()
    >>> os.path.relpath(start=initial_dir, path=os.getcwd())
    '.'
    >>> with working_dir('testdata/'):
    ...     os.path.relpath(start=initial_dir, path=os.getcwd())
    'testdata'
    >>> os.path.relpath(start=initial_dir, path=os.getcwd())
    '.'
    '''
    def __init__(self, dir_name):
        self.working_dir_name = dir_name
        
    def __enter__(self):
        self.current_working_dir_name = os.getcwd()
        os.chdir(self.working_dir_name)
        
    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.current_working_dir_name)

