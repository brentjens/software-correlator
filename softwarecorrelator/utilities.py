import os

class working_dir(object):
    '''
    Context manager for use in a "with" statement. Executes body of with
    statement in the specified sub directory, returning to the current
    working directory upon exit.
    
    **Example**
    
    >>> def hdf5_file_names(dir_name):
    >>>     with working_dir(dir_name):
    >>>         hdf5_file_names = glob.glob('*.h5')
    >>>
    >>> hdf5_file_names('testdata/L590569-LBA_OUTER-20170501/')
    >>>
    '''
    def __init__(self, dir_name):
        self.working_dir_name = dir_name
        
    def __enter__(self):
        self.current_working_dir_name = os.getcwd()
        os.chdir(self.working_dir_name)
        
    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.current_working_dir_name)

