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




def parse_subband_list(parset_subband_list):
    r'''
    Parse a subband list from a parset or SAS/MAC / MoM spec.

    **Parameters**

    parset_subband_list : string
        Value of Observation.Beam[0].subbandList

    **Returns**

    A list of integers containing the subband numbers.

    **Raises**

    ValueError
        If a syntax problem is encountered.

    **Examples**

    >>> sb_spec = '[154..163,185..194,215..224,245..254,275..284,10*374]'
    >>> parse_subband_list(sb_spec)
    [154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 374, 374, 374, 374, 374, 374, 374, 374, 374, 374]
    >>> sb_spec = '[77..87,116..127,155..166,194..205,233..243,272..282]'
    >>> parse_subband_list(sb_spec)
    [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282]
    >>> parse_subband_list('[]')
    []
    >>> parse_subband_list('1,2,10..15,200..202,400')
    [1, 2, 10, 11, 12, 13, 14, 15, 200, 201, 202, 400]
    '''
    stripped_subband_list = parset_subband_list.strip('[] \n\t')
    if stripped_subband_list == '':
        return []
    sub_lists = [word.strip().split('..')
                 for word in stripped_subband_list.split(',')]
    subbands = []
    for sub_list in sub_lists:
        if len(sub_list) == 1:
            multiplication = sub_list[0].split('*')
            if len(multiplication) == 2:
                subbands += [int(multiplication[1])]*int(multiplication[0])
            else:
                subbands.append(int(sub_list[0]))
        elif len(sub_list) == 2:
            subbands += range(int(sub_list[0]), int(sub_list[1])+1)
        else:
            raise ValueError('%r is not a valid sub_range in a subband list' %
                             sub_list)
    return subbands




