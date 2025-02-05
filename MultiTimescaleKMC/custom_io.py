import os
import pickle
from smol.moca import CompositeProcessor

class Custom_IO:
    
    """
    This class only contains static methods for I/O operations. 
    """
    
    @staticmethod
    def Lines_from_file(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        return lines

    @staticmethod
    def Write_lines_to_file(lines, filename):
        with open(filename, 'w') as file:
            file.writelines( lines )

    @staticmethod
    def load_pickle(filename, directory=os.getcwd()):
        pickle_file = os.path.join(directory, filename)
        with open(pickle_file, 'rb') as handle:
            return pickle.load(handle)
        
    @staticmethod        
    def write_pickle(contents, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_processor(Processor_filename):
        with open(Processor_filename, 'rb') as handle:
            return CompositeProcessor.from_dict(pickle.load(handle))
            
    @staticmethod        
    def write_step_file(s, step_file_name):
        with open(step_file_name, "w") as file:
            file.write(str(s))
            
    @staticmethod
    def New_Directory_Maker(Existing_Directory_Name, New_Sub_Directory_Name):

        new_directory = os.path.join(Existing_Directory_Name, New_Sub_Directory_Name)
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)

        return new_directory