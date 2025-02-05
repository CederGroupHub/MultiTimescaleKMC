import json
from custom_io import Custom_IO
import numpy as np
from smol.cofe import ClusterExpansion
from smol.moca.processor import EwaldProcessor, CompositeProcessor,ClusterDecompositionProcessor

with open('/global/u2/s/sanando/DRX_to_Delta/FUNCTIONS/wdr_74_3.json') as fin:
    ce = ClusterExpansion.from_dict(json.load(fin))

interact_tensors = ce.cluster_interaction_tensors

interact_tensors[32][0][2][2] = 10.0 # oct Mn2+ - tet Li - oct Mn2+
interact_tensors[32][0][2][3] = 10.0 # oct Mn2+ - tet Li - oct Mn3+
interact_tensors[32][0][2][4] = 10.0 # oct Mn2+ - tet Li - oct Mn4+

interact_tensors[32][0][3][2] = 10.0 # oct Mn3+ - tet Li - oct Mn2+
interact_tensors[32][0][3][3] = 10.0 # oct Mn3+ - tet Li - oct Mn3+
interact_tensors[32][0][3][4] = 10.0 # oct Mn3+ - tet Li - oct Mn4+

interact_tensors[32][0][4][2] = 10.0 # oct Mn4+ - tet Li - oct Mn2+
interact_tensors[32][0][4][3] = 10.0 # oct Mn4+ - tet Li - oct Mn3+
interact_tensors[32][0][4][4] = 10.0 # oct Mn4+ - tet Li - oct Mn4+

interact_tensors[29][0][0][0] = 10.0 #Li-Li-Li
interact_tensors[29][0][2][0] = 10.0 #Li-Mn2-Li
interact_tensors[29][0][3][0] = 10.0 #Li-Mn3-Li
interact_tensors[29][0][4][0] = 10.0 #Li-Mn4-Li

Custom_IO.write_pickle(interact_tensors, 'altered_int_tensor.json')

class Processor_Maker():
    
    def __init__(self,a,b,c):  

        self.cell_size = 4*a*b*c
        
        print("This instance of Processor_Maker can build processors for orthogonal cells.") 
        print("Please pass even numbers for a,b and c dimensions if you intend to calculate Spinel order parameter.")
        print(f"The number of anion sites in the cell that the current instance will build = {self.cell_size}")
        
        transformer = np.array([[-1, 1, 1],
                              [1, -1, 1],
                              [1, 1, -1]])
        
        blow_upper = np.array([a, b, c])
        self.sc_matrix = np.array([list(blow_upper[i] * transformer[i]) for i in range(len(blow_upper))])
        
        self.interact_tensors = np.load('altered_int_tensor.json', allow_pickle=True)
        
        with open('/global/u2/s/sanando/DRX_to_Delta/FUNCTIONS/wdr_74_3.json') as fin:
            self.ce = ClusterExpansion.from_dict(json.load(fin))
        
    def Processor_Maker(self):
        
        cd_processor = ClusterDecompositionProcessor(self.ce.cluster_subspace,
                                                     self.sc_matrix, self.interact_tensors)
        ewald_processor = EwaldProcessor(self.ce.cluster_subspace, self.sc_matrix,
                                         self.ce.cluster_subspace.external_terms[0],
                                         coefficient=self.ce.coefs[-1])
        composite = CompositeProcessor(self.ce.cluster_subspace, self.sc_matrix)
        composite.add_processor(cd_processor)
        composite.add_processor(ewald_processor)

        test_occu = [0 for x in range(4*self.cell_size)]
        
        print(composite.compute_property(np.array(test_occu))[0])

        Processor_filename = 'Processor_'+str(self.cell_size)+'_O.pickle'
        
        Custom_IO.write_pickle(composite.as_dict(), Processor_filename)