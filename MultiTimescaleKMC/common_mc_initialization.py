import os
import random
import numpy as np
from custom_io import Custom_IO
import matplotlib.pyplot as plt
from pymatgen.core import Species
import smol.cofe.space.domain as ForVac
from local_structure_details import Local_Structure_Details

class Common_Class():
    _processor_cache = {}
    
    """
    This class initializes all important attributes inherited by the CMC and KMC classes by using the processor_file location (str).
    Generates Cluster Expansion structure and local structure details using the Local_Structure_Details class for it such as
        (i) Octahedral and tetrahedral cation site indices.
        (ii) Nearest Neighbors of the cation sites
    Also contains other useful methods for resetting the occupancy string used in smol structures, Swap MC, plotting energy.

    Attributes:
        Li, Mn2, Mn3, Mn4, Ti4 (Species): description of the various atomic species in the structure. 
        Vac (Vacancy): description of the Vacancy species in the structure.
        kB (float): Boltzman constant
        processor (CompositeProcessor): pre-built smol processor for fast cluster expansion calculations on a super-cell of your choice. 
        n_sites (int): number of sites in the supercell for which the processor was built. 
        site_encodings (list[list[Species | Vacancy]]): A 2D list representing the types of species allowed on each site.
        indices (defaultdict[list]): Dictionary of tetrahedral and octahedral sites within the supercell structure of the processor.
        nns (defaultdict[list]): Dictionary of nearest neighbors for all possible cation sites in the supercell structure of the processor.
    
    """
    
    def __init__(self, processor_file: str):

        """
        Args:
            processor_file (str): Location of the pre-built processor file.
        """
        
        self.Li = Species.from_str("Li+")
        self.Vac = ForVac.Vacancy()
        self.Mn2 = Species.from_str("Mn2+")
        self.Mn3 = Species.from_str("Mn3+")
        self.Mn4 = Species.from_str("Mn4+")
        self.Ti4 = Species.from_str("Ti4+")
        self.O2 = Species.from_str("O2-")
        
        self.kB = 8.617*10**-5

        # ---- Cache the expensive part ----
        key = os.path.abspath(processor_file)
        cached = Common_Class._processor_cache.get(key)
        
        if cached is None:
            processor = Custom_IO.load_processor(key)

            n_sites = processor.num_sites
            site_encodings = processor.allowed_species
            indices = Local_Structure_Details.struct_indicies(processor)
            nns = Local_Structure_Details.Nearest_Neighbor_Calculator(processor, indices)

            cached = {
                "processor": processor,
                "n_sites": n_sites,
                "site_encodings": site_encodings,
                "indices": indices,
                "nns": nns,
            }
            Common_Class._processor_cache[key] = cached

        # Attach cached shared objects to this instance (cheap)
        self.processor = cached["processor"]
        self.n_sites = cached["n_sites"]
        self.site_encodings = cached["site_encodings"]
        self.indices = cached["indices"]
        self.nns = cached["nns"]
        
    def Occupancy_Resetter(self, spec_type = None, spec_indices = None):

        """
        Method to calculate the occupancies of all sites within the structure, given the types of species in the structure and their 
        corresponding indices.

        Args:
            spec_type (list[Species | Vacancy], optional): List of Species present in the structure. Deaults to stealing the spec_type 
                                                           attribute from its child class (see if statement below).
            spec_indices (list[list[int]], optional): A 2D list of integer type representing indices in the supercell structure. Indices 
                                                      in the inner lists correspond to the Species defined in spec_type. First dimension 
                                                      of spec_indices and length of spec_indices must be the same. Deaults to building it 
                                                      using the Species_Indices method of its child class (see if statement below).
        
        """
        
        species_list = [0 for x in range(self.n_sites)]
        
        if (spec_type==None) and (spec_indices==None):
            spec_indices = self.Species_Indices()
            spec_type = self.spec_type

        for idx, si in enumerate(spec_indices):
            for ind in si:
                species_list[ind] = spec_type[idx]

        occ = self.processor.encode_occupancy(species_list)

        return occ
    
    def Swap_MC(self):

        """
        Method to perform swaps in canonical MC.
        """
        
        Perturbation = self.Perturbation_Calculator()

        r1 = random.uniform(0, 1)
        p = np.exp(-(Perturbation['e_change'])/(self.kB*self.T_sample))

        if (Perturbation['e_change'] < 0) or ((Perturbation['e_change'] > 0) and (p > r1)):

            self.energy += Perturbation['e_change']

            a, b = Perturbation['swap_pair_lists']                     
            b[Perturbation['swap_pair_indices'][1]], a[Perturbation['swap_pair_indices'][0]] = Perturbation['swap_pair']  # Updating lists 
            self.occ[Perturbation['swap_pair'][0]], self.occ[Perturbation['swap_pair'][1]] = Perturbation['occ_exchange_pairs'] # Updating occs 
            
    def plot_energy_evolution(self):

        """
        Method to plot how the energy of the system has evolved in the MC simulation upto this point.
        """
        
        plt.figure(figsize=(12,8))
        
        Steps = [x for x in range(len(self.Energy_All))]
        Energy = (self.Energy_All-self.Energy_All[0])/self.n_atoms
        
        plt.scatter(Steps,Energy)
        
        plt.tight_layout()
        
        fontsize = 35
        plt.xlabel('Steps',fontsize = fontsize)
        plt.ylabel('Energy (eV/atom)', fontsize = fontsize)
        plt.xticks([0,int(np.max(Steps)/2),np.max(Steps)], fontsize = fontsize)
        plt.yticks(fontsize = fontsize)        
        plt.show()