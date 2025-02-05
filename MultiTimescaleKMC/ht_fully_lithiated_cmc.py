import random
import numpy as np
from random import sample
from custom_io import Custom_IO
import matplotlib.pyplot as plt
from collections import defaultdict

from common_mc_initialization import Common_Class
from initial_structures import Initial_Structure_Makers

class DRX_CMC(Common_Class):

    """
    This class performs Canonical Monte carlo (CMC) simulations at room temperature for delithiated DRX compositions. 
    The simulation proposes Li-Vac, Mn3+-Ti4+ and Li-Ti4+ swaps. 

    Attributes:
        sampling_steps (int): number of MC steps proposed. 
        T_sample (int): temperature of the simulation. 
        HT_Configuration_filename (str): File in which Conf is written.
        Species_Lists (dict): Dictionary with lists of all Species including vacancies in the structure
        comp_Li, comp_Ti (float): Composition of Li and Ti for the DRX structure in the LixTiyMn(2-x-y)O2 format.
        n_atoms (int): Total number of atoms in the structure
        occ: smol occupancies for all sites in the structure.
        energy (float): Energy of the structure in the current configuration
        Energy_All (list[float]): List of energies as the simulation proceeds.  
        Energy_Unique (np.array): Array of all unique energies of the equilibrated system not sampled previously during the the simulation.
        Mn3_Configs, Mn4_Configs, Ti4_Configs, Li_Configs (list[list[int]]): 2D arrays of Mn3+, Mn4+, Ti4+ and Li indices in corresponding to the energies in Energy_Unique. 
        swaps (list[int]): numerical representations of the two types of perturbations that can occur.
        Conf (defaultdict(dict)): Atomic configuration of the equilibrated MC structure.
        
    """
    
    
    def __init__(self, comp_Li: float, comp_Ti: float, processor_file: str, sampling_steps: int, T_sample: int):
        super().__init__(processor_file)
        
        self.sampling_steps = sampling_steps
        self.T_sample = T_sample
        
        self.HT_Configuration_filename = "HT_DRX.pickle"
        
        self.comp_Li = comp_Li
        self.comp_Ti = comp_Ti
        
        Structure_Maker = Initial_Structure_Makers()
        self.Species_Lists= Structure_Maker.initialize_random_specie_indices(self)

        spec_type = [self.Li, self.Vac, self.Mn3, self.Mn4, self.Ti4, self.O2]
        spec_indices = [self.Species_Lists[species] for species in self.Species_Lists]+[self.indices['O2']]
        self.n_atoms = np.sum([len(self.Species_Lists[species]) for species in self.Species_Lists if (species!='Li_Vac') and (species!='Vac')])

        self.occ = self.Occupancy_Resetter(spec_type = spec_type, spec_indices = spec_indices)

        self.energy = self.processor.compute_property(self.occ)[0]
        self.Energy_All = np.array([self.energy])
        self.Energy_Unique = np.array([])
        
        self.Mn3_Configs = []
        self.Mn4_Configs = []
        self.Ti4_Configs = []
        self.Li_Configs = []
        
        self.swaps = [0,1,2]
        
        self.Conf = defaultdict(dict)                                          #Configurational information at each step
        
    def HT_CMC(self):

        """
        Method to perform high temperature CMC
        """

        for i in range(self.sampling_steps):             #Metropolis algorithm for canonical swaps accross sublattices.
            self.Swap_MC()
            if (i>100000) and (self.energy not in self.Energy_All):
                self.DRX_Configs_Update()
            self.Energy_All = np.append(self.Energy_All,self.energy)
        self.Write_Configuration()
        self.plot_energy_evolution()
                
    def Write_Configuration(self):

        """
        Method to write the equilibrated atomic configuration after the simulation has finished.
        """
        
        idx = np.argmin(np.abs(self.Energy_Unique - np.mean(self.Energy_Unique)))

        self.Conf = {
            'Li':self.Li_Configs[idx],
            'Mn3':self.Mn3_Configs[idx],
            'Mn4':self.Mn4_Configs[idx],
            'Ti4':self.Ti4_Configs[idx],
            'Energy_All':self.Energy_All.copy()
        }
        
        Custom_IO.write_pickle(self.Conf, self.HT_Configuration_filename)
        
    def DRX_Configs_Update(self): 

        """
        Method to keep track of all unique equilibrated DRX configurations.
        """
        
        self.Mn3_Configs.append(self.Species_Lists['Mn3'].copy())
        self.Mn4_Configs.append(self.Species_Lists['Mn4'].copy())
        self.Ti4_Configs.append(self.Species_Lists['Ti4'].copy())
        self.Li_Configs.append(self.Species_Lists['Li'].copy())
        self.Energy_Unique = np.append(self.Energy_Unique,self.energy)
        
    def Perturbation_Calculator(self):

        """
        Method to select a perturbation for proposing to the MC simulation.
        """

        r = random.sample(self.swaps,1)[0] 

        if r == 0:   
            a = self.Species_Lists['Li']
            b = self.Species_Lists['Mn3']

            a_specie = self.Li
            b_specie = self.Mn3

        elif r == 1:    
            a = self.Species_Lists['Mn3']
            b = self.Species_Lists['Ti4']

            a_specie = self.Mn3
            b_specie = self.Ti4

        elif r == 2:            
            a = self.Species_Lists['Ti4']
            b = self.Species_Lists['Li']

            a_specie = self.Ti4
            b_specie = self.Li

        a_Swap = sample(a,1)[0]
        b_Swap = sample(b,1)[0]

        energy_change = self.processor.compute_property_change(self.occ,[(a_Swap, self.site_encodings[a_Swap].index(b_specie)), (b_Swap, self.site_encodings[b_Swap].index(a_specie))])[0]

        occ_a_to_b = self.site_encodings[a_Swap].index(b_specie)
        occ_b_to_a = self.site_encodings[b_Swap].index(a_specie)
        
        Perturbation = {
            'swap_pair':[a_Swap,b_Swap],
            'swap_pair_lists':[a, b], 
            'swap_pair_indices':[a.index(a_Swap), b.index(b_Swap)],
            'occ_exchange_pairs':[occ_a_to_b, occ_b_to_a],
            'e_change':energy_change
        }
        
        return Perturbation