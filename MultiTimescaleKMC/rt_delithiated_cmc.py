import os
import random
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from custom_io import Custom_IO
from collections import defaultdict
from common_mc_initialization import Common_Class
from initial_structures import Initial_Structure_Makers

class Delithiated_RT_CMC(Common_Class):

    """
    This class performs Canonical Monte carlo (CMC) simulations at room temperature for delithiated DRX compositions. 
    The simulation proposes Li-Vac and Mn3+-Mn4+ swaps. 

    Attributes:
        Species_Lists (dict): Dictionary with lists of all Species including vacancies in the structure
        comp_Li (float): Delithiated composition of Li for the DRX structure where comp_Li=z in (Liz,Vac)xTiyMn(2-x-y)O2 format.
        occ: smol occupancies for all sites in the structure.
        sampling_steps (int): number of MC steps proposed. 
        T_sample (int): temperature of the simulation. 
        energy (float): Energy of the structure in the current configuration
        Energy_All (list[float]): List of energies as the simulation proceeds.  
        swaps (list[int]): numerical representations of the two types of perturbations that can occur.
        Conf (defaultdict(dict)): Configuration at the lowest energy of the CMC simulations.
        RT_Configuration_filename (str): File in which Conf is written.
        n_atoms (int): Total number of atoms in the structure
        
    """

    def __init__(self, comp_Li, processor_file: str, sampling_steps: int, RT_Configuration_filename: str = "Delithiated_RT_DRX.pickle"):
        
        super().__init__(processor_file)
        self.Species_Lists = Custom_IO.load_pickle(os.getcwd(), "HT_DRX.pickle")
        self.Species_Lists.pop('Energy_All')
        
        self.comp_Li = comp_Li
        
        Structure_Maker = Initial_Structure_Makers()
        Structure_Maker.initialize_delithiated_disordered_state(self)
        
        spec_type = [self.Li, self.Mn3, self.Mn4, self.Ti4, self.Vac, self.O2]
        spec_indices = [self.Species_Lists[species] for species in self.Species_Lists]+[self.indices['O2']]

        self.occ = self.Occupancy_Resetter(spec_type = spec_type, spec_indices = spec_indices)

        self.sampling_steps = sampling_steps
        self.T_sample = 300

        self.energy = self.processor.compute_property(self.occ)[0]
        self.Energy_All = np.array([self.energy])
        
        self.swaps = [0,1]
        
        self.Conf = defaultdict(dict)                                          #Configurational information at each step
        self.RT_Configuration_filename = RT_Configuration_filename
        self.n_atoms = np.sum([len(self.Species_Lists[species]) for species in self.Species_Lists if (species!='Li_Vac') and (species!='Vac')])

    def RT_CMC(self):

        """
        Method to perform room temperature CMC
        """        
        
        for i in range(self.sampling_steps):             #Metropolis algorithm for canonical swaps accross sublattices.
            self.Swap_MC()
            self.Energy_All = np.append(self.Energy_All,self.energy)
            if np.min(self.Energy_All) == self.energy:
                self.Write_Configuration()
                
        self.plot_energy_evolution()

    def Write_Configuration(self):

        """
        Method to write the lowest energy atomic configuration obtained during the simulation.
        """
        
        self.Conf = {
            'Li':self.Species_Lists['Li'].copy(),
            'Vac':self.Species_Lists['Vac'].copy(),
            'Mn3':self.Species_Lists['Mn3'].copy(),
            'Mn4':self.Species_Lists['Mn4'].copy(),
            'Ti4':self.Species_Lists['Ti4'].copy(),
            'Energy_All':self.Energy_All.copy()}  

        Custom_IO.write_pickle(self.Conf, self.RT_Configuration_filename)
        
    def Perturbation_Calculator(self):    

        """
        Method to select a perturbation for proposing to the MC simulation.
        """
        
        r = random.sample(self.swaps,1)[0]

        if r == 0:            
            a = self.Species_Lists['Li']
            b = self.Species_Lists['Vac']

            a_specie = self.Li
            b_specie = self.Vac

        elif r == 1:            
            a = self.Species_Lists['Mn3']
            b = self.Species_Lists['Mn4']

            a_specie = self.Mn3
            b_specie = self.Mn4

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