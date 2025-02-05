import os
import random
import pickle
import numpy as np
from random import sample
from pymatgen.core import Species
from collections import defaultdict
import smol.cofe.space.domain as ForVac

class Fast_Processes_MC():

    """
    Class to perform canonical monte-carlo (CMC) between transition metal (TM) hops in Kinetic Monte carlo (KMC) simulations 
    for the relatively faster kinetic processes such as Li-Vac swaps and --- presumably --- the charge transfer processes 
    between TM valence states (Mn2+, Mn3+, Mn4+). The class deliberately avoids defining a constructor method so that it only 
    has to be instantiated once outside the for loop of the run_KMC method in the the Multi_Time_Scale_KMC class. Instead, an 
    instance of the Multi_Time_Scale_KMC class (self_kmc) is passed in as arguement to the Fast_Configurations_Initialization 
    method which initializes the Fast_Configs dictionary relevant for this class. This dictionary is edited and returned by 
    multiple methods in this class.
    
    """

    def Select_Fast_Configuration(self, self_kmc, s: int) -> dict:

        """
        This method calls upon other methods within the class to list multiple (Li/ Vac/ TM charge) configurations by perfoming 
        CMC and choose one based on boltzman statistics.

        Args:
            self_kmc (Multi_Time_Scale_KMC): instance of the kmc class.
            s (int): step number
        
        """

        #if not isinstance(self_kmc, Multi_Time_Scale_KMC):
        #    raise TypeError(f"{self_kmc} must be an instance of the Multi_Time_Scale_KMC class")
        
        Fast_Configs = self.Fast_CMC(self_kmc,s)

        minimum = np.min(Fast_Configs['Energy_All'])

        eq_idx = np.where((Fast_Configs['Energy_All']-minimum)<self_kmc.e_cut)[0]

        self_kmc.av_energy = np.mean(Fast_Configs['Energy_All'][eq_idx])
        self_kmc.Energy_All = np.append(self_kmc.Energy_All,self_kmc.av_energy)

        prob_Li_Vac = np.exp(-(Fast_Configs['Energy_All']-minimum)/(self_kmc.kB*self_kmc.T_KMC))

        probs = [np.sum(prob_Li_Vac[0:i+1])/np.sum(prob_Li_Vac) for i in range(len(prob_Li_Vac))]
        r = random.uniform(0, 1)
        idx = probs.index([i for i in probs if i > r][0])

        self_kmc.energy = Fast_Configs['Energy_All'][idx]

        self_kmc.Species_Lists['Vac'] = list(Fast_Configs['Vac_configs'][idx])
        self_kmc.Species_Lists['Li'] = [j for j in self_kmc.Species_Lists['Li_Vac'] if j not in self_kmc.Species_Lists['Vac']]

        self_kmc.Species_Lists['Mn2'] = list(Fast_Configs['Mn2_configs'][idx])
        self_kmc.Species_Lists['Mn3'] = list(Fast_Configs['Mn3_configs'][idx])
        self_kmc.Species_Lists['Mn4'] = list(Fast_Configs['Mn4_configs'][idx])
        
        self_kmc.occ = self_kmc.Occupancy_Resetter()
            
    def Fast_CMC(self, self_kmc, s: int) -> dict:    

        """
        Method performs (Li/ Vac/ TM charge) CMC.

        Args:
            self_kmc (Multi_Time_Scale_KMC): instance of the kmc class.
            s (int): step number
        """
        
        if s==0:
            sampling_steps = 50000
        else:
            sampling_steps = 500
        
        Fast_Configs = self.Fast_Configurations_Initialization(self_kmc)

        for i in range(sampling_steps):             #Metropolis algorithm for canonical Li-Vac swaps accross sublattices.
            swaps = self.Swap_List_Calculator(self_kmc, Fast_Configs)
            Perturbation = self.Perturbation_Calculator(self_kmc, swaps, Fast_Configs)
            r1 = random.uniform(0, 1)
            p = np.exp(-(Perturbation['e_change'])/(self_kmc.kB*self_kmc.T_sample))
            
            if (Perturbation['e_change'] < 0) or ((Perturbation['e_change'] > 0) and (p > r1)):
                self_kmc.energy += Perturbation['e_change']                     #Fast Process Energy change
                a_Swap,b_Swap = Perturbation['swap_pair']
                self_kmc.occ[a_Swap], self_kmc.occ[b_Swap] = Perturbation['occ_exchange_pairs']
                Fast_Configs = self.Fast_Process_Executer(self_kmc, Perturbation, Fast_Configs)
            if self_kmc.energy not in Fast_Configs['Energy_All']:
                Fast_Configs = self.Fast_Config_Updater(self_kmc, Fast_Configs)
            
        return Fast_Configs
    
    def Fast_Configurations_Initialization(self, self_kmc) -> dict:

        """
        Method initializes and returns a dictionary which is updated with the (Li/ Vac/ TM charge) configurations during CMC.

        Args:
            self_kmc (Multi_Time_Scale_KMC): instance of the kmc class.
        """
        
        Energy_All = np.array([self_kmc.energy])

        Mn2_configs = [self_kmc.Species_Lists['Mn2'].copy()]
        Mn3_configs = [self_kmc.Species_Lists['Mn3'].copy()]
        Mn4_configs = [self_kmc.Species_Lists['Mn4'].copy()]
        Vac_configs = [self_kmc.Species_Lists['Vac'].copy()]

        Mn2_oct = [x for x in self_kmc.Species_Lists['Mn2'] if x in self_kmc.indices['oct']]                                  #list(np.intersect1d(Mn2_l, tet_oct_ind['oct']))
        Mn3_oct = [x for x in self_kmc.Species_Lists['Mn3'] if x in self_kmc.indices['oct']]                                  #list(np.intersect1d(Mn3_l, tet_oct_ind['oct']))  

        Fast_Configs = {
            'Energy_All':Energy_All,
            'Mn2_configs':Mn2_configs,
            'Mn3_configs':Mn3_configs,
            'Mn4_configs':Mn4_configs,
            'Vac_configs':Vac_configs,
            'Mn2_oct':Mn2_oct,
            'Mn3_oct':Mn3_oct
        }
        
        return Fast_Configs 

    def Swap_List_Calculator(self, self_kmc, Fast_Configs: dict) -> list:

        """
        This method returns a list of the type of perturbations possible.

        Args:
            self_kmc (Multi_Time_Scale_KMC): instance of the kmc class.
            Fast_Configs (dict): Configurations sampled in the Canonical Monte-carlo of the fast-processes.

        Returns:
            swaps (list): List of numbers representing the type of perturbations allowed in the current configuration of the structure.
        """
        
        if (len(self_kmc.Species_Lists['Mn2'])==0) and (len(self_kmc.Species_Lists['Mn3'])==0):
            swaps = [0]

        elif (len(self_kmc.Species_Lists['Mn2'])==0) and (len(self_kmc.Species_Lists['Mn3'])>0) and (len(Fast_Configs['Mn3_oct'])==0):
            swaps = [0]

        elif (len(self_kmc.Species_Lists['Mn2'])==0) and (len(Fast_Configs['Mn3_oct'])>0):     ###CORRECT
            swaps = [0, 2, 4]

        elif (len(self_kmc.Species_Lists['Mn2'])>0) and (len(Fast_Configs['Mn2_oct'])==0) and (len(self_kmc.Species_Lists['Mn3'])==0):
            swaps = [0, 5]

        elif (len(self_kmc.Species_Lists['Mn3'])==0) and (len(Fast_Configs['Mn2_oct'])>0):
            swaps = [0, 3, 5]

        elif (len(self_kmc.Species_Lists['Mn2'])>0) and (len(Fast_Configs['Mn2_oct'])==0) and (len(self_kmc.Species_Lists['Mn3'])>0) and (len(Fast_Configs['Mn3_oct'])==0):
            swaps = [0, 1, 5]

        elif (len(Fast_Configs['Mn2_oct'])>0) and (len(self_kmc.Species_Lists['Mn3'])>0) and (len(Fast_Configs['Mn3_oct'])==0):
            swaps = [0, 1, 3, 5]

        elif (len(self_kmc.Species_Lists['Mn2'])>0) and (len(Fast_Configs['Mn2_oct'])==0) and (len(Fast_Configs['Mn3_oct'])>0):
            swaps = [0, 1, 2, 4, 5]

        elif (len(Fast_Configs['Mn2_oct'] )>0) and (len(Fast_Configs['Mn3_oct'])>0):
            swaps = [0, 1, 2, 3, 4, 5]

        return swaps
    
    def Perturbation_Calculator(self, self_kmc, swaps: list, Fast_Configs: dict) -> dict:

        """
        This method returns a dictionary containing information regarding the swap chosen for proposing as a KMC step and the 
        energy change associated with it.

        Args:
            self_kmc (Multi_Time_Scale_KMC): instance of the kmc class.
            swaps (list): List of numbers representing the type of perturbations allowed in the current configuration of the structure. 
            Fast_Configs (dict): Configurations sampled in the Canonical Monte-carlo of the fast-processes.

        Returns:
            Perturbation (dict): Information about the perturbation which will actually be proposed.     
        """
        
        a, b = 0, 0
        swap = random.sample(swaps,1)[0]

        if swap == 0:
            a = self_kmc.Species_Lists['Li']
            b = self_kmc.Species_Lists['Vac']

            a_Swap = sample(a,1)[0]
            b_Swap = sample(b,1)[0]

            a_specie = self_kmc.Li
            b_specie = self_kmc.Vac

        elif swap == 1:                         #Swap
            a = self_kmc.Species_Lists['Mn2']
            b = self_kmc.Species_Lists['Mn3']

            a_Swap = sample(a,1)[0]
            b_Swap = sample(b,1)[0]

            a_specie = self_kmc.Mn2
            b_specie = self_kmc.Mn3

        elif swap == 2:                         #Swap
            a = self_kmc.Species_Lists['Mn3']
            b = self_kmc.Species_Lists['Mn4']

            a_Swap = sample(Fast_Configs['Mn3_oct'],1)[0]
            b_Swap = sample(b,1)[0]

            a_specie = self_kmc.Mn3
            b_specie = self_kmc.Mn4

        elif swap == 3:                          #Swap
            a = self_kmc.Species_Lists['Mn4']
            b = self_kmc.Species_Lists['Mn2']

            a_Swap = sample(a,1)[0]
            b_Swap = sample(Fast_Configs['Mn2_oct'],1)[0]

            a_specie = self_kmc.Mn4
            b_specie = self_kmc.Mn2

        elif swap == 4:                               ##############Disproportionation Reaction

            a_Swap = sample(Fast_Configs['Mn3_oct'],1)[0]
            if len([x for x in self_kmc.Species_Lists['Mn3'] if x != a_Swap])==0:
                print(Fast_Configs['Mn3_oct'], self_kmc.Species_Lists['Mn3'], [x for x in self_kmc.Species_Lists['Mn3'] if x != a_Swap])
            b_Swap = sample([x for x in self_kmc.Species_Lists['Mn3'] if x != a_Swap],1)[0]

            a_specie = self_kmc.Mn2
            b_specie = self_kmc.Mn4

        elif swap == 5:                               ##############Reverse Disproportionation Reaction

            a_Swap = sample(self_kmc.Species_Lists['Mn2'],1)[0]
            b_Swap = sample(self_kmc.Species_Lists['Mn4'],1)[0]

            a_specie = self_kmc.Mn3
            b_specie = self_kmc.Mn3

        occ_a_to_b = self_kmc.site_encodings[a_Swap].index(b_specie)
        occ_b_to_a = self_kmc.site_encodings[b_Swap].index(a_specie)

        energy_change = self_kmc.processor.compute_property_change(self_kmc.occ,[(a_Swap, occ_a_to_b), (b_Swap, occ_b_to_a)])[0]

        Perturbation = {
            'swap':swap,
            'swap_pair':[a_Swap,b_Swap],
            'swap_pair_lists':[a, b],
            'occ_exchange_pairs':[occ_a_to_b, occ_b_to_a],
            'e_change':energy_change
        }

        return Perturbation

    def Fast_Process_Executer(self, self_kmc, Perturbation: dict, Fast_Configs: dict) -> dict:

        """
        This method executes the CMC perturbation by randomly choosing one and updates the relevant dictionaries/ lists.

        Args:
            self_kmc (Multi_Time_Scale_KMC): instance of the kmc class.
            Perturbation (dict): List of numbers representing the type of perturbations allowed in the current configuration of the structure. 
            Fast_Configs (dict): Configurations sampled in the Canonical Monte-carlo of the fast-processes. 

        Returns:
            Fast_Configs (dict): Updates the dictionary passed in as arguement.    
        """
        
        r = Perturbation['swap'] 
        a_Swap, b_Swap = Perturbation['swap_pair'] 
        a, b = Perturbation['swap_pair_lists']
        
        if r in [0,1,2,3]:

            a_idx = a.index(a_Swap)
            b_idx = b.index(b_Swap)

            a[a_idx] = b_Swap
            b[b_idx] = a_Swap

            if r==1:
                if a_Swap in Fast_Configs['Mn2_oct']:
                    Fast_Configs['Mn2_oct'].remove(a_Swap)
                    Fast_Configs['Mn3_oct'].append(a_Swap)

                if b_Swap in Fast_Configs['Mn3_oct']:
                    Fast_Configs['Mn3_oct'].remove(b_Swap)
                    Fast_Configs['Mn2_oct'].append(b_Swap)

            if r==2:
                a_idx = Fast_Configs['Mn3_oct'].index(a_Swap)
                Fast_Configs['Mn3_oct'][a_idx] = b_Swap

            elif r==3:
                b_idx = Fast_Configs['Mn2_oct'].index(b_Swap)
                Fast_Configs['Mn2_oct'][b_idx] = a_Swap

        elif (r == 4):
            self_kmc.Species_Lists['Mn4'].append(a_Swap)
            self_kmc.Species_Lists['Mn2'].append(b_Swap)
            self_kmc.Species_Lists['Mn3'].remove(a_Swap)
            self_kmc.Species_Lists['Mn3'].remove(b_Swap)

            Fast_Configs['Mn3_oct'].remove(a_Swap)

            if b_Swap in Fast_Configs['Mn3_oct']:
                Fast_Configs['Mn3_oct'].remove(b_Swap)
                Fast_Configs['Mn2_oct'].append(b_Swap)

        elif (r == 5):
            self_kmc.Species_Lists['Mn2'].remove(a_Swap)
            self_kmc.Species_Lists['Mn4'].remove(b_Swap)
            self_kmc.Species_Lists['Mn3'].append(a_Swap)
            self_kmc.Species_Lists['Mn3'].append(b_Swap)

            if a_Swap in Fast_Configs['Mn2_oct']:
                Fast_Configs['Mn2_oct'].remove(a_Swap)
                Fast_Configs['Mn3_oct'].append(a_Swap)

            Fast_Configs['Mn3_oct'].append(b_Swap)

        return Fast_Configs
        
    def Fast_Config_Updater(self, self_kmc, Fast_Configs: dict) -> dict:            
        
        """
        This method updates the dictionary initialized in Fast_Configurations_Initialization

        Args:
            self_kmc (Multi_Time_Scale_KMC): instance of the kmc class.
            Fast_Configs (dict): Configurations sampled in the Canonical Monte-carlo of the fast-processes.

        Returns:
            Fast_Configs (dict): Updates the dictionary passed in as arguement.  
        
        """
        
        Fast_Configs['Energy_All'] = np.append(Fast_Configs['Energy_All'], self_kmc.energy)
        Fast_Configs['Mn2_configs'].append(self_kmc.Species_Lists['Mn2'].copy())
        Fast_Configs['Mn3_configs'].append(self_kmc.Species_Lists['Mn3'].copy())
        Fast_Configs['Mn4_configs'].append(self_kmc.Species_Lists['Mn4'].copy())
        Fast_Configs['Vac_configs'].append(self_kmc.Species_Lists['Vac'].copy())

        return Fast_Configs