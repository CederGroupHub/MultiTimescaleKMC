import os
import random
import pickle
import numpy as np
from random import sample
from pymatgen.core import Species
from collections import defaultdict
import smol.cofe.space.domain as ForVac


class Initial_Structure_Makers():

    """
    Class contains multiple methods to generate the initial structures for different MC simulations.
    """
    
    def Initial_Layered_Structure(self, self_kmc):
        
        Li_Layer = self_kmc.Starting_Configuration['Li_Layer'].copy()
        Mn_Layer = self_kmc.Starting_Configuration['Mn_layer'].copy()

        Replace_Li=random.sample(Li_Layer, int(self_kmc.disorder_fraction*len(Li_Layer)))
        Replace_Mn=random.sample(Mn_Layer, int(self_kmc.disorder_fraction*len(Mn_Layer)))

        for r in Replace_Li:
            Li_Layer.remove(r)
            Mn_Layer.append(r)

        for r in Replace_Mn:
            Li_Layer.append(r)
            Mn_Layer.remove(r)

        Li_l = random.sample(Li_Layer,int(len(Li_Layer)/4)).copy()
        Vac_l = self_kmc.indices['tet'].copy() + [octa for octa in Li_Layer if octa not in Li_l].copy()
        List_Li_Vac = Li_l.copy()+Vac_l.copy()

        Mn4_l = random.sample(Mn_Layer,int(0.75*len(Mn_Layer))).copy()
        Mn3_l = [octa for octa in Mn_Layer if octa not in Mn4_l].copy()

        Ti4_l = []
        Mn2_l = []
        
        O2_l = self_kmc.indices['O2'].copy()
        
        Species_Lists = {
            'Vac': Vac_l,
            'Li': Li_l,
            'Li_Vac': List_Li_Vac,
            'Mn2': Mn2_l,
            'Mn3': Mn3_l,
            'Mn4': Mn4_l,
            'Ti4': Ti4_l,
            'O2': O2_l
        }
        
        return Species_Lists
    
        
    def initialize_random_specie_indices(self, self_DRX):

        """
        Method to initialize a disordered structure using the comp_Li and comp_Ti atributes of the DRX_CMC class.
        DRX_CMC class instance (self_DRX) is passed in as an arguement.
        The method returns the atomic configurations as part of the dictionary Species_Lists.
        """
        
        mn = 2-self_DRX.comp_Ti-self_DRX.comp_Li

        idx = np.argmin(np.abs(np.array([4*(x*0.01)+3*(mn-(x*0.01)) for x in range(int(mn*100))])-(4-4*self_DRX.comp_Ti-self_DRX.comp_Li)))
        mn3 = np.array([np.round(mn-(x*0.01),2) for x in range(int(mn*100))])[idx]
        
        n_oct_sites = len(self_DRX.indices['oct'])

        n_Li = int( (self_DRX.comp_Li/2)*(n_oct_sites) )
        n_Ti = int( (self_DRX.comp_Ti/2)*(n_oct_sites) )
        n_Mn3 = int( (mn3/2)*(n_oct_sites) )
        
        n_Mn4 = n_oct_sites - (n_Li+n_Ti+n_Mn3)

        Tets = self_DRX.indices['tet'].copy()
        Octs = self_DRX.indices['oct'].copy()
        O2_l = self_DRX.indices['O2'].copy()

        Ti4_l =  sample(Octs, n_Ti)
        for j in Ti4_l:
            Octs.remove(j)

        Mn3_l = sample(Octs, n_Mn3)
        for j in Mn3_l:
            Octs.remove(j)
            
        Mn4_l = sample(Octs, n_Mn4)
        for j in Mn4_l:
            Octs.remove(j)

        Li_l = Octs.copy()

        Species_Lists = {
            'Li': Li_l,
            'Tets': Tets,
            'Mn3': Mn3_l,
            'Mn4': Mn4_l,
            'Ti4': Ti4_l
        }

        return Species_Lists
    
    def initialize_delithiated_disordered_state(self, self_RT_DRX):

        """
        Method the Species_Lists attribute o initialize a disordered structure using the comp_Li atribute of the Delithiated_RT_CMC class.
        Delithiated_RT_CMC class instance (self_RT_DRX) is passed in as an arguement.
        The method updates configurations as part of the dictionary Species_Lists.
        """
    
        #Randomly chosing Li-Vac sites after delithiation.
        Li_Vac_l = self_RT_DRX.Species_Lists['Li'].copy() 
        
        n_Li = int((self_RT_DRX.comp_Li/2)*len(self_RT_DRX.indices['oct']))
        self_RT_DRX.Species_Lists['Li'] = sample(Li_Vac_l, n_Li)
        self_RT_DRX.Species_Lists['Vac'] = [x for x in Li_Vac_l if x not in self_RT_DRX.Species_Lists['Li']]+self_RT_DRX.indices['tet'].copy()
        
        del_Mn3 = len(Li_Vac_l)-n_Li
        new_Mn4_l = sample(self_RT_DRX.Species_Lists['Mn3'], del_Mn3)
        
        for mn4 in new_Mn4_l:
            self_RT_DRX.Species_Lists['Mn3'].remove(mn4)
            
        self_RT_DRX.Species_Lists['Mn4']+=new_Mn4_l