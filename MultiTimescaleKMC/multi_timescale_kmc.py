import os
import random
import pickle
import numpy as np
from custom_io import Custom_IO
from pymatgen.core import Species
from collections import defaultdict
from fast_processes import Fast_Processes_MC
from common_mc_initialization import Common_Class
from initial_structures import Initial_Structure_Makers
from local_structure_details import Local_Structure_Details

class Multi_Time_Scale_KMC(Common_Class):
    
    """
    This class performs Canonical Monte carlo (CMC) simulations at high temperatures for fully-lithiated DRX compositions. 
    The simulation proposes Li-Vac and Mn3+-Mn4+ swaps. 

    Attributes:
        Species_Lists (dict): Dictionary with lists of all Species including vacancies in the structure
        comp_Li (float): Delithiated composition of Li
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
    
    
    def __init__(self, T_KMC: int, traj_steps: int, processor_file: str, RT_CMC_results_file:str = "Delithiated_RT_DRX.pickle"):    #, disorder_fraction
        
        super().__init__(processor_file)
        
        self.Species_Lists = Custom_IO.load_pickle(RT_CMC_results_file)         #You should have this file within the current directory
        self.Species_Lists.pop('Energy_All')
        self.Species_Lists["Mn2"]=[]
        self.Species_Lists["O2"]=self.indices['O2']
        self.Species_Lists['Li_Vac'] = self.Species_Lists['Vac'].copy()+self.Species_Lists['Li'].copy()
        self.n_atoms = np.sum([len(self.Species_Lists[species]) for species in self.Species_Lists if (species!='Li_Vac') and (species!='Vac')])
        
        self.spec_type = [self.Li, self.Vac, self.Mn3, self.Mn4, self.Ti4, self.Mn2, self.O2]

        self.occ = self.Occupancy_Resetter()
        self.energy = self.processor.compute_property(self.occ)[0]
        self.av_energy = 0
        self.Energy_All = np.array([])
        
        self.T_KMC = T_KMC
        self.e_cut = 6.96*10**-3 * self.T_KMC
        self.T_sample = 2000
        
        self.traj_steps = traj_steps
        #self.disorder_fraction = disorder_fraction

        self.All_Hops = {}
        
        self.Conf = defaultdict(dict)           
        self.evolution_filename = f"Evolution_{self.T_KMC}K.pickle"
        self.Time = 0 
        self.step_file_name = "Step_number.txt"
    
    def run_KMC(self):
        
        """
        This method calls upon other funtions to run KMC simulations with the number of TM hops equal to the traj_steps attribute. 
        The Select_Fast_Configuration method of the Fast_Processes_MC class is used to requilbrate the faster kinetic processes after each hop. 
        The Hop method of this class is used to perform the TM hop.
        
        """
        
        Fast_Processes = Fast_Processes_MC()
        for s in range(self.traj_steps):    
            Fast_Processes.Select_Fast_Configuration(self, s)      #JUST TO PROVE THAT WE HAVE DONE OUR DUE DILLIGENCE
            self.Hop(s)
            if (s!=0) and (s%2000==0):
                self.plot_energy_evolution()
        
    def Hop(self, s: int):
        
        """
        This method calls upon other methods to list all possible hops, advance the time based on the list, 
        find the relevant hop to be executed, save infomation regarding this hop, execute the hop and write the step number of the 
        current hop in the file named in the step_file_name attribute. 

        Args:
            s (int): current KMC step number
        """
        
        self.All_Possible_Hops()       #CAN GNNS DO THIS FASTER?
        self.Time_Advancement()
        the_hop, encoding = self.Hop_Finder()                     #The TM Hop energy change was updated
        self.Write_Evolution_File(s, the_hop, encoding)                   #encoding is for keeping track of the mechanism
        self.Hop_Executer(the_hop, encoding)  
        Custom_IO.write_step_file(s, self.step_file_name)
        
    def Time_Advancement(self):
        
        """
        Method to advance the time associated with the TM hop. 
        """
        
        attempt_frequency = 1e13

        rate_consts = np.exp(-np.array(self.All_Hops['Activation_Barriers'])/(self.kB*self.T_KMC))*attempt_frequency
        r_time = random.uniform(0, 1)
        self.Time = self.Time - (np.log(r_time)/np.sum(rate_consts))
        
    def Hop_Finder(self) -> tuple[dict , int]:
        
        """
        Method to find the hop to be executed from a the dictionary All_Hops attribute of all possible hops.
        """

        hop_probs = np.exp(-(np.array(self.All_Hops['Activation_Barriers']))/(self.kB*self.T_KMC))
        probs = [np.sum(hop_probs[0:i+1])/np.sum(hop_probs) for i in range(len(hop_probs))]
        r = random.uniform(0, 1)
        idx = probs.index([i for i in probs if i > r][0])

        the_hop = self.All_Hops['Hops'][idx]
        encoding = list(the_hop.keys())[0]

        self.energy += self.All_Hops['Energy_Changes'][idx]

        return the_hop, encoding
        
    def Write_Evolution_File(self, s: int, the_hop: dict, encoding: int):            

        """
        Method to write the evolution file for tracking the hop information, average energy and time associated with the hop.

        Args:
            s (int): current KMC step number
            the_hop (dict): information for regarding the hop occuring
            encoding (int): code for the type of hop occuring
        """
        
        if s%20==0:
            
            print("Trajectory Step Number:  " + str(s) + "\n")
            
            self.Conf[s] = {
                'Mn2_l':self.Species_Lists['Mn2'].copy(),
                'Mn3_l':self.Species_Lists['Mn3'].copy(),
                'Mn4_l':self.Species_Lists['Mn4'].copy(),
                'Li_l':self.Species_Lists['Li'].copy(),
                'Ti_l':self.Species_Lists['Ti4'].copy(),
                'Av_Energy':self.av_energy,
                'Hop': the_hop[encoding],
                'Encoding': encoding,
                'time':self.Time
            } 
            
            Custom_IO.write_pickle(self.Conf, self.evolution_filename)

        else:
            self.Conf[s] = {
                'Av_Energy':self.av_energy,
                'time':self.Time,
                'Hop': the_hop[encoding],
                'Encoding': encoding
            }
        
    def Li_Vac_Updater(self, mn: int, vac: int):

        """
        Method to update the Li and vacancy species lists.

        Args:
            mn (int): index of the TM site involved in the hop
            vac (int): index of the Vac site involved in the hop
        
        """
        
        idx_Vac = self.Species_Lists['Li_Vac'].index(vac)
        self.Species_Lists['Li_Vac'][idx_Vac] = mn

        idx_Vac = self.Species_Lists['Vac'].index(vac)
        self.Species_Lists['Vac'][idx_Vac] = mn 
    
    def Hop_Executer(self, the_hop: dict, encoding: int):
        
        """
        Method to execute the TM hop.

        Args:
            the_hop (dict): information for regarding the hop occuring
            encoding (int): code for the type of hop occuring
        """
        
        mn3_mn3 = [1,2,3,4]
        mn2_mn2 = [5,6,7,8,9,10]

        tm, vac = the_hop[encoding]

        if encoding in mn3_mn3:                              ### Mn3+ ----> Mn3+

            idx_Mn3 = self.Species_Lists['Mn3'].index(tm)
            self.Species_Lists['Mn3'][idx_Mn3] = vac

            post_hop_specie = self.Mn3

        if encoding in mn2_mn2:                              ### Mn2+ ----> Mn2+

            idx_Mn2 = self.Species_Lists['Mn2'].index(tm)
            self.Species_Lists['Mn2'][idx_Mn2] = vac

            post_hop_specie = self.Mn2

        if encoding==11:
            idx_Mn = self.Species_Lists['Mn4'].index(tm)
            self.Species_Lists['Mn4'][idx_Mn] = vac

            post_hop_specie = self.Mn4

        if encoding==12:
            idx_Ti = self.Species_Lists['Ti4'].index(tm)
            self.Species_Lists['Ti4'][idx_Ti] = vac

            post_hop_specie = self.Ti4

        self.occ[tm] = self.site_encodings[tm].index(self.Vac)
        self.occ[vac] = self.site_encodings[vac].index(post_hop_specie)

        self.Li_Vac_Updater(tm, vac)    

    def Barrier_Calculator(self, kra: float, mn: int, vac: int, end1: Species, ec: int) -> dict:

        """
        Method to calculate the migration barrier associated with a PROPOSED hop using its KRA and end point energies.

        Args:
            kra (float): Kinetically resolved activation barrier for the PROPOSED hop.
            mn (int): index of the TM site involved in the PROPOSED hop.
            vac (int): index of the Vac site involved in the PROPOSED hop.
            end1 (Species): the type of species of the TM in the final position if the PROPOSED hop were to occur.
            ec (int): code for the type of hop PROPOSED.
        """
        
        self.All_Hops['counter']+=1

        self.All_Hops['Hops'][self.All_Hops['counter']][ec] = mn,vac
        change = self.processor.compute_property_change(self.occ,[(mn, self.site_encodings[mn].index(self.Vac)), (vac, self.site_encodings[vac].index(end1))])[0]

        self.All_Hops['Energy_Changes'].append(change)
        barrier = (change/2)+ kra
        self.All_Hops['Activation_Barriers'].append(barrier)

    def All_Possible_Hops(self):
        
        """
        Method to list all possible Transition metal hops.
        """
        
        e_kra_Mn2_MonoVac = 0.45          #Use for Tet-to-Oct only?
        e_kra_Mn2_DiVac = 0.3
        e_kra_Mn2_TriVac = 0.3

        e_kra_Mn3_DiVac = 0.75            #Use for Tet-to-Oct only? No
        e_kra_Mn3_TriVac = 0.67

        e_kra_Ti4_TriVac = 0.87

        e_kra_Mn4_TriVac = 1.5

        mn_vac_neighbors = defaultdict(dict)
        ti_vac_neighbors = defaultdict(list)
        mn4_vac_neighbors = defaultdict(list)
        
        self.All_Hops = {
            'counter':-1,
            'Hops':defaultdict(dict),
            'Activation_Barriers':[],
            'Energy_Changes':[]
        }

        Mn3_tet = [x for x in self.Species_Lists['Mn3'] if x in self.indices['tet']]
        Mn3_oct = [x for x in self.Species_Lists['Mn3'] if x in self.indices['oct']]

        Mn2_tet = [x for x in self.Species_Lists['Mn2'] if x in self.indices['tet']]         #np.intersect1d(Mn2_l, tet_oct_ind['tet'])
        Mn2_oct = [x for x in self.Species_Lists['Mn2'] if x in self.indices['oct']]         #np.intersect1d(Mn2_l, tet_oct_ind['oct'])

        Mobile_Mn_tet = Mn2_tet+Mn3_tet
        Mobile_Mn_oct = Mn2_oct+Mn3_oct
        Mobile_Mn = self.Species_Lists['Mn2']+self.Species_Lists['Mn3']

        tet_vac = [x for x in self.Species_Lists['Vac'] if x in self.indices['tet']]

        tet_in_3_vac = []                                               #Tri-vacancy mechanism when tetrahedral site is vacant
        tet_in_2_vac = []
        tet_in_1_vac = []

        for vac in tet_vac:                    #Differentiating vacant tetraherdrals for hopping mechanism
            v_int = len([x for x in self.nns[vac] if x in self.Species_Lists['Vac']])
            if (v_int==3):
                m_int = len([x for x in self.nns[vac] if x in Mobile_Mn_oct])
                if (m_int==1):    #Trivac
                    tet_in_3_vac.append(vac) 
            elif (v_int==2):
                m_int = len([x for x in self.nns[vac] if x in Mobile_Mn_oct])
                li_int = len([x for x in self.nns[vac] if x in self.Species_Lists['Li']])
                if (m_int==1) and (li_int==1):    #Di-vac
                    tet_in_2_vac.append(vac)         
            elif (v_int==1):
                m_int = len([x for x in self.nns[vac] if x in Mobile_Mn_oct])
                li_int = len([x for x in self.nns[vac] if x in self.Species_Lists['Li']])
                if (m_int==1) and (li_int==2):    #Mono-vac mechanism
                    tet_in_1_vac.append(vac) 

        for mn in Mobile_Mn_oct:                  #finding Vacancies Mobile Mn neighborhood
            mn_vac_neighbors[mn]["Tri-Vac"] =  [x for x in self.nns[mn] if x in tet_in_3_vac]              #np.intersect1d(nns[mn], tet_in_3_vac).astype(int)
            mn_vac_neighbors[mn]["Di-Vac"] =  [x for x in self.nns[mn] if x in tet_in_2_vac]
            mn_vac_neighbors[mn]["Mono-Vac"] =  [x for x in self.nns[mn] if x in tet_in_1_vac]

        tet_out_3_Mn = [] #Tri-vacancy mechanism

        Mn3_tet_TriVac = []
        Mn2_tet_TriVac = []

        tet_out_2_Mn = [] #Di-vacancy mechanism

        Mn3_tet_DiVac = []
        Mn2_tet_DiVac = []

        tet_out_1_Mn = [] #Mono-vacancy mechanism

        Mn2_tet_MonoVac = []

        for mn in Mobile_Mn_tet:               #Differentiating Mn tetraherdrals for hopping mechanism
            v_int = len([x for x in self.nns[mn] if x in self.Species_Lists['Vac']])
            if (v_int==4):                                   #Trivac
                tet_out_3_Mn.append(mn)
                if (mn in Mn3_tet):
                    Mn3_tet_TriVac.append(mn)
                if (mn in Mn2_tet):
                    Mn2_tet_TriVac.append(mn)
            elif (v_int==3):                                   #Divac
                li_int = len([x for x in self.nns[mn] if x in self.Species_Lists['Li']])
                if li_int==1:
                    tet_out_2_Mn.append(mn)
                    if (mn in Mn3_tet):
                        Mn3_tet_DiVac.append(mn)
                    if (mn in Mn2_tet):
                        Mn2_tet_DiVac.append(mn)
            elif (v_int==2):                                   #Monovac
                li_int = len([x for x in self.nns[mn] if x in self.Species_Lists['Li']])
                if li_int==2:
                    tet_out_1_Mn.append(mn)
                    if (mn in Mn2_tet):
                        Mn2_tet_MonoVac.append(mn)

        for mn in tet_out_3_Mn+tet_out_2_Mn+tet_out_1_Mn:
            mn_vac_neighbors[mn] = [x for x in self.nns[mn] if x in self.Species_Lists['Vac']]                   #np.intersect1d(nns[mn], Vac_l).astype(int)

        for mn in Mn3_oct:                ####(Reaction Encoding: 1) Mn3+ (Oct) --[Tri-Vac]--> Vac (as Mn3+)
            for vac in mn_vac_neighbors[mn]["Tri-Vac"]:
                self.Barrier_Calculator(e_kra_Mn3_TriVac, int(mn), vac, self.Mn3, ec=1)
                
        for mn in Mn3_tet_TriVac:         ####(Reaction Encoding: 2) Mn3+ (Tet) --[Tri-Vac]--> Vac (as Mn3+)
            for vac in mn_vac_neighbors[mn]:
                self.Barrier_Calculator(e_kra_Mn3_TriVac, int(mn), vac, self.Mn3, ec=2)
                
        for mn in Mn3_oct:                ####(Reaction Encoding: 3) Mn3+ (Oct) --[Di-Vac]--> Vac (as Mn3+)
            for vac in mn_vac_neighbors[mn]["Di-Vac"]:
                self.Barrier_Calculator(e_kra_Mn3_DiVac, int(mn), vac, self.Mn3, ec=3)

        for mn in Mn3_tet_DiVac:         ####(Reaction Encoding: 4) Mn3+ (Tet) --[Di-Vac]--> Vac (as Mn3+)
            for vac in mn_vac_neighbors[mn]:
                self.Barrier_Calculator(e_kra_Mn3_DiVac, int(mn), vac, self.Mn3, ec=4)

        for mn in Mn2_oct:              ####(Reaction Encoding: 5) Mn2+ (Oct) ----[Tri-Vac]--> Vac (as Mn2+)
            for vac in mn_vac_neighbors[mn]["Tri-Vac"]:
                self.Barrier_Calculator(e_kra_Mn2_TriVac, int(mn), vac, self.Mn2, ec=5)

        for mn in Mn2_tet_TriVac:       ####(Reaction Encoding: 6) Mn2+ (Tet) ----[Tri-Vac]--> Vac (as Mn2+)
            for vac in mn_vac_neighbors[mn]:
                self.Barrier_Calculator(e_kra_Mn2_TriVac, int(mn), vac, self.Mn2, ec=6)

        for mn in Mn2_oct:              ####(Reaction Encoding: 7) Mn2+ (Oct) ----[Di-Vac]--> Vac (as Mn2+)
            for vac in mn_vac_neighbors[mn]["Di-Vac"]:
                self.Barrier_Calculator(e_kra_Mn2_DiVac, int(mn), vac, self.Mn2, ec=7)

        for mn in Mn2_tet_DiVac:       ####(Reaction Encoding: 8) Mn2+ (Tet) ----[Di-Vac]--> Vac (as Mn2+)
            for vac in mn_vac_neighbors[mn]:
                self.Barrier_Calculator(e_kra_Mn2_DiVac, int(mn), vac, self.Mn2, ec=8)

        for mn in Mn2_oct:              ####(Reaction Encoding: 9) Mn2+ (Oct) ----[Mono-Vac]--> Vac (as Mn2+)
            for vac in mn_vac_neighbors[mn]["Mono-Vac"]:
                self.Barrier_Calculator(e_kra_Mn2_MonoVac, int(mn), vac, self.Mn2, ec=9)

        for mn in Mn2_tet_MonoVac:       ####(Reaction Encoding: 10) Mn2+ (Tet) ----[Mono-Vac]--> Vac (as Mn2+)
            for vac in mn_vac_neighbors[mn]:
                self.Barrier_Calculator(e_kra_Mn2_MonoVac, int(mn), vac, self.Mn2, ec=10)

        oct_vac = [x for x in self.Species_Lists['Vac'] if x in self.indices['oct']]
        z_TM_oct_vacs = []
        Pristine_oct_vacs = []

        for o in oct_vac:
            oct_vac_TM_nns = len([x for x in self.nns[o] if x in Mobile_Mn])    # Not a single TM cation  - Easier Bar to cross
            oct_pristine_vacs = len([x for x in self.nns[o] if x in self.Species_Lists['Vac']])     # Not a single FS cation
            if oct_vac_TM_nns==0:
                z_TM_oct_vacs.append(o)
            if oct_pristine_vacs==8:
                Pristine_oct_vacs.append(o)

        for vac in tet_vac:                    #Differentiating vacant tetraherdrals for hopping mechanism
            m_int = [x for x in self.nns[vac] if x in self.Species_Lists['Mn4']]
            if (len(m_int)==1):           
                v_octs = [x for x in self.nns[vac] if x in z_TM_oct_vacs]
                if (len(v_octs)==3):
                    for v_oct in v_octs:                
                        mn4_vac_neighbors[m_int[0]].append(v_oct)                                

        for mn in mn4_vac_neighbors: 
            for vac in mn4_vac_neighbors[mn]:
                self.Barrier_Calculator(e_kra_Mn4_TriVac, int(mn), int(vac), self.Mn4, ec=11)

        Pristine_Hops_Counter = 0
        Normal_Hops_Counter = 0

        for vac in tet_vac:                    #Differentiating vacant tetraherdrals for hopping mechanism
            t_int = [x for x in self.nns[vac] if x in self.Species_Lists['Ti4']]
            if (len(t_int)==1):           
                v_octs = [x for x in self.nns[vac] if x in z_TM_oct_vacs]
                if (len(v_octs)==3):
                    for v_oct in v_octs:                
                        ti_vac_neighbors[t_int[0]].append(v_oct)                    
                    Normal_Hops_Counter+=len(v_octs)            

        for ti in self.Species_Lists['Ti4']: 
            for vac in ti_vac_neighbors[ti]:
                self.Barrier_Calculator(e_kra_Ti4_TriVac, int(ti), int(vac), self.Ti4, ec=12)
    
    def Species_Indices(self):
        spec_indices = [self.Species_Lists[species] for species in self.Species_Lists if species != 'Li_Vac']
        return spec_indices