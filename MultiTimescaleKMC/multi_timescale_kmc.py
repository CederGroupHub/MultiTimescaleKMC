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

        self.Tet_Oct_Updater()
        
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

        #self.Redox_Neighbors = Redox_Center_Calculator()
        
        self.Conf = defaultdict(dict)           
        self.evolution_filename = f"Evolution_{self.T_KMC}K.pickle"
        self.Time = 0 
        self.step_file_name = "Step_number.txt"

        self.All_Hops = {
            'counter':-1,
            'Hops':defaultdict(dict),
            'Activation_Barriers':[],
            'Energy_Changes':[]
        }       

        self.Hop_Mechanisms = {
            "Tri-Vac":{},
            "Di-Vac":{},
            "Mono-Vac":{},
        }
        
        BarrierParams = namedtuple("BarrierParams", ["kra", "end_state", "encoding"])
        
        self.barrier_map = {
            "Tri-Vac": {
                "Mn3_oct": BarrierParams(0.67, self.Mn3, 1),
                "Mn3_tet": BarrierParams(0.67, self.Mn3, 2),
                "Mn2_oct": BarrierParams(0.3, self.Mn2, 3),
                "Mn2_tet": BarrierParams(0.3, self.Mn2, 4),  
                "Mn4": BarrierParams(1.5, self.Mn4, 5),
                "Ti4": BarrierParams(0.87, self.Ti4, 6),
            },
            "Di-Vac": {
                "Mn3_oct": BarrierParams(0.75, self.Mn3, 7),
                "Mn3_tet": BarrierParams(0.75, self.Mn3, 8),
                "Mn2_oct": BarrierParams(0.3, self.Mn2, 9),
                "Mn2_tet": BarrierParams(0.3, self.Mn2, 10),
            },
            "Mono-Vac": {
                "Mn2_oct": BarrierParams(0.45, self.Mn2, 11),
                "Mn2_tet": BarrierParams(0.45, self.Mn2, 12),
            },
        }

        self.hop_types = {
            "mn3_mn3": [1, 2, 7, 8],
            "mn2_mn2": [3, 4, 9, 10, 11, 12],
            "mn4_mn4": [5],
            "ti4_ti4": [6]
        }
    
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
        
        tm, vac = the_hop[encoding]

        if encoding in self.hop_types["mn3_mn3"]:                              ### Mn3+ ----> Mn3+

            idx_Mn3 = self.Species_Lists['Mn3'].index(tm)
            self.Species_Lists['Mn3'][idx_Mn3] = vac

            post_hop_specie = self.Mn3

            if encoding in [1,7]:
                self.Species_Lists['Mn3_oct'].remove(tm)
                self.Species_Lists['Mn3_tet'].append(vac)
            elif encoding in [2,8]:
                self.Species_Lists['Mn3_tet'].remove(tm)
                self.Species_Lists['Mn3_oct'].append(vac)               

        if encoding in self.hop_types["mn2_mn2"]:                              ### Mn2+ ----> Mn2+

            idx_Mn2 = self.Species_Lists['Mn2'].index(tm)
            self.Species_Lists['Mn2'][idx_Mn2] = vac

            post_hop_specie = self.Mn2

            if encoding in [3, 9, 11]:
                self.Species_Lists['Mn2_oct'].remove(tm)
                self.Species_Lists['Mn2_tet'].append(vac)
            elif encoding in [4, 10, 12]:
                self.Species_Lists['Mn2_tet'].remove(tm)
                self.Species_Lists['Mn2_oct'].append(vac)               

        if encoding in self.hop_types["mn4_mn4"]:
            idx_Mn = self.Species_Lists['Mn4'].index(tm)
            self.Species_Lists['Mn4'][idx_Mn] = vac

            post_hop_specie = self.Mn4

        if encoding in self.hop_types["ti4_ti4"]:
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

    def Redox_Center_Calculator(self, Redox_cutoff_dist=3):
    
        Redox_Neighbors = defaultdict(list)
        all_sites = self.indices['tet'] + self.indices['oct']
    
        for site1 in all_sites:
            for site2 in all_sites:
                if (site1!=site2) and (self.processor.structure[site1].distance(self.processor.structure[site2]) <= Redox_cutoff_dist):
                    Redox_Neighbors[site1].append(site2)
    
        return Redox_Neighbors
    
    def All_Possible_Hops(self):
        
        """
        Method to list all possible Transition metal hops.
        """
        
        self.All_Hops = {
            'counter':-1,
            'Hops':defaultdict(dict),
            'Activation_Barriers':[],
            'Energy_Changes':[]
        }       

        self.Hop_Mechanisms = {
            "Tri-Vac":{},
            "Di-Vac":{},
            "Mono-Vac":{},
        }
        
        Mobile_Mn_tet = self.Species_Lists["Mn2_tet"]+self.Species_Lists["Mn3_tet"]
        Mobile_Mn_oct = self.Species_Lists["Mn2_oct"]+self.Species_Lists["Mn3_oct"]
        Mobile_Mn = self.Species_Lists['Mn2']+self.Species_Lists['Mn3']

        tet_vac = [x for x in self.Species_Lists['Vac'] if x in self.indices['tet']]
        oct_vac = [x for x in self.Species_Lists['Vac'] if x in self.indices['oct']]
        Pristine_oct_vacs = []

        for o in oct_vac:
            oct_pristine_vacs = len([x for x in self.nns[o] if x in self.Species_Lists['Vac']])
            if oct_pristine_vacs==8:
                Pristine_oct_vacs.append(o)
        
        for vac in tet_vac:                    #Differentiating vacant tetraherdrals for hopping mechanism
            
            v_int = len([x for x in self.nns[vac] if x in self.Species_Lists['Vac']])
            if (v_int in [1,2,3]):
                FS_TMs = [x for x in self.nns[vac] if x in Mobile_Mn_oct]
                if (v_int==3) and (len(FS_TMs)==1):    #Trivac
                    self.Mechanism_Update(FS_TMs[0], "Tri-Vac")
                    self.Hop_Mechanisms["Tri-Vac"][FS_TMs[0]].append(vac) 
                elif (v_int==2):
                    li_int = len([x for x in self.nns[vac] if x in self.Species_Lists['Li']])
                    if (len(FS_TMs)==1) and (li_int==1):    #Di-vac
                        self.Mechanism_Update(FS_TMs[0], "Di-Vac")
                        self.Hop_Mechanisms["Di-Vac"][FS_TMs[0]].append(vac)         
                elif (v_int==1):
                    li_int = len([x for x in self.nns[vac] if x in self.Species_Lists['Li']])
                    if (len(FS_TMs)==1) and (li_int==2):    #Mono-vac mechanism
                        self.Mechanism_Update(FS_TMs[0], "Mono-Vac")
                        self.Hop_Mechanisms["Mono-Vac"][FS_TMs[0]].append(vac) 

            FS_Mn4s = [x for x in self.nns[vac] if x in self.Species_Lists['Mn4']]
            if (len(FS_Mn4s)==1):           
                v_octs = [x for x in self.nns[vac] if x in Pristine_oct_vacs]
                if (len(v_octs)==3):
                    self.Mechanism_Update(FS_Mn4s[0], "Tri-Vac")
                    for v_oct in v_octs:                
                        self.Hop_Mechanisms["Tri-Vac"][FS_Mn4s[0]].append(v_oct) 
                        
            FS_Tis = [x for x in self.nns[vac] if x in self.Species_Lists['Ti4']]
            if (len(FS_Tis)==1):           
                v_octs = [x for x in self.nns[vac] if x in Pristine_oct_vacs]
                if (len(v_octs)==3):
                    self.Mechanism_Update(FS_Tis[0], "Tri-Vac")
                    for v_oct in v_octs:                
                        self.Hop_Mechanisms["Tri-Vac"][FS_Tis[0]].append(v_oct)
        
        for mn in Mobile_Mn_tet:               #Differentiating Mn tetraherdrals for hopping mechanism
            FS_Vacs = [x for x in self.nns[mn] if x in self.Species_Lists['Vac']]
            if (len(FS_Vacs)==4):                                   #Trivac
                self.Hop_Mechanisms["Tri-Vac"][mn] = FS_Vacs
            elif (len(FS_Vacs)==3):                                   #Divac
                li_int = len([x for x in self.nns[mn] if x in self.Species_Lists['Li']])
                if li_int==1:
                    self.Hop_Mechanisms["Di-Vac"][mn] = FS_Vacs
            elif (len(FS_Vacs)==2):                                   #Monovac
                li_int = len([x for x in self.nns[mn] if x in self.Species_Lists['Li']])
                if li_int==2:
                    self.Hop_Mechanisms["Mono-Vac"][mn] = FS_Vacs 

        for mechanism in self.Hop_Mechanisms:
            for mn in self.Hop_Mechanisms[mechanism]:
                for vac in self.Hop_Mechanisms[mechanism][mn]:
                    for cation_description, hop_info in self.barrier_map[mechanism].items():
                        if mn in self.Species_Lists[cation_description]:  # Dynamically fetch the correct set (Mn3_oct, etc.) #globals()[mn_type]
                            self.Barrier_Calculator(hop_info.kra, int(mn), vac, hop_info.end_state, hop_info.encoding)
                            break  # Exit the loop once a match is found

    def Mechanism_Update(self, tm, mechanism):
        
        if tm not in self.Hop_Mechanisms[mechanism]:
            self.Hop_Mechanisms[mechanism][tm]=[]

    def Species_Indices(self):
        
        unwanted_lists = ['Li_Vac', "Mn3_tet", "Mn3_oct", "Mn2_tet", "Mn2_oct"]
        spec_indices = [self.Species_Lists[species] for species in self.Species_Lists if species not in unwanted_lists]
        return spec_indices

    def Tet_Oct_Updater(self):

        self.Species_Lists["Mn3_tet"] = [x for x in self.Species_Lists['Mn3'] if x in self.indices['tet']]
        self.Species_Lists["Mn3_oct"] = [x for x in self.Species_Lists['Mn3'] if x in self.indices['oct']]
        self.Species_Lists["Mn2_tet"] = [x for x in self.Species_Lists['Mn2'] if x in self.indices['tet']]   
        self.Species_Lists["Mn2_oct"] = [x for x in self.Species_Lists['Mn2'] if x in self.indices['oct']]