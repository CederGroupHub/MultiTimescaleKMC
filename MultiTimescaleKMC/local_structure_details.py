from collections import defaultdict
import numpy as np

class Local_Structure_Details:

    @staticmethod
    def struct_indicies(processor):

        """
        Method takes a processor as and arguement and returns a dictionary of tetrahedral, octahedral and oxygen site indices in the 
        supercell for which the processor was built.
        """

        tet_oct_ind = defaultdict(list)

        for i,species in enumerate(processor.allowed_species):
            if len(species)==4:
                tet_oct_ind['tet'].append(i)
            if len(species)==6:
                tet_oct_ind['oct'].append(i)
            if len(species)==1:
                tet_oct_ind['O2'].append(i)

        return tet_oct_ind

    @staticmethod
    def Nearest_Neighbor_Calculator(processor, tet_oct_ind):

        """
        Method to calculate nearest neighbors for each cation site in the structure.
        """
        
        nn_dist = 1.81865

        nn_to_tet = defaultdict(list)
        nn_to_oct = defaultdict(list)
        nns = defaultdict(list)

        for tet_ind in tet_oct_ind['tet']:
            for oct_ind in tet_oct_ind['oct']:
                if abs(processor.structure[tet_ind].distance(processor.structure[oct_ind]) - nn_dist) <= 0.2:
                    nn_to_tet[tet_ind].append(oct_ind)
                    nn_to_oct[oct_ind].append(tet_ind)
                    nns[oct_ind].append(tet_ind)
                    nns[tet_ind].append(oct_ind)

        return nns

    @staticmethod
    def Oct_Oct_Neighbor_Types(processor, tet_oct_ind):

        """
        Method to calculate the 1st to 4th nearest neighbor octahedral sites for a given octahedral site.
        """
        
        neighbor_octs = defaultdict(list)

        for oct_ind1 in tet_oct_ind['oct']:
            neighbor_octs[oct_ind1] = {"nn1":[],"nn2":[],"nn3":[],"nn4":[]}
            for oct_ind2 in tet_oct_ind['oct']:
                if oct_ind2!=oct_ind1:
                    dist = processor.structure[oct_ind1].distance(processor.structure[oct_ind2])
                    if (dist < 3):
                        neighbor_octs[oct_ind1]["nn1"].append(oct_ind2)
                    elif (dist > 4) and ( dist < 5):
                        neighbor_octs[oct_ind1]["nn2"].append(oct_ind2)
                    elif (dist > 5) and ( dist < 5.5):
                        neighbor_octs[oct_ind1]["nn3"].append(oct_ind2)
                    elif (dist > 5.5) and ( dist < 6):
                        neighbor_octs[oct_ind1]["nn4"].append(oct_ind2)

        return neighbor_octs 
    
    @staticmethod
    def Triplet_Identifier(cut_off_radius, neighbor_octs):

        """
        Method to calculate all nearest neighbor octahedral triplets in which the different octahedral 
        sites within the structure are involved in. This information is useful in Spinel order parameter calculations.
        """
        
        nn1_dist = 2.97
        nn2_dist = 4.2
        nn3_dist = 5.14
        nn4_dist = 5.94
        
        Triplets = defaultdict(list)
        oct_indicies = list(neighbor_octs.keys())
        neighbor_types = list(neighbor_octs[oct_indicies[0]].keys())

        for idx in oct_indicies:   
            Triplets[idx] = []
            neighbors_of_interest = [idx]

            if (cut_off_radius > nn1_dist) and (cut_off_radius < nn2_dist):
                for i in range(1):
                    neighbors_of_interest += neighbor_octs[idx][neighbor_types[i]]
            elif (cut_off_radius > nn2_dist) and (cut_off_radius < nn3_dist):
                for i in range(2):
                    neighbors_of_interest += neighbor_octs[idx][neighbor_types[i]]
            elif (cut_off_radius > nn3_dist) and (cut_off_radius < nn4_dist):
                for i in range(3):
                    neighbors_of_interest += neighbor_octs[idx][neighbor_types[i]]
            elif (cut_off_radius > nn4_dist) and (cut_off_radius < 6):
                for i in range(4):
                    neighbors_of_interest += neighbor_octs[idx][neighbor_types[i]]

            for n1 in neighbors_of_interest:
                for n2 in neighbors_of_interest:
                    if (n1!=n2) and (n2 in neighbor_octs[n1]["nn1"]):
                        common_neighbors = np.intersect1d(neighbor_octs[n1]["nn1"],neighbor_octs[n2]["nn1"]) 
                        relevant_common_neighbors = [x for x in common_neighbors if x in neighbors_of_interest]
                        for c in relevant_common_neighbors:
                            proposed_triplet = list(np.sort([n1, n2, c]))
                            if proposed_triplet not in Triplets[idx]:
                                Triplets[idx].append(proposed_triplet)

        return Triplets
    
    @staticmethod
    def Indices(structure):

        """
        Method takes a structure as and arguement and returns a dictionary of tetrahedral, octahedral and oxygen 
        site indices in the supercell.
        """

        indices = defaultdict(list)

        for i in range(len(structure)):
            num = len(structure[i].species)
            if num==3:
                indices['tet'].append(i)
            elif num==5:
                indices['oct'].append(i)
            elif num==1:
                indices['o2'].append(i)

        return indices
    
    @staticmethod
    def NN_Calculator(structure, tet_oct_ind):

        """
        Method takes a structure as and arguement to calculate nearest neighbors for each cation site in the structure.
        """
        
        nn_dist = 1.81865

        nn_to_tet = defaultdict(list)
        nn_to_oct = defaultdict(list)
        nns = defaultdict(list)

        for tet_ind in tet_oct_ind['tet']:
            for oct_ind in tet_oct_ind['oct']:
                if abs(structure[tet_ind].distance(structure[oct_ind]) - nn_dist) <= 0.2:
                    nn_to_tet[tet_ind].append(oct_ind)
                    nn_to_oct[oct_ind].append(tet_ind)
                    nns[oct_ind].append(tet_ind)
                    nns[tet_ind].append(oct_ind)

        return nns
    
    @staticmethod
    def Oct_Oct_Neighbors(structure, tet_oct_ind):

        """
        Method takes a structure as and arguement to calculate the 1st to 4th nearest neighbor 
        octahedral sites for a given octahedral site.
        """

        neighbor_octs = defaultdict(list)

        for oct_ind1 in tet_oct_ind['oct']:
            neighbor_octs[oct_ind1] = {"nn1":[],"nn2":[],"nn3":[],"nn4":[]}
            for oct_ind2 in tet_oct_ind['oct']:
                if oct_ind2!=oct_ind1:
                    dist = structure[oct_ind1].distance(structure[oct_ind2])
                    if (dist < 3):
                        neighbor_octs[oct_ind1]["nn1"].append(oct_ind2)
                    elif (dist > 4) and ( dist < 5):
                        neighbor_octs[oct_ind1]["nn2"].append(oct_ind2)
                    elif (dist > 5) and ( dist < 5.5):
                        neighbor_octs[oct_ind1]["nn3"].append(oct_ind2)
                    elif (dist > 5.5) and ( dist < 6):
                        neighbor_octs[oct_ind1]["nn4"].append(oct_ind2)

        return neighbor_octs 