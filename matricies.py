import os
import numpy as np
import random

"""gets scatac and scrna from the Data folder"""
def get_matricies():
    data_folder = os.path.join(os.getcwd(), 'Data/')
    scatac_feat_path = os.path.join(data_folder, 'scatac_feat.npy')
    scrna_feat_path = os.path.join(data_folder, 'scrna_feat.npy')
    scatac_feat = np.load(scatac_feat_path)
    scrna_feat = np.load(scrna_feat_path)
    
    #Shapes are 1047x19 and 1047x10
    #print(scatac_feat)
   # print(scrna_feat)
    return scatac_feat, scrna_feat

"""Returns a list containing random paired elements (no index). this should be updated to get a random numpy state and randomize from that state"""
def random_match(m1, m2):
    #could be done with just numpy matricies
    rows = min(len(m1), len(m2))
    if rows == 0:
        return None
    
    paired_results = np.empty((rows, 2), dtype=object)
    m1 = m1.tolist()
    m2 = m2.tolist()
    for i in range(rows):
        id1 = random.randint(0, len(m1) - 1)
        id2 = random.randint(0, len(m2) - 1)
        paired_results[i, 0] = m1.pop(id1)
        paired_results[i, 1] = m2.pop(id2)
        
    return paired_results


# def Zsc_normalized(m1):
#     log_m1 = np.log2(m1 + 1)
#     means = np.mean(log_m1, axis=0)
#     stds = np.std(log_m1, axis=0)
#     z_scores = (log_m1 - means) / stds
    

def run():
    print("hello world")

