from tdc.multi_pred import DDI
import pandas as pd
import numpy as np
from tdc.utils import get_label_map
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect,  GetHashedTopologicalTorsionFingerprintAsBitVect
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect, GetErGFingerprint

labels = get_label_map(name = 'DrugBank', task = 'DDI')

data = DDI(name = 'DrugBank')
split = data.get_split()

drug1_rdfp = []
drug2_rdfp = []

drug1_ap = []
drug2_ap = []

drug1_tt = []
drug2_tt = []

drug1_morgan = []
drug2_morgan = []

drug1_avalon = []
drug2_avalon = []

drug1_erg = []
drug2_erg = []

y_train = []

count = 0.0

for x in range(len(split['test']['Drug1'])):
    mol1 = Chem.MolFromSmiles(split['test']['Drug1'][x])
    mol2 = Chem.MolFromSmiles(split['test']['Drug2'][x])
    if mol1 == None or mol2 == None:
        pass
    else:
        #y_train.append(split['test']['Y'][x])
        #fingerprint_rdk = RDKFingerprint(mol1)
        #fingerprint_rdk_np = np.array(fingerprint_rdk)
        #drug1_rdfp.append(fingerprint_rdk_np)

        #fingerprint_rdk = RDKFingerprint(mol2)
        #fingerprint_rdk_np = np.array(fingerprint_rdk)
        #drug2_rdfp.append(fingerprint_rdk_np)

        #atom_pair_FP = GetHashedAtomPairFingerprintAsBitVect(mol1, nBits=512)
        #fingerprint_AP_np = np.array(atom_pair_FP)
        #drug1_ap.append(fingerprint_AP_np)

        #atom_pair_FP = GetHashedAtomPairFingerprintAsBitVect(mol2, nBits=512)
        #fingerprint_AP_np = np.array(atom_pair_FP)
        #drug2_ap.append(fingerprint_AP_np)

        #tt_FP = GetHashedTopologicalTorsionFingerprintAsBitVect(mol1, nBits=512)
        #tt_FP_np = np.array(tt_FP)
        #drug1_tt.append(tt_FP_np)

        #tt_FP = GetHashedTopologicalTorsionFingerprintAsBitVect(mol2, nBits=512)
        #tt_FP_np = np.array(tt_FP)
        #drug2_tt.append(tt_FP_np)

        #morgan_FP = GetMorganFingerprintAsBitVect(mol1, 2, nBits = 512)
        #morgan_FP_np = np.array(morgan_FP)
        #drug1_morgan.append(morgan_FP_np)

        #morgan_FP = GetMorganFingerprintAsBitVect(mol2, 2, nBits = 512)
        #morgan_FP_np = np.array(morgan_FP)
        #drug2_morgan.append(morgan_FP_np)

        #avalon_FP = GetAvalonFP(mol1, nBits = 512)
        #avalon_FP_np = np.array(avalon_FP)
        #drug1_avalon.append(avalon_FP_np)

        #avalon_FP = GetAvalonFP(mol2, nBits = 512)
        #avalon_FP_np = np.array(avalon_FP)
        #drug2_avalon.append(avalon_FP_np)

        #erg_FP = GetErGFingerprint(mol1)
        #erg_FP_np = np.array(erg_FP)
       #drug1_erg.append(erg_FP_np)

        #erg_FP = GetErGFingerprint(mol2)
        #erg_FP_np = np.array(erg_FP)
        #drug2_erg.append(erg_FP_np)
        count += 1
        print(count)


#output = np.hstack((drug1_rdfp, drug2_rdfp))
#np.savetxt("rdk_fingerprint_test.txt", output, fmt='%i')

#output = np.hstack((drug1_ap, drug2_ap))
#np.savetxt("ap_fingerprint_test.txt", output, fmt='%i')

#output = np.hstack((drug1_tt, drug2_tt))
#np.savetxt("tt_fingerprint_test.txt", output, fmt='%i')

#output = np.hstack((drug1_morgan, drug2_morgan))
#np.savetxt("morgan_fingerprint_test.txt", output, fmt='%i')

#output = np.hstack((drug1_avalon, drug2_avalon))
#np.savetxt("avalon_fingerprint_test.txt", output, fmt='%i')

#output = np.hstack((drug1_erg, drug2_erg))
#np.savetxt("erg_fingerprint_test.txt", output, fmt='%i')

#np.savetxt("y_train.txt", y_train, fmt='%i')