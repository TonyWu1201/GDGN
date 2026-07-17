import pandas as pd
import deepchem as dc
from rdkit import Chem
import numpy as np
import pickle
from tqdm import tqdm

drugs = pd.read_csv("data/raw/drug_structures/compound_cid_smiles.csv")

# 创建一次featurizer实例，避免重复创建
featurizer = dc.feat.ConvMolFeaturizer()

drug_feats = {}

for drug in tqdm(drugs.itertuples(), total=drugs.shape[0]):
    
    try:
        # 转换SMILES到分子对象
        molecule = Chem.MolFromSmiles(drug.smiles)
        if molecule is None:
            print(f"Warning: Could not parse SMILES for {drug.cid}: {drug.smiles}")
            continue
            
        # 特征化分子
        mol_objects = featurizer.featurize([molecule])
        
        if len(mol_objects) > 0:
            mol_object = mol_objects[0]
            features = mol_object.atom_features
            degree_list = mol_object.deg_list
            adj_list = mol_object.canon_adj_list
            
            drug_feats[drug.cid] = [features, adj_list, degree_list]
        else:
            print(f"Warning: No features generated for {drug.cid}")
            
    except Exception as e:
        print(f"Error processing {drug.cid}: {str(e)}")
        continue

with open("data/processed/drug_structures/feats.pkl", "wb") as handle:
    pickle.dump(drug_feats, handle)