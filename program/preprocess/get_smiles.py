import pubchempy as pcp
from tqdm import tqdm
import pandas as pd

drugs = pd.read_csv("data/common_drugs_pubchem.csv")

results = []
for drug in tqdm(drugs.itertuples(), total=drugs.shape[0]):
    cid = drug.CID
    compound = pcp.Compound.from_cid(cid)
    smiles = compound.smiles
    results.append([cid, smiles])

print(results)
results_dataframe = pd.DataFrame(results, columns=['cid', 'smiles'])
results_dataframe.to_csv("data/raw/drug_structures/compound_cid_smiles.csv")