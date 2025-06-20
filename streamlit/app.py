import os
import json
import uuid
import re
import logging
from databricks.sdk import WorkspaceClient
import pandas as pd
import numpy as np

import sys
import distro
import subprocess

import rdkit
from rdkit.Chem import MolFromSmiles, AllChem, Draw
from typing import List
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.ERROR)
print(sys.version) #3.11
print(distro.info()) #Ubuntu 22.04 jammy

# try:
#     # no apt, dpkg, conda, cmake
#     # subprocess.run(["tar", "xvf", "data.tar"], check=True)
#     # result = subprocess.run(["dpkg -i "], shell=True, check=True)
#     # print(result.stdout)
#     subprocess.run('ls -al "$LD_LIBRARY_PATH/x86_64-linux-gnu"', check=True, shell=True)
# except subprocess.CalledProcessError as e:
#     print(f"Error installing libxrender1: {e}")

st.set_page_config(page_title="DrugBank Similarity Search", layout="wide")

# Configuration variables
#VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX_NAME")
VECTOR_SEARCH_INDEX_NAME = "yen.qsar.drugbank_vs"
col_display = ['id', 'name', 'smiles', 'molecular_weight']
col_vector = ['ECFP']
col_simscore = 'score'


# Initialize Databricks SDK client
workspace_client = WorkspaceClient()


def get_ecfp(mol: rdkit.Chem.rdchem.Mol, radius: int=3) -> np.array:
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    return fpgen.GetFingerprintAsNumPy(mol)


def multi_smiles_to_list(smiles: str) -> List[str]:
    _mols = smiles.split(",")
    smile_list = [m.strip() for m in _mols]
    return smile_list

def smiles2vector(smiles: str) -> np.array:
    try:
        mol = MolFromSmiles(smiles)
        fp_vec = get_ecfp(mol)
        return fp_vec.tolist()
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None


def run_vector_search(smiles: str) -> List:
    prompt_vector = smiles2vector(smiles)
    if prompt_vector is None:
        return "Failed to generate embeddings for the prompt."

    try:
        query_result = workspace_client.vector_search_indexes.query_index(
            index_name=VECTOR_SEARCH_INDEX_NAME,
            columns=col_display,
            query_vector=prompt_vector,
            num_results=3,
        )
        return pd.DataFrame(query_result.result.data_array,
                            columns=col_vector.append(col_simscore))
    except Exception as e:
        logging.error(f"Error during vector search: {e}")
        return f"Error during vector search: {e}"


def generate_random_dataframe():
    # Generate a random DataFrame
    df = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))
    return df


def display_mol(smiles: str) -> str:
    import mols2grid
    mol = MolFromSmiles(smiles)
    im = mols2grid.display([mol])
    html = im.data
    #print(html)
    return html


def mol2svg(smiles: str) -> str:
    smiles = multi_smiles_to_list(smiles)
    im = Draw.MolsToGridImage(
        [MolFromSmiles(mol) for mol in smiles],
        molsPerRow=5,
        legends=smiles,
        subImgSize=(200, 200),
        useSVG=True,
        returnPNG=False
    )
    # remove xml tag
    return re.sub(r"^<\?.+\?>\s*", "", im)


st.markdown("# DrugBank similarity search")
query_input = st.text_input("Enter SMILES")

if st.button("Search"):
    if query_input:
        try:
            # If using mol2grid
            # Display seed molecule
            #html_output = display_mol(query_input)
            #st.components.v1.html(html_output, height=10000)

            # If using MolsToGridImage
            img = mol2svg(query_input)
            st.image(img, caption='Molecule Grid', use_container_width=True)
            # Display most similar molecules
            pandas_output = run_vector_search(query_input[0])

            st.dataframe(pandas_output)
        except Exception as e:
            st.error(f"Error processing SMILES: {e}")
    else:
        st.warning("Please enter a SMILES string")
