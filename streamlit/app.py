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
from rdkit.Chem import MolFromSmiles, AllChem
import pubchempy as pcp
from typing import List
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.ERROR)
print(sys.version)  # 3.11
print(distro.info())  # Ubuntu 22.04 jammy

# try:
#     # no apt, dpkg, conda, cmake
#     # subprocess.run(["tar", "xvf", "data.tar"], check=True)
#     # result = subprocess.run(["dpkg -i "], shell=True, check=True)
#     print(result.stdout)
#     subprocess.run('ls -al "$LD_LIBRARY_PATH/x86_64-linux-gnu"', check=True, shell=True)
# except subprocess.CalledProcessError as e:
#     print(f"Error installing libxrender1: {e}")

st.set_page_config(page_title="ChemSearch", layout="wide")

# Configuration variables
# VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX_NAME")
VECTOR_SEARCH_INDEX_NAME = "yen.qsar.drugbank_vs"
col_display = ["id", "name", "smiles", "molecular_weight"]
col_vector = ["ECFP"]
col_simscore = ["score"]


# Initialize Databricks SDK client
workspace_client = WorkspaceClient()


def get_ecfp(mol: rdkit.Chem.rdchem.Mol, radius: int = 3) -> np.array:
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    return fpgen.GetFingerprintAsNumPy(mol)


def multi_inputs_to_list(inputs: str) -> List[str]:
    _inputs = inputs.split(",")
    input_list = [m.strip() for m in _inputs]
    return input_list


def input_to_pcpCompound(input: str) -> pcp.Compound:
    if input.isdigit():
        # if all integer => CID
        cid = int(input)
        try:
            # look up in pubchem
            cpd = pcp.Compound.from_cid(cid)
            # smiles = cpd.isomeric_smiles
            return cpd
        except Exception as e:
            raise Exception(f"Failed to look up CID {cid} in PubChem: {e}")
    else:
        try:
            # if can be converted to a Mol => SMILES
            mol = MolFromSmiles(input, sanitize=False)
            cpd = pcp.get_compounds(input, "smiles")[0]
            return cpd
        except:
            # Not CID or SMILES => name
            # then lookup SMILES by name
            try:
                cpd = pcp.get_compounds(input, "name")[0]
                return cpd
            except Exception as e:
                raise Exception(
                    f"Failed to look up SMILES/name {input} in PubChem: {e}"
                )


def smiles2vector(smiles: str) -> np.array:
    try:
        mol = MolFromSmiles(smiles)
    except Exception as e:
        raise Exception(f"{e}. Not a valid SMILES string.")

    try:
        fp_vec = get_ecfp(mol)
        return fp_vec.tolist()
    except Exception as e:
        raise Exception(f"Error generating embeddings: {e}")


def run_vector_search(
    smiles: str,
    score_threshold: float = 0.0,
    num_results: int = 3,
    filters_json: str = None,
) -> List:
    prompt_vector = smiles2vector(smiles)
    if prompt_vector is None:
        return "Failed to generate embeddings for the prompt."

    try:
        query_result = workspace_client.vector_search_indexes.query_index(
            index_name=VECTOR_SEARCH_INDEX_NAME,
            columns=col_display,
            query_vector=prompt_vector,
            #            query_type="HYBRID",
            num_results=num_results,
            score_threshold=score_threshold,
            filters_json=filters_json,
        )
        return pd.DataFrame(
            query_result.result.data_array, columns=col_display + col_simscore
        )
    except Exception as e:
        logging.error(f"Error during vector search: {e}")
        return f"Error during vector search: {e}"


def display_mol(smiles: str) -> str:
    import mols2grid

    mol = MolFromSmiles(smiles)
    im = mols2grid.display([mol])
    html = im.data
    return html


def mol2svg(smiles: List[str], labels: List[str] = None) -> str:
    from rdkit.Chem import Draw

    im = Draw.MolsToGridImage(
        [MolFromSmiles(mol) for mol in smiles],
        molsPerRow=5,
        # TODO: 'list' object is not callable
        legends=labels,
        subImgSize=(200, 200),
        useSVG=True,
        returnPNG=False,
    )
    # remove xml tag
    return re.sub(r"^<\?.+\?>\s*", "", im)


st.markdown("# ChemSearch")

# Create a container for the layout
with st.container():
    # Row 1: SMILES input and similarity slider
    col1, col2 = st.columns([4, 1])

    with col2:
        st.markdown("### Filters")
        top_k = st.number_input(
            "Number of hits",
            min_value=1,
            max_value=None,
            value=3,
            step=1,
            help="Show this number of most similar molecules",
        )

        min_similarity = 0
        # min_similarity = st.number_input(
        #     "Minimum similarity",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=0.03,
        #     step=0.01,
        #     help="Show molecules with similarity score above this threshold",
        # )

        mw_range = st.slider(
            "Molecular weight",
            min_value=0,
            max_value=5000,
            value=(200, 1000),
            step=10,
            help="Show compounds with molecular weight within this range",
        )

    with col1:
        query_input = st.text_input("Enter SMILES")
        search_button = st.button("Search")

        if search_button:
            if query_input:
                input_list = multi_inputs_to_list(query_input)
                first_input = input_list[0]
                if len(input_list) > 1:
                    st.info(
                        "Multiple inputs detected. Only the first one will be used."
                    )

                try:
                    cpd = input_to_pcpCompound(first_input)
                    first_smile = cpd.isomeric_smiles
                    name = cpd.synonyms[0]
                    url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cpd.cid}"

                    # Put smile and name in a list
                    smiles_list = [first_smile]
                    name_list = [name]

                    # If using mol2grid
                    # Display seed molecule
                    # html_output = display_mol(query_input)
                    # st.components.v1.html(html_output, height=10000)

                    # If using MolsToGridImage
                    print(smiles_list, name)
                    img = mol2svg(smiles_list, name_list)
                    print(img)
                    st.image(img, use_container_width=True)
                    st.markdown(
                        f'<a href="{url}" target="_blank" style="color: blue; text-decoration: underline;">PubChem URL</a>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Missing libxrender1 to render molecules. {e}")

            else:
                st.warning("Please enter a SMILES string")


if search_button:
    if query_input:
        try:
            mw_low, mw_upp = mw_range
            filters_str = f"""{{"molecular_weight >": {str(mw_low)}, 
            "molecular_weight <=": {str(mw_upp)}}}"""
            pandas_output = run_vector_search(
                first_smile,
                score_threshold=min_similarity,
                num_results=top_k,
                filters_json=filters_str,
            )

        except Exception as e:
            st.error(f"Error with vector search: {e}")

        try:
            # If using PandasTools to render molecules
            st.markdown("#### With libxrender1 to render molecules")
            from rdkit.Chem import PandasTools

            PandasTools.AddMoleculeColumnToFrame(pandas_output, "smiles", "Molecule")
            pandas_html = re.sub(
                r"^<table", '<table width="100%"', pandas_output.to_html(escape=False)
            )
            st.markdown(pandas_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error with PandaTools: {e}")

        try:
            # For debugging with vanilla pandas
            st.markdown("#### Without libxrender1 to render molecules")
            st.dataframe(pandas_output)

        except Exception as e:
            st.error(f"Error: {e}")
