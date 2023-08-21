from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from rdkit import Chem
import numpy as np
from argparse import ArgumentParser
import torch
from torch_geometric.data import Data

from typing import Dict, List, Union

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--smiles", type=str, default="O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5", help="")
    args = parser.parse_args()
    return args


def ReorderCanonicalRankAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order


def smiles2graph(smiles_string, removeHs=True, reorder_atoms=False) -> Dict[str, np.ndarray]:
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph 


def create_pyg_graph(graph: Dict, additional_node_files = [], additional_edge_files = []) -> Data:
    """
    Converts graph to pytorch geometric Data object
    :input: graph object
    :return: pytorch geometric Data object
    """
    g = Data()
    g.num_nodes = graph['num_nodes']
    g.edge_index = torch.from_numpy(graph['edge_index'])

    del graph['num_nodes']
    del graph['edge_index']

    if graph['edge_feat'] is not None:
        g.edge_attr = torch.from_numpy(graph['edge_feat'])
        del graph['edge_feat']

    if graph['node_feat'] is not None:
        g.x = torch.from_numpy(graph['node_feat'])
        del graph['node_feat']

    for key in additional_node_files:
        g[key] = torch.from_numpy(graph[key])
        del graph[key]

    for key in additional_edge_files:
        g[key] = torch.from_numpy(graph[key])
        del graph[key]
    
    if graph['graph_label'] is not None:
        g.y = torch.tensor(graph['graph_label'])
        del graph['graph_label']

    return g


def create_pyg_graph_from_smiles(smiles_string: Union[str, List], removeHs=True, reorder_atoms=False, labels=None, additional_node_files = [], additional_edge_files = []) -> List[Data]:
    """
    Converts list of SMILES string to list of pytorch geometric Data object
    :input: SMILES string (str, list)
    :return: pytorch geometric Data object
    """
    if isinstance(smiles_string, str):
        smiles_string = [smiles_string]
    
    graphs = []
    pyg_graphs = []
    for i, smiles in enumerate(smiles_string):
        graph = smiles2graph(smiles, removeHs, reorder_atoms)
        if labels is not None:
            graph['graph_label'] = labels[i].view(1,-1).to(torch.float32)
        graphs.append(graph)
        pyg_graphs.append(create_pyg_graph(graph, additional_node_files = [], additional_edge_files = []))
    
    return pyg_graphs

"""
def main():
    args = parse_args()
    smiles_string = args.smiles
    graphs = create_pyg_graph_from_smiles(smiles_string)
    print(graphs)
"""