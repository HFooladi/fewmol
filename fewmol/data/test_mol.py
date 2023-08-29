import pytest
import torch
from torch_geometric.data import Data
from fewmol.data.mol import smiles2graph, create_pyg_graph, create_pyg_graph_from_smiles


@pytest.fixture
def smiles():
    s1 = "CCO"
    s2 = "CCC"
    return [s1, s2]


class TestMol:
    def test_smiles2graph(self, smiles):
        graph = smiles2graph(smiles[0])
        assert isinstance(graph, dict)
        assert graph["num_nodes"] == 3
        assert graph["node_feat"].shape == (3, 9)
        assert graph["edge_index"].shape == (2, 4)
        assert graph["edge_feat"].shape == (4, 3)

    def test_create_pyg_graph(self, smiles):
        graph = smiles2graph(smiles[0])
        pyg_graph = create_pyg_graph(graph)
        assert isinstance(pyg_graph, Data)
        assert pyg_graph.num_nodes == 3
        assert pyg_graph.x.shape == (3, 9)
        assert pyg_graph.edge_index.shape == (2, 4)
        assert pyg_graph.edge_attr.shape == (4, 3)

    def test_create_pyg_graph_from_smiles(self, smiles):
        labels = [torch.tensor(0), torch.tensor(1)]

        assert isinstance(labels[0], torch.Tensor), "labels should be a list of torch.Tensor"

        pyg_graphs = create_pyg_graph_from_smiles(smiles[0])
        pyg_graphs_list = create_pyg_graph_from_smiles(smiles)
        pyg_graphs_with_labels = create_pyg_graph_from_smiles(smiles, labels=labels)

        assert isinstance(pyg_graphs, list)
        assert isinstance(pyg_graphs_list, list)
        assert isinstance(pyg_graphs_with_labels, list)

        assert len(pyg_graphs) == 1
        assert len(pyg_graphs_list) == 2
        assert len(pyg_graphs_with_labels) == 2

        assert isinstance(pyg_graphs[0], Data)
        assert isinstance(pyg_graphs_list[0], Data)
        assert isinstance(pyg_graphs_with_labels[0], Data)

        assert pyg_graphs_with_labels[0].y == labels[0]
