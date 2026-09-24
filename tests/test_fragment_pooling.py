import unittest

import torch
from torch_geometric.data import Data

from train_mfrag import DockingDataset, collate_mfrag_batch, mean_pool_frag_embeddings


def make_single_atom_graph():
    return Data(
        x=torch.ones((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
    )


class MeanPoolFragmentEmbeddingsTest(unittest.TestCase):
    def test_pools_each_positive_fragment_group(self):
        embeddings = torch.tensor(
            [
                [1.0, 3.0],
                [3.0, 5.0],
                [8.0, 10.0],
            ]
        )

        pooled = mean_pool_frag_embeddings(embeddings, [2, 1])

        expected = torch.tensor([[2.0, 4.0], [8.0, 10.0]])
        torch.testing.assert_close(pooled, expected)

    def test_rejects_zero_fragment_group(self):
        embeddings = torch.tensor([[1.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "must be positive"):
            mean_pool_frag_embeddings(embeddings, [1, 0])

    def test_rejects_group_size_mismatch(self):
        embeddings = torch.tensor([[1.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "sum to 2"):
            mean_pool_frag_embeddings(embeddings, [2])

    def test_dataset_rejects_zero_fragment_sample(self):
        dataset = DockingDataset(
            [(make_single_atom_graph(), [], {"parp1": 1.0})],
            target="parp1",
        )

        with self.assertRaisesRegex(ValueError, "has no fragments"):
            dataset[0]

    def test_collate_rejects_zero_fragment_sample(self):
        sample = (
            make_single_atom_graph(),
            [],
            0,
            torch.tensor(1.0),
            torch.tensor(1.0),
        )

        with self.assertRaisesRegex(ValueError, "batch positions \[0\]"):
            collate_mfrag_batch([sample])


if __name__ == "__main__":
    unittest.main()
