import json

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import BRICS, RWMol
from rdkit.Chem import rdMolDescriptors as rdDesc
from torch_geometric.data import Data
from tqdm import tqdm


TARGET_COLUMNS = [
    'parp1',
    'fa7',
    '5ht1b',
    'braf',
    'jak2',
    'qed',
    'sa',
    'amlodipine_mpo',
    'fexofenadine_mpo',
    'osimertinib_mpo',
    'perindopril_mpo',
    'ranolazine_mpo',
    'sitagliptin_mpo',
    'zaleplon_mpo',
]


def cut_smiles_with_brics_multiple(smiles, bonds_to_cut):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    rw_mol = RWMol(mol)
    dummy_map_num = 1

    for start_idx, end_idx in bonds_to_cut:
        rw_mol.RemoveBond(start_idx, end_idx)

        dummy1 = Chem.Atom(0)
        dummy2 = Chem.Atom(0)
        dummy1.SetAtomMapNum(dummy_map_num)
        dummy2.SetAtomMapNum(dummy_map_num)

        rw_mol.AddAtom(dummy1)
        rw_mol.AddAtom(dummy2)

        dummy1_idx = rw_mol.GetNumAtoms() - 2
        dummy2_idx = rw_mol.GetNumAtoms() - 1

        rw_mol.AddBond(start_idx, dummy1_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(end_idx, dummy2_idx, Chem.BondType.SINGLE)

    fragments = Chem.GetMolFrags(rw_mol, asMols=True)
    return [Chem.MolToSmiles(frag, isomericSmiles=True) for frag in fragments]


def smiles_breaker(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    original_bonds = set((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds())
    frag_mol = BRICS.BreakBRICSBonds(mol)
    remaining_bonds = set((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in frag_mol.GetBonds())
    cut_bonds = original_bonds - remaining_bonds
    return cut_smiles_with_brics_multiple(smiles, cut_bonds)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f'input {x} not in allowable set {allowable_set}')
    return [x == item for item in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == item for item in allowable_set]


def get_atom_features(atom, stereo, features, explicit_h=False):
    possible_atoms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
    ])
    atom_features += [int(item) for item in list(f'{features:06b}')]

    if not explicit_h:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception:
        atom_features += [False, False, atom.HasProp('_ChiralityPossible')]

    return np.array(atom_features)


def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
    ])
    return np.array(bond_feats)


def get_graph_from_frag(smiles, idx=0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = rdDesc.GetFeatureInvariants(mol)
    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()
    for atom_idx, label in stereo:
        chiral_centers[atom_idx] = label

    node_features = []
    edge_features = []
    bonds = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        node_features.append(get_atom_features(atom, chiral_centers[atom_idx], features[atom_idx]))
        for neighbor_idx in range(mol.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
            if bond is not None:
                bonds.append([atom_idx, neighbor_idx])
                edge_features.append(get_bond_features(bond))

    atom_feats = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(np.array(bonds), dtype=torch.long).T
    edge_feats = torch.tensor(np.array(edge_features), dtype=torch.float)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    return Data(x=atom_feats, edge_index=edge_index, edge_attr=edge_feats, idx=idx, smiles=canonical)


def build_dataset(df):
    processed = []
    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        smiles = row['smiles']
        graph = get_graph_from_frag(smiles, idx)
        if graph is None:
            continue
        frag_list = [get_graph_from_frag(item) for item in smiles_breaker(smiles)]
        frag_list = [frag for frag in frag_list if frag is not None]
        if len(frag_list) == 0:
            raise ValueError(
                f"Molecule at row {idx} produced no valid fragments: {smiles}"
            )
        value = {target: row[target] for target in TARGET_COLUMNS}
        value['smiles'] = smiles
        processed.append((graph, frag_list, value))
    return processed


def main():
    df = pd.read_csv('data/zinc250k.csv')
    with open('data/valid_idx_zinc250k.json') as handle:
        test_idx = json.load(handle)
    train_idx = [idx for idx in range(len(df)) if idx not in test_idx]
    train = build_dataset(df.iloc[train_idx])
    test = build_dataset(df.iloc[test_idx])
    torch.save((train, test), 'data/zinc250k_frag.pt')


if __name__ == '__main__':
    main()
