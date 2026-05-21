import functools
from itertools import product
from typing import Dict, Set, Union
from selfies.constants import ELEMENTS, INDEX_ALPHABET

_DEFAULT_CONSTRAINTS = {
    "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1,
    "B": 3, "B+1": 2, "B-1": 4,
    "O": 2, "O+1": 3, "O-1": 1,
    "N": 3, "N+1": 4, "N-1": 2,
    "C": 4, "C+1": 5, "C-1": 3,
    "P": 5, "P+1": 6, "P-1": 4,
    "S": 6, "S+1": 7, "S-1": 5,
    "?": 8
}

_PRESET_CONSTRAINTS = {
    "default": dict(_DEFAULT_CONSTRAINTS),
    "octet_rule": dict(_DEFAULT_CONSTRAINTS),
    "hypervalent": dict(_DEFAULT_CONSTRAINTS)
}
_PRESET_CONSTRAINTS["octet_rule"].update(
    {"S": 2, "S+1": 3, "S-1": 1, "P": 3, "P+1": 4, "P-1": 2}
)
_PRESET_CONSTRAINTS["hypervalent"].update(
    {"Cl": 7, "Br": 7, "I": 7, "N": 5}
)

_current_constraints = _PRESET_CONSTRAINTS["default"]


def get_preset_constraints(name: str) -> Dict[str, int]:
    """Returns the preset semantic constraints with the given name.

    Besides the aforementioned default constraints, :mod:`selfies` offers
    other preset constraints for convenience; namely, constraints that
    enforce the `octet rule <https://en.wikipedia.org/wiki/Octet_rule>`_
    and constraints that accommodate `hypervalent molecules
    <https://en.wikipedia.org/wiki/Hypervalent_molecule>`_.

    The differences between these constraints can be summarized as follows:

    .. table::
        :align: center
        :widths: auto

        +-----------------+-----------+---+---+-----+-----+---+-----+-----+
        |                 | Cl, Br, I | N | P | P+1 | P-1 | S | S+1 | S-1 |
        +-----------------+-----------+---+---+-----+-----+---+-----+-----+
        | ``default``     |     1     | 3 | 5 |  6  |  4  | 6 |  7  |  5  |
        +-----------------+-----------+---+---+-----+-----+---+-----+-----+
        | ``octet_rule``  |     1     | 3 | 3 |  4  |  2  | 2 |  3  |  1  |
        +-----------------+-----------+---+---+-----+-----+---+-----+-----+
        | ``hypervalent`` |     7     | 5 | 5 |  6  |  4  | 6 |  7  |  5  |
        +-----------------+-----------+---+---+-----+-----+---+-----+-----+

    :param name: the preset name: ``default`` or ``octet_rule`` or
        ``hypervalent``.
    :return: the preset constraints with the specified name, represented
        as a dictionary which maps atoms (the keys) to their bonding capacities
        (the values).
    """

    if name not in _PRESET_CONSTRAINTS:
        raise ValueError("unrecognized preset name '{}'".format(name))
    return dict(_PRESET_CONSTRAINTS[name])


def get_semantic_constraints() -> Dict[str, int]:
    """Returns the semantic constraints that :mod:`selfies` is currently
    operating on.

    :return: the current semantic constraints, represented as a dictionary
        which maps atoms (the keys) to their bonding capacities (the values).
    """

    global _current_constraints
    return dict(_current_constraints)


def set_semantic_constraints(
        bond_constraints: Union[str, Dict[str, int]] = "default"
) -> None:
    """Updates the semantic constraints that :mod:`selfies` operates on.

    If the input is a string, the new constraints are taken to be
    the preset named ``bond_constraints``
    (see :func:`selfies.get_preset_constraints`).

    Otherwise, the input is a dictionary representing the new constraints.
    This dictionary maps atoms (the keys) to non-negative bonding
    capacities (the values); the atoms are specified by strings
    of the form ``E`` or ``E+C`` or ``E-C``,
    where ``E`` is an element symbol and ``C`` is a positive integer.
    For example, one may have:

       * ``bond_constraints["I-1"] = 0``
       * ``bond_constraints["C"] = 4``

    This dictionary must also contain the special ``?`` key, which indicates
    the bond capacities of all atoms that are not explicitly listed
    in the dictionary.

    :param bond_constraints: the name of a preset, or a dictionary
        representing the new semantic constraints.
    :return: ``None``.
    """

    global _current_constraints

    if isinstance(bond_constraints, str):
        _current_constraints = get_preset_constraints(bond_constraints)

    elif isinstance(bond_constraints, dict):

        # error checking
        if "?" not in bond_constraints:
            raise ValueError("bond_constraints missing '?' as a key")

        for key, value in bond_constraints.items():

            # error checking for keys
            j = max(key.find("+"), key.find("-"))
            if key == "?":
                valid = True
            elif j == -1:
                valid = (key in ELEMENTS)
            else:
                valid = (key[:j] in ELEMENTS) and key[j + 1:].isnumeric()
            if not valid:
                err_msg = "invalid key '{}' in bond_constraints".format(key)
                raise ValueError(err_msg)

            # error checking for values
            if not (isinstance(value, int) and value >= 0):
                err_msg = "invalid value at " \
                          "bond_constraints['{}'] = {}".format(key, value)
                raise ValueError(err_msg)

        _current_constraints = dict(bond_constraints)

    else:
        raise ValueError("bond_constraints must be a str or dict")

    # clear cache since we changed alphabet
    get_semantic_robust_alphabet.cache_clear()
    get_bonding_capacity.cache_clear()


@functools.lru_cache()
def get_semantic_robust_alphabet() -> Set[str]:
    """Returns a subset of all SELFIES symbols that are constrained
    by :mod:`selfies` under the current semantic constraints.

    :return: a subset of all SELFIES symbols that are semantically constrained.
    """

    alphabet_subset = set()
    bonds = {"": 1, "=": 2, "#": 3 ,'\\': 1, '/': 1, '\\': 2, '/': 2 ,'\\': 3, '/': 3}

    # add atomic symbols
    for (a, c), (b, m) in product(_current_constraints.items(), bonds.items()):
        if (m > c) or (a == "?"):
            continue
        symbol = "[{}{}]".format(b, a)
        alphabet_subset.add(symbol)

    # add branch and ring symbols
    for i in range(1, 4):
        alphabet_subset.add("[Ring{}]".format(i))
        alphabet_subset.add("[=Ring{}]".format(i))
        alphabet_subset.add("[Branch{}]".format(i))
        alphabet_subset.add("[=Branch{}]".format(i))
        alphabet_subset.add("[#Branch{}]".format(i))

    alphabet_subset.update(INDEX_ALPHABET)

    return alphabet_subset


@functools.lru_cache()
def get_bonding_capacity(element: str, charge: int) -> int:
    """Returns the bonding capacity of a given atom, under the current
    semantic constraints.

    :param element: the element of the input atom.
    :param charge: the charge of the input atom.
    :return: the bonding capacity of the input atom.
    """

    key = element
    if charge != 0:
        key += "{:+}".format(charge)

    if key in _current_constraints:
        return _current_constraints[key]
    else:
        return _current_constraints["?"]

import selfies as sf
import numpy as np
import pandas as pd
import random
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MolFromSmiles, MolFromSmarts, MolToSmiles, MolToSmarts
from rdkit import RDLogger   
from difflib import SequenceMatcher

RDLogger.DisableLog('rdApp.*') 
rdBase.DisableLog('rdApp.error')

import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))

from SA_Score import sascorer
from NP_Score import npscorer

def SA_score_calc(mol):
    return sascorer.calculateScore(mol)

def selfies_encoding(smiles):
    #for individual SELFIES string
    smiles = MolToSmiles(MolFromSmiles(smiles),kekuleSmiles = True)
    selfie = sf.encoder(smiles).replace('.', '[.]')
    return selfie

def get_selfie_chars(selfie):
    selfies_char_list_pre = selfie[1:-1].split('][')
    selfies_char_list = []
    for selfies_element in selfies_char_list_pre:
        selfies_char_list.append('['+selfies_element+']')
    return selfies_char_list

def canonical_smiles(smiles):
    mol = MolFromSmiles(smiles)
    canon_smiles_string = MolToSmiles(mol , isomericSmiles=True, canonical=True)
    generated_mol = MolFromSmiles(canon_smiles_string)
    return generated_mol, canon_smiles_string

def substructure_preserver(generated_mol, substructure_smiles):
    sub_mol = MolFromSmiles(substructure_smiles)
    atom_count = sub_mol.GetNumAtoms()
    sub_mol = Chem.AddHs(sub_mol, onlyOnAtoms = [i for i in range(0,atom_count) if i!=12]) # atom number 12 is the methyl
    sub_smarts= MolToSmarts(sub_mol, isomericSmiles=True)
    sub_scaffold = MolFromSmarts(sub_smarts)
    if generated_mol.HasSubstructMatch(sub_scaffold,useChirality=True) == True:
        return True
    else: 
        return False # Molecule does not have substructure!



def mutate_selfie(selfie, substructure_smiles):

    chars_selfie = get_selfie_chars(selfie)
    initial_alphabet = get_semantic_robust_alphabet() # 69 possible alphabets #using the modified semantic alphabet to include the extra bonds
    # including the chiral centers
    initial_alphabet.update({"[/Br]",  "[/Cl]", "[/F]", "[/H]", "[/I]",  "[/O]"})
    initial_alphabet.update({"[\\Br]", "[\\Cl]", "[\\F]", "[\\H]", "[\\I]","[\\O]"})
    initial_alphabet.update({'[C@]','[C@@]','[\\C@]','[\\C@@]','[/C@]','[/C@@]' })
    initial_alphabet.update({'[C@H1]','[C@@H1]','[\\C@H1]','[\\C@@H1]','[/C@H1]','[/C@@H1]' })

    removal_charged_boron = set([ '[#B-1]','[#B+1]', '[=B+1]', '[=B-1]', '[B+1]', '[B-1]' ,  '[\\B-1]', '[/B-1]'])
    removal_charged_phosphorus = set(['[#P+1]', '[#P-1]', '[=P+1]', '[=P-1]', '[P-1]', '[P+1]',  '[\\P+1]', '[\\P-1]','[/P+1]', '[/P-1]'])
    removal_charged_sulphur = set (['[=S+1]', '[#S+1]', '[#S-1]' ,'[S+1]', '[S-1]', '[=S-1]',  '[\\S+1]', '[\\S-1]','[/S+1]', '[/S-1]'])
    removal_charged_carbon = set(['[#C+1]', '[#C-1]', '[=C+1]', '[=C-1]', '[C-1]', '[C+1]',  '[\\C+1]', '[\\C-1]','[/C+1]', '[/C-1]'])
    removal_charged_nitrogen = set(['[#N+1]', '[#N-1]', '[=N+1]', '[=N-1]', '[N-1]', '[N+1]',  '[\\N+1]','[/N+1]'])
    removal_charged_oxygen = set(['[#O+1]','[=O+1]', '[O+1]', '[O-1]',  '[\\O+1]','[/O+1]'])
    full_alphabet = list(initial_alphabet - removal_charged_boron - removal_charged_phosphorus - removal_charged_sulphur 
                         -removal_charged_carbon - removal_charged_nitrogen - removal_charged_oxygen)
    choices = ["Insert", "Replace", "Delete"]

    
    random_choice = np.random.choice(choices, 1)[0]
    if random_choice == "Insert":
        random_index = np.random.randint(len(chars_selfie) + 1)
        random_character = np.random.choice(full_alphabet, 1)[0]
        new_selfie_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]
        
    elif random_choice == "Replace":
        random_index = np.random.randint(len(chars_selfie))
        random_character = np.random.choice(full_alphabet, 1)[0]
        if random_index == 0:
            new_selfie_chars = [random_character] + chars_selfie[1:]
        else:
            new_selfie_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
            
    elif random_choice == "Delete":
        random_index = np.random.randint(len(chars_selfie))
        if random_index == 0:
            new_selfie_chars = chars_selfie[1:]
        else:
            new_selfie_chars = chars_selfie[:random_index]  + chars_selfie[random_index+1:]
            
    final_selfie = "".join(x for x in new_selfie_chars)
    smiles= sf.decoder(final_selfie)
    mol, canon_smiles = canonical_smiles(smiles)
    mol = Chem.AddHs(mol)
    present= substructure_preserver(mol, substructure_smiles)
    return final_selfie, canon_smiles, present


def mutate_selfie_terminal_10(selfie, substructure_smiles):

    chars_selfie = get_selfie_chars(selfie)
    initial_alphabet = get_semantic_robust_alphabet() # 69 possible alphabets #using the modified semantic alphabet to include the extra bonds
    # including the chiral centers
    initial_alphabet.update({"[/Br]",  "[/Cl]", "[/F]", "[/H]", "[/I]",  "[/O]"})
    initial_alphabet.update({"[\\Br]", "[\\Cl]", "[\\F]", "[\\H]", "[\\I]","[\\O]"})
    initial_alphabet.update({'[C@]','[C@@]','[\\C@]','[\\C@@]','[/C@]','[/C@@]' })
    initial_alphabet.update({'[C@H1]','[C@@H1]','[\\C@H1]','[\\C@@H1]','[/C@H1]','[/C@@H1]' })
    removal_charged_boron = set([ '[#B-1]','[#B+1]', '[=B+1]', '[=B-1]', '[B+1]', '[B-1]' ,  '[\\B-1]', '[/B-1]'])
    removal_charged_phosphorus = set(['[#P+1]', '[#P-1]', '[=P+1]', '[=P-1]', '[P-1]', '[P+1]',  '[\\P+1]', '[\\P-1]','[/P+1]', '[/P-1]'])
    removal_charged_sulphur = set (['[=S+1]', '[#S+1]', '[#S-1]' ,'[S+1]', '[S-1]', '[=S-1]',  '[\\S+1]', '[\\S-1]','[/S+1]', '[/S-1]'])
    removal_charged_carbon = set(['[#C+1]', '[#C-1]', '[=C+1]', '[=C-1]', '[C-1]', '[C+1]',  '[\\C+1]', '[\\C-1]','[/C+1]', '[/C-1]'])
    removal_charged_nitrogen = set(['[#N+1]', '[#N-1]', '[=N+1]', '[=N-1]', '[N-1]', '[N+1]',  '[\\N+1]','[/N+1]'])
    removal_charged_oxygen = set(['[#O+1]','[=O+1]', '[O+1]', '[O-1]',  '[\\O+1]','[/O+1]'])
    full_alphabet = list(initial_alphabet - removal_charged_boron - removal_charged_phosphorus - removal_charged_sulphur 
                         -removal_charged_carbon - removal_charged_nitrogen - removal_charged_oxygen)
    choices = ["Insert", "Replace", "Delete"]

    
    random_choice = np.random.choice(choices, 1)[0]
    if random_choice == "Insert":
        random_index =  np.random.randint(low = int(0.9 *len(chars_selfie)) , high = len(chars_selfie) + 1)
        random_character = np.random.choice(full_alphabet, 1)[0]
        new_selfie_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]
        
    elif random_choice == "Replace":
        random_index =  np.random.randint(low = int(0.9 *len(chars_selfie)) , high = len(chars_selfie))
        random_character = np.random.choice(full_alphabet, 1)[0]
        if random_index == 0:
            new_selfie_chars = [random_character] + chars_selfie[1:]
        else:
            new_selfie_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
            
    elif random_choice == "Delete":
        random_index =  np.random.randint(low = int(0.9 *len(chars_selfie)) , high = len(chars_selfie))
        if random_index == 0:
            new_selfie_chars = chars_selfie[1:]
        else:
            new_selfie_chars = chars_selfie[:random_index]  + chars_selfie[random_index+1:]
            
    final_selfie = "".join(x for x in new_selfie_chars)
    smiles= sf.decoder(final_selfie)
    mol, canon_smiles = canonical_smiles(smiles)
    mol = Chem.AddHs(mol)
    present= substructure_preserver(mol, substructure_smiles)
    return final_selfie, canon_smiles, present

def mutate_selfie_adaptive(selfie, substructure_smiles, checkers = []):

    #baf pattern checker

    chars_selfie = get_selfie_chars(selfie)
    initial_alphabet = get_semantic_robust_alphabet() # 69 possible alphabets #using the modified semantic alphabet to include the extra bonds
    # including the chiral centers
    initial_alphabet.update({"[/Br]",  "[/Cl]", "[/F]", "[/H]", "[/I]",  "[/O]"})
    initial_alphabet.update({"[\\Br]", "[\\Cl]", "[\\F]", "[\\H]", "[\\I]","[\\O]"})
    initial_alphabet.update({'[C@]','[C@@]','[\\C@]','[\\C@@]','[/C@]','[/C@@]' })
    initial_alphabet.update({'[C@H1]','[C@@H1]','[\\C@H1]','[\\C@@H1]','[/C@H1]','[/C@@H1]' })
    removal_charged_boron = set([ '[#B-1]','[#B+1]', '[=B+1]', '[=B-1]', '[B+1]', '[B-1]' ,  '[\\B-1]', '[/B-1]'])
    removal_charged_phosphorus = set(['[#P+1]', '[#P-1]', '[=P+1]', '[=P-1]', '[P-1]', '[P+1]',  '[\\P+1]', '[\\P-1]','[/P+1]', '[/P-1]'])
    removal_charged_sulphur = set (['[=S+1]', '[#S+1]', '[#S-1]' ,'[S+1]', '[S-1]', '[=S-1]',  '[\\S+1]', '[\\S-1]','[/S+1]', '[/S-1]'])
    removal_charged_carbon = set(['[#C+1]', '[#C-1]', '[=C+1]', '[=C-1]', '[C-1]', '[C+1]',   '[\\C+1]', '[\\C-1]','[/C+1]', '[/C-1]'])
    removal_charged_nitrogen = set(['[#N+1]', '[#N-1]', '[=N+1]', '[=N-1]', '[N-1]', '[N+1]',  '[\\N+1]','[/N+1]'])
    removal_charged_oxygen = set(['[#O+1]','[=O+1]', '[O+1]', '[O-1]',  '[\\O+1]','[/O+1]'])
    full_alphabet = list(initial_alphabet - removal_charged_boron - removal_charged_phosphorus - removal_charged_sulphur 
                         -removal_charged_carbon - removal_charged_nitrogen - removal_charged_oxygen)

    choices = ["Insert", "Replace", "Delete"]

    
    random_choice = np.random.choice(choices, 1)[0]
    removal_index = []
    if random_choice == "Insert":
        for x in checkers:
            starting_position  = len(get_selfie_chars(selfie.split(x)[0]))
            intermediate_position = starting_position + len(get_selfie_chars(x))
            removal_index += list(range(starting_position, intermediate_position))
            
        full_index = list(range(0, len(get_selfie_chars(selfie)) +1))
        final_index = [i for i in full_index if i not in removal_index]
        
    else:
        for x in checkers:
            starting_position  = len(get_selfie_chars(selfie.split(x)[0]))
            intermediate_position = starting_position + len(get_selfie_chars(x))
            removal_index += list(range(starting_position, intermediate_position))
            
        full_index = list(range(0, len(get_selfie_chars(selfie))))
        final_index = [i for i in full_index if i not in removal_index]
    
    
    if random_choice == "Insert":
        random_index =  np.random.choice(final_index,1)[0]
        random_character = np.random.choice(full_alphabet, 1)[0]
        new_selfie_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]
        
    elif random_choice == "Replace":
        random_index =  np.random.choice(final_index,1)[0]
        random_character = np.random.choice(full_alphabet, 1)[0]
        if random_index == 0:
            new_selfie_chars = [random_character] + chars_selfie[1:]
        else:
            new_selfie_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
            
    elif random_choice == "Delete":
        random_index =  np.random.choice(final_index,1)[0]
        if random_index == 0:
            new_selfie_chars = chars_selfie[1:]
        else:
            new_selfie_chars = chars_selfie[:random_index]  + chars_selfie[random_index+1:]
            
    final_selfie = "".join(x for x in new_selfie_chars)
    smiles= sf.decoder(final_selfie)
    mol, canon_smiles = canonical_smiles(smiles)
    mol = Chem.AddHs(mol)
    present= substructure_preserver(mol, substructure_smiles)
    return final_selfie, canon_smiles, present
        


def mutate_selfie_context_specific(selfie, substructure_smiles):
    checkers = ['[C][Branch1][C][F][Branch1][C][F][F]',
         '[Branch1][C][F]',
         '[=C][C][Branch1]',
         '[=C][C][Branch1][=Branch2]',
         '[C][=C][Ring2]',
         '[C][=C][C][Branch1]',
         '[C][=C][C][=C][Branch2]',
         '[C][=Branch1][C]',
         '[=N][C][=C][C][Branch1]',
         '[C][=Branch1][C][=O]',
         '[=Branch2][C][=C][C][=C]',
         '[=C][Ring1][=Branch1]',
         '[C][=N][C][=C][C]',
         '[N][C][=N][C][=C][C][Branch1][=Branch2][C][=C][C][=C][N][=C][Ring1][=Branch1][=N][Ring1][N]',
         '[C][Branch1][C]',
         '[=Branch1][C][=O]',
         '[Ring1][=Branch1][=N][Ring1][N]',
         '[Branch1][=Branch2][C][=C]',
         '[C][=C][C][=C]',
         '[C][=C][C]',
         '[C][=C][Branch2]',
         '[C][=C][N][=C][Ring1]',
         '[C][Branch1][=Branch2]',
         '[C][C][=C][C][=C][Branch2]',
         '[C][C][=C][C][=C]']


    chars_selfie = get_selfie_chars(selfie)
    initial_alphabet = get_semantic_robust_alphabet() # 69 possible alphabets #using the modified semantic alphabet to include the extra bonds
    # including the chiral centers
    initial_alphabet.update({"[/Br]",  "[/Cl]", "[/F]", "[/H]", "[/I]",  "[/O]"})
    initial_alphabet.update({"[\\Br]", "[\\Cl]", "[\\F]", "[\\H]", "[\\I]","[\\O]"})
    initial_alphabet.update({'[C@]','[C@@]','[\\C@]','[\\C@@]','[/C@]','[/C@@]' })
    initial_alphabet.update({'[C@H1]','[C@@H1]','[\\C@H1]','[\\C@@H1]','[/C@H1]','[/C@@H1]' })
    removal_charged_boron = set([ '[#B-1]','[#B+1]', '[=B+1]', '[=B-1]', '[B+1]', '[B-1]' ,  '[\\B-1]', '[/B-1]'])
    removal_charged_phosphorus = set(['[#P+1]', '[#P-1]', '[=P+1]', '[=P-1]', '[P-1]', '[P+1]',  '[\\P+1]', '[\\P-1]','[/P+1]', '[/P-1]'])
    removal_charged_sulphur = set (['[=S+1]', '[#S+1]', '[#S-1]' ,'[S+1]', '[S-1]', '[=S-1]',  '[\\S+1]', '[\\S-1]','[/S+1]', '[/S-1]'])
    removal_charged_carbon = set(['[#C+1]', '[#C-1]', '[=C+1]', '[=C-1]', '[C-1]', '[C+1]',  '[\\C+1]', '[\\C-1]','[/C+1]', '[/C-1]'])
    removal_charged_nitrogen = set(['[#N+1]', '[#N-1]', '[=N+1]', '[=N-1]', '[N-1]', '[N+1]',  '[\\N+1]','[/N+1]'])
    removal_charged_oxygen = set(['[#O+1]','[=O+1]', '[O+1]', '[O-1]',  '[\\O+1]','[/O+1]'])
    full_alphabet = list(initial_alphabet - removal_charged_boron - removal_charged_phosphorus - removal_charged_sulphur 
                         -removal_charged_carbon - removal_charged_nitrogen - removal_charged_oxygen)
    
    choices = ["Insert", "Replace", "Delete"]

    
    random_choice = np.random.choice(choices, 1)[0]
    removal_index = []
    if random_choice == "Insert":
        for x in checkers:
            starting_position  = len(get_selfie_chars(selfie.split(x)[0]))
            intermediate_position = starting_position + len(get_selfie_chars(x))
            full_index = list(range(0, len(get_selfie_chars(selfie)) +1))
            removal_index += list(range(starting_position, intermediate_position))
    
        final_index = [i for i in full_index if i not in removal_index]
    else:
        for x in checkers:
            starting_position  = len(get_selfie_chars(selfie.split(x)[0]))
            intermediate_position = starting_position + len(get_selfie_chars(x))
            full_index = list(range(0, len(get_selfie_chars(selfie))))
            removal_index += list(range(starting_position, intermediate_position))
    
        final_index = [i for i in full_index if i not in removal_index]
    
    
    if random_choice == "Insert":
        random_index =  np.random.choice(final_index,1)[0]
        random_character = np.random.choice(full_alphabet, 1)[0]
        new_selfie_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]
        
    elif random_choice == "Replace":
        random_index =  np.random.choice(final_index,1)[0]
        random_character = np.random.choice(full_alphabet, 1)[0]
        if random_index == 0:
            new_selfie_chars = [random_character] + chars_selfie[1:]
        else:
            new_selfie_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
            
    elif random_choice == "Delete":
        random_index =  np.random.choice(final_index,1)[0]
        if random_index == 0:
            new_selfie_chars = chars_selfie[1:]
        else:
            new_selfie_chars = chars_selfie[:random_index]  + chars_selfie[random_index+1:]
            
    final_selfie = "".join(x for x in new_selfie_chars)
    smiles= sf.decoder(final_selfie)
    mol, canon_smiles = canonical_smiles(smiles)
    mol = Chem.AddHs(mol)
    present= substructure_preserver(mol, substructure_smiles)
    return final_selfie, canon_smiles, present
    
def get_mutated_SELFIES(selfies_ls, substructure_smiles, num_mutations, checkers, scaffold_check = 'adaptive' ): 
    for i in range(num_mutations): 
        total_scaffold_selfies = 0
        selfie_ls_mut_ls = []
        for selfie in selfies_ls: 
            if scaffold_check== 'yes':
                mutated_selfie, mutated_smiles, present = mutate_selfie_context_specific(selfie, substructure_smiles)
            elif scaffold_check== 'no':
                mutated_selfie, mutated_smiles, present = mutate_selfie(selfie, substructure_smiles)
            elif scaffold_check =='terminal10':
                mutated_selfie, mutated_smiles, present = mutate_selfie_terminal_10(selfie, substructure_smiles)
            elif scaffold_check== 'adaptive':
                mutated_selfie, mutated_smiles, present = mutate_selfie_adaptive(selfie, substructure_smiles, checkers)
            else:
                print("No Choice made")
                
            selfie_ls_mut_ls.append(mutated_selfie)
            if present == True:
                total_scaffold_selfies+=1 
        selfies_ls = selfie_ls_mut_ls.copy()
        total_selfies = len(selfies_ls)
        
        
    smiles_ls_mut_ls = []
    unique_smiles = set()
    
    for selfie in selfies_ls:
        _, canon_smiles = canonical_smiles(sf.decoder(selfie))
        smiles_ls_mut_ls.append(canon_smiles)
        unique_smiles.add(canon_smiles)

    unique_scaffold_smiles = set()
    for z in unique_smiles:
        mol = MolFromSmiles(z)
        mol = Chem.AddHs(mol)
        present= substructure_preserver(mol, substructure_smiles)
        if present == True:
            unique_scaffold_smiles.add(z)
            
    return  selfies_ls, smiles_ls_mut_ls,total_scaffold_selfies, total_selfies, unique_smiles, unique_scaffold_smiles


def get_radicals(selfie):
    smile = sf.decoder(selfie)
    mol = MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    sum_radicals = 0
    for atom in mol.GetAtoms():
        sum_radicals+= atom.GetNumRadicalElectrons()
    return sum_radicals

def get_scaffold(selfie):
    smile = sf.decoder(selfie)
    mol = MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    present= substructure_preserver(mol, substructure_smiles)
    if present == True:
        scaffold = 1
    else:
        scaffold = 0 
    return scaffold


def get_sascore(selfie):
    smile = sf.decoder(selfie)
    mol = MolFromSmiles(smile)
    sa_score = SA_score_calc(mol)
    return sa_score


def selfies_decoding(selfie):
    smile = sf.decoder(selfie)
    mol = MolFromSmiles(smile)
    canon_smiles_string = MolToSmiles(mol , isomericSmiles=True, canonical=True)
    return canon_smiles_string


def get_fp_scores(smiles_back, target_smiles): 
    #ECFP4 
    
    smiles_scores = []
    target_mol  = MolFromSmiles(target_smiles)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius= 2, includeChirality = True, countSimulation=True)
    fp_target = fp_gen.GetFingerprint(target_mol) 
    generated_mol = MolFromSmiles(smiles_back)
    fp_generated = fp_gen.GetFingerprint(generated_mol)
    score  = TanimotoSimilarity(fp_generated, fp_target)
    return round(score,3)


# Function to find longest common substring
def longest_Substring(s1,s2):

  # Create sequence matcher object
  seq_match = SequenceMatcher(None,s1,s2, autojunk=False) 
  
  # Find longest matching substring
  match = seq_match.find_longest_match(0, len(s1), 0, len(s2))

  # If match found, return substring
  if (match.size!=0):  
    return (s1[match.a: match.a + match.size])
  
  # Else no match found
  else:
    return ('')

def get_pattern(best_population):
    max_molecules = min(len(best_population), 50)
    new_scaffolds = set()
    for i in range(max_molecules):
        length = 20 #or whatever arbitrary high number
        char_length = 20 #or whatever arbitrary high number
        old_scaffolds = set()
        random_one = random.randrange(len(best_population))
        random_two = random.randrange(len(best_population))
        if random_two == random_one:
            random_two = random.randrange(len(best_population))
        first_string = best_population[random_one]
        second_string = best_population[random_two]
        a = longest_Substring(first_string, second_string)
        char_length = len(a)
        while length > 2 and char_length > 0: # to avoid  0 as non-matches 
            if a[0] != '[':
                if len(a.split('[',1)) > 1:
                    a = '[' + a.split('[',1)[1]
                else:
                    a = '[' + a
            if a[-1] != ']':
                a= a.rsplit(']',1)[0] + ']'
            length = len(list(sf.split_selfies(a)))
            if length > 2 and char_length > 4:
                old_scaffolds.update([a])
                first_half = first_string.split(a,1)[0]
                remainder = first_string.split(a,1)[1]
                first_string = ''.join([first_half, remainder])
            a = longest_Substring(first_string, second_string)
            char_length = len(a)
            
        new_scaffolds.update(old_scaffolds)
        
    final_scaffolds = list(new_scaffolds)
    for scaffold in list(new_scaffolds):
        for i in best_population:
            if scaffold not in i:
                final_scaffolds.remove(scaffold)
                break
    return final_scaffolds
    
noncan_substructure = 'CC1=CC=C(C=C1NC2=NC=CC(C3=CC=CN=C3)=N2)NC(C4=CC(C(F)(F)F)=C(C=C4)C)=O'
_, substructure_smiles =  canonical_smiles(noncan_substructure)

smiles = 'O=C(NC1=CC=C(C)C(NC2=NC=CC(C3=CC=CN=C3)=N2)=C1)C4=CC(C(F)(F)F)=C(C=C4)CN5C[C@H](CC5)N(C)C'
can_mol, can_smiles = canonical_smiles(smiles)
starting_selfie = [selfies_encoding(can_smiles)]
similarity_goal = can_smiles

num_generations = 100
    
with open('GA_SA_500k.txt', 'w') as content:
    content.write("Optimization starting")
    
convergence_dataset= pd.DataFrame(columns = ['Generation', 'Num_unique_smiles_with_scaffold_and_sascore_no_radicals', 'Missing'])

# optimization SA +  expansion
all_selfies_mutations = np.array([])
unique_smiles_mutations = set()
overall_unique_scaffold = 0
best_scaffold_best_score_unique_smiles = []
gen_size = 1000000
population = np.random.choice(starting_selfie, size=gen_size)
full_dataset = pd.DataFrame(columns = ['Generation', 'selfies', 'smiles', 'sa_score','scaffold'])
checkers = []

for gen in range(0, num_generations):
    print(gen)
    print(len(population))
    sa_score_list = []
    scaffold_list = []
    radical_list = []
    best_scaffold_best_score_unique_selfies = []
    best_scaffold_best_score_unique_selfies_duplicate = []
    expanded_best_scaffold_worst_score_selfies =[]
    selected_worst_scaffold_selfies = []
    smiles_ls_mut_ls = []

    for x in population:
        score= get_sascore(x)
        sa_score_list.append(score)
    sa_score_list = np.array(sa_score_list)

    for x in population:
        scaffold = get_scaffold(x)
        scaffold_list.append(scaffold)
    scaffold_list = np.array(scaffold_list)

    for x in population:
        smile = selfies_decoding(x)
        smiles_ls_mut_ls.append(smile)

    for x in population:
        radicals = get_radicals(x)
        radical_list.append(radicals)
    radical_list = np.array(radical_list)
   

    new_rows = pd.DataFrame({'Generation': np.repeat(gen,len(population)), 'selfies' : population, 'smiles': smiles_ls_mut_ls, 
                             'sa_score': sa_score_list, 'scaffold' : scaffold_list, 'radicals': radical_list})
    full_dataset = pd.concat([full_dataset, new_rows], ignore_index=True)

    if gen==0:
        selfie_ls_mut_ls, _, _, _, _, _ = get_mutated_SELFIES(population.copy(),substructure_smiles, num_mutations = 1,checkers = checkers,
                                                              scaffold_check = 'adaptive')
        population = np.array(list(selfie_ls_mut_ls))
        
    if gen!=0:

        #selfies aren't unique so needs to be based on the smiles
        best_scaffold_best_score_unique_smiles = list(set(full_dataset['smiles'][(full_dataset['Generation'] == gen) 
        & (full_dataset['scaffold'] == 1) 
         & (full_dataset['sa_score'] <=3.4)
                                                          & (full_dataset['radicals'] ==0.0)]))


        
        if (2*len(best_scaffold_best_score_unique_smiles)) < gen_size:
            
            expand_size = (gen_size) - len(best_scaffold_best_score_unique_smiles)

            for x in best_scaffold_best_score_unique_smiles:
                best_scaffold_best_score_unique_selfies.append(selfies_encoding(x))
                
            best_scaffold_best_score_unique_selfies_duplicate  = np.random.choice(np.array(best_scaffold_best_score_unique_selfies),
                                                                                  size = expand_size,
                                                                                  replace = True)
                
            if (gen+1)%10 == 0:
                checkers = get_pattern(best_scaffold_best_score_unique_selfies)
                
            selfie_ls_mut_ls_best_scaffold_best_score_duplicate,_, _, _, _, _ = get_mutated_SELFIES(best_scaffold_best_score_unique_selfies_duplicate.copy(),substructure_smiles, 
                                                                                                    num_mutations = 1, checkers= checkers,scaffold_check = 'adaptive')
    
    
            selfie_ls_mut_ls = best_scaffold_best_score_unique_selfies +selfie_ls_mut_ls_best_scaffold_best_score_duplicate
            population = np.array(selfie_ls_mut_ls)
            
        else: 
            expand_size = (gen_size) - len(best_scaffold_best_score_unique_smiles)
            similarity_list  = []
            
            for x in best_scaffold_best_score_unique_smiles:
                best_scaffold_best_score_unique_selfies.append(selfies_encoding(x))
                similarity = get_fp_scores(x,similarity_goal)
                similarity_list.append(similarity)
                
            similarity_list = np.array(similarity_list)
            similarity_sort = similarity_list[np.argsort(similarity_list)]
            scaffold_sort = np.array(best_scaffold_best_score_unique_selfies)[np.argsort(similarity_list)]
            similarity90_100 = scaffold_sort[list(np.where((similarity_sort >= np.percentile(similarity_sort, 90)))[0])]
            
            best_scaffold_best_score_unique_selfies_duplicate  = np.random.choice(np.array(similarity90_100), size = expand_size,
                                                                                  replace = True)

            
            if (gen+1)%10 == 0:
                checkers = get_pattern(best_scaffold_best_score_unique_selfies)
                
            selfie_ls_mut_ls_best_scaffold_best_score_duplicate,_, _, _, _, _ = get_mutated_SELFIES(best_scaffold_best_score_unique_selfies_duplicate.copy(),
                                                                                                    substructure_smiles, num_mutations = 1, checkers = checkers, 
                                                                                                    scaffold_check = 'adaptive')
            
            selfie_ls_mut_ls = selfie_ls_mut_ls_best_scaffold_best_score_duplicate + best_scaffold_best_score_unique_selfies
            
            population = np.array(selfie_ls_mut_ls)

        #there should be no missing population, but this can happen if num_mutations is set to higher than 1
        new_population = []
        missing = 0
        
        for x in selfie_ls_mut_ls:
            smile = sf.decoder(x)
            mol = MolFromSmiles(smile)
            if len(smile) == 0:
                missing +=1
            else:
                new_population.append(x)
            
        missing_pop = np.random.choice(np.array(new_population), size= missing, replace=True)

        population = np.array(new_population + list(missing_pop))
        
        convergence_dataset.loc[len(convergence_dataset)] = [gen, len(best_scaffold_best_score_unique_smiles), missing]
            

full_dataset.to_csv("GA_SA_500k.csv", index = False)
convergence_dataset.to_csv("GA_SA_500k_convergence.csv", index = False)

print("Final Checkers:")
print(checkers)

with open('GA_SA_500k.txt', 'a') as content:
    content.write("Results Saved")







