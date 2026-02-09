from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.analysis.local_env import VoronoiNN

from ase.atoms import Atoms
from ase.io import read
from ase import visualize
from ase.visualize.plot import plot_atoms
from collections.abc import Sequence

from .constants import METALS
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import json
import numpy as np
import io
import base64


def cif_string_to_atoms(cif_string, type='struct'):
    struct = CifParser.from_str(cif_string).parse_structures(primitive=True)[0]
    if type == 'atoms':
        return AseAtomsAdaptor.get_atoms(struct)
    else:
        return struct


def atoms_to_cif_string(atoms):
    assert type(atoms) in [Atoms, Structure, Slab]
    
    if type(atoms) == Atoms:
        return CifWriter(struct=AseAtomsAdaptor.get_structure(atoms)).__str__()
    else:
        return CifWriter(struct=atoms).__str__()

def sanity_check(str):
    try:
        atoms = read(str)
    except ValueError:
        return False

    return True
    
def load_pickle(path):
    with open(path,'rb') as fr:
        return pickle.load(fr)

def save_pickle(path, obj):
    with open(path,'wb') as fw:
        pickle.dump(obj, fw)
        
def load_json(path):
    with open(path,'rb') as fr:
        return json.load(fr)

def save_json(path, obj):
    with open(path,'wb') as fw:
        json.dump(obj, fw)
        
def view(atoms):
    if isinstance(atoms, list):
        atoms = [AseAtomsAdaptor.get_atoms(atom) if isinstance(atom, Structure) else atom for atom in atoms]
        visualize.view(atoms)
    else:
        visualize.view(AseAtomsAdaptor.get_atoms(atoms) if isinstance(atoms, Structure) else atoms)
        
def get_attempt_indices(base_dir: Path, name, n_attempts):
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        int(p.name) for p in run_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]

    start = max(existing) + 1 if existing else 0
    return [start + i for i in range(n_attempts)]
        
def sort_index_cw(atoms, center_idx, neighbor_indices):
    c = atoms[center_idx].position[:2]  
    items = []
    for idx in neighbor_indices:
        v = atoms[idx].position[:2] - c 
        angle = np.arctan2(v[1], v[0]) 
        items.append((idx, v, angle))
    def transform(angle):
        return (np.pi/2 - angle) % (2*np.pi)
    items.sort(key=lambda x: transform(x[2]))
    return [x[0] for x in items]

def get_vnn_idx(atoms):
    
    struct = AseAtomsAdaptor.get_structure(atoms)
    vnn = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=10) #tight
    
    center_idx = [idx for idx, atom in enumerate(atoms) if atom.symbol in METALS][0]
    ligand_idx = [idx for idx, atom in enumerate(atoms) if atom.tag == 2]
    fnn_info = vnn.get_nn_info(struct,center_idx)
    fnn_idx = sort_index_cw(atoms, center_idx, [info['site_index'] for info in fnn_info if info['site_index'] not in ligand_idx])
    
    snn_info = []
    for idx in fnn_idx:
        snn_info.extend(vnn.get_nn_info(struct,idx))
    snn_idx = sort_index_cw(atoms, center_idx, [info['site_index'] for info in snn_info if info['site_index'] not in [center_idx]+ligand_idx+fnn_idx])
    
    tnn_info = []
    for idx in snn_idx:
        tnn_info.extend(vnn.get_nn_info(struct,idx))
    tnn_idx = sort_index_cw(atoms, center_idx, [info['site_index'] for info in tnn_info if info['site_index'] not in [center_idx]+ligand_idx+fnn_idx+snn_idx])
    return center_idx, fnn_idx, snn_idx, tnn_idx

def get_vnn_positions(atoms):
    center_idx, fnn_idx, snn_idx, tnn_idx = get_vnn_idx(atoms)
    center_pos = atoms[center_idx].position
    fnn_pos = atoms[fnn_idx].get_positions()
    snn_pos = atoms[snn_idx].get_positions()
    tnn_pos = atoms[tnn_idx].get_positions()

    return center_pos, fnn_pos, snn_pos, tnn_pos

def is_empty(atoms, position):
    position = np.array(position)
    positions = atoms.get_positions()
    return all(np.linalg.norm(positions-position,axis=-1) > 0.5)
    
def make_figure(atoms):
    fig, ax = plt.subplots(1, 2)
    plot_atoms(atoms,ax=ax[0])
    ax[0].set_axis_off()
    
    plot_atoms(atoms,ax=ax[1],rotation='270x,90y')
    ax[1].set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_bytes = buf.getvalue()
    return image_bytes


def get_slabs(atoms:Atoms) -> tuple:
    '''
    Remove adsorbates (tagged by 2) from catalyst
    
    Arg:
        atoms      ase.atoms.Atoms
         
    Returns:
        slabs      Surface without adsorbates
        ads_idx    indicies of adsorbate atoms
        ads_pos    position of adsorbate atoms    
    
    '''
    
    ads_idx = [i for i in range(len(atoms)) if atoms[i].tag == 2]
    slabs = atoms.copy()
    del slabs[[ads_idx]]
    
    return slabs
    
def get_adsorbates_positions(atoms):
    positions = np.array([atoms.get_positions(wrap=True)[i] for i, atom in enumerate(atoms) if atom.tag==2])
    argsort = np.argsort([i[2] for i in positions])
    return positions[argsort]

def get_active_string(atoms):
    assert 2 in atoms.get_tags(), "Need adsorbate atoms tagged as 2"

    slabs = get_slabs(atoms)
    ads_pos = get_adsorbates_positions(atoms)

    slabs += Atoms('U', positions=[ads_pos[0]])
    vnn = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=10)
    U_idx = [i for i in range(len(slabs)) if slabs[i].symbol=='U'][0]
    
    struct = AseAtomsAdaptor.get_structure(slabs)
    nn_info = vnn.get_nn_info(struct, n=U_idx)
    return __get_coordination_string(nn_info) 

def __get_coordination_string(nn_info):
    '''
    This helper function takes the output of the `VoronoiNN.get_nn_info` method
    and gives you a standardized coordination string.

    Arg:
        nn_info     The output of the
                    `pymatgen.analysis.local_env.VoronoiNN.get_nn_info` method.
    Returns:
        coordination    A string indicating the coordination of the site
                        you fed implicitly through the argument, e.g., 'Cu-Cu-Cu'
    '''
    coordinated_atoms = [neighbor_info['site'].species_string
                         for neighbor_info in nn_info
                         if neighbor_info['site'].species_string != 'U']
    coordination = '-'.join(sorted(coordinated_atoms))
    if len(coordinated_atoms) == 1:
        pos = " (Top)"
    elif len(coordinated_atoms) == 2:
        pos = " (Bridge)"
    elif len(coordinated_atoms) >= 3:
        pos = " (Hollow)"
        
    return coordination + pos   