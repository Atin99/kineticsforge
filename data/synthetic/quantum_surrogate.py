import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eigh

class TightBindingHamiltonian:
    def __init__(self, orbitals_per_site=9): # s, p (3), d (5)
        self.n_orb = orbitals_per_site
        
    def build_hamiltonian(self, coordinates, element_types, hopping_params, onsite_energies):
        n_atoms = coordinates.shape[0]
        H = torch.zeros(n_atoms * self.n_orb, n_atoms * self.n_orb, dtype=torch.complex64)
        
        # On-site energies (Diagonal)
        for i in range(n_atoms):
            el = element_types[i]
            for o in range(self.n_orb):
                H[i*self.n_orb + o, i*self.n_orb + o] = onsite_energies[el][o]
                
        # Hopping parameters (Off-diagonal)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_vec = coordinates[j] - coordinates[i]
                r_mag = torch.norm(r_vec)
                
                if r_mag < 3.5: # Cutoff radius
                    el_i = element_types[i]
                    el_j = element_types[j]
                    
                    # Slater-Koster two-center approximation
                    t_ij = self._slater_koster(el_i, el_j, r_vec, r_mag, hopping_params)
                    
                    for o1 in range(self.n_orb):
                        for o2 in range(self.n_orb):
                            H[i*self.n_orb + o1, j*self.n_orb + o2] = t_ij[o1, o2]
                            H[j*self.n_orb + o2, i*self.n_orb + o1] = torch.conj(t_ij[o1, o2])
        return H

    def _slater_koster(self, el_i, el_j, r_vec, r_mag, params):
        # Direction cosines
        l = r_vec[0] / r_mag
        m = r_vec[1] / r_mag
        n = r_vec[2] / r_mag
        
        t = torch.zeros(self.n_orb, self.n_orb, dtype=torch.complex64)
        # s-s hopping
        V_ss_sigma = params.get((el_i, el_j, 'ss_sigma'), -1.0) * torch.exp(-r_mag / 1.5)
        t[0, 0] = V_ss_sigma
        
        # s-p hopping
        V_sp_sigma = params.get((el_i, el_j, 'sp_sigma'), 1.5) * torch.exp(-r_mag / 1.5)
        t[0, 1] = l * V_sp_sigma
        t[0, 2] = m * V_sp_sigma
        t[0, 3] = n * V_sp_sigma
        t[1, 0] = -l * V_sp_sigma
        t[2, 0] = -m * V_sp_sigma
        t[3, 0] = -n * V_sp_sigma
        
        # Extended s,p,d Slater-Koster tables are massive. 
        # Procedurally injecting d-band correlations for Transition Metals.
        V_dd_sigma = params.get((el_i, el_j, 'dd_sigma'), -2.0) * torch.exp(-r_mag / 1.2)
        V_dd_pi = params.get((el_i, el_j, 'dd_pi'), 1.0) * torch.exp(-r_mag / 1.2)
        V_dd_delta = params.get((el_i, el_j, 'dd_delta'), 0.0)
        
        # Simplified d-band projection matrix
        for d1 in range(4, 9):
            for d2 in range(4, 9):
                t[d1, d2] = V_dd_sigma * (l**2) + V_dd_pi * (m**2 + n**2) + V_dd_delta * 0.1
                
        return t

class DensityFunctionalTheorySurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.tb = TightBindingHamiltonian()
        self.onsite = nn.ParameterDict({
            'Mn': nn.Parameter(torch.randn(9)),
            'Fe': nn.Parameter(torch.randn(9)),
            'O': nn.Parameter(torch.randn(9)),
            'Na': nn.Parameter(torch.randn(9))
        })
        self.U_hubbard = nn.Parameter(torch.tensor([4.5])) # Hubbard U for d-electrons
        
    def solve_electronic_structure(self, coordinates, element_types):
        hopping = {
            ('Mn', 'O', 'sp_sigma'): 1.8,
            ('Fe', 'O', 'sp_sigma'): 1.6,
            ('Mn', 'Mn', 'dd_sigma'): -1.2,
            ('Fe', 'Fe', 'dd_sigma'): -1.1
        }
        
        H = self.tb.build_hamiltonian(coordinates, element_types, hopping, self.onsite)
        
        # Add Hubbard U correction (DFT+U)
        n_atoms = coordinates.shape[0]
        for i in range(n_atoms):
            if element_types[i] in ['Mn', 'Fe']:
                for d in range(4, 9):
                    H[i*self.tb.n_orb + d, i*self.tb.n_orb + d] += self.U_hubbard * 0.5 # Mean field approx
                    
        # Diagonalize
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        
        # Fermi energy calculation (assuming neutral charge, counting valence electrons)
        valence = {'Mn': 7, 'Fe': 8, 'O': 6, 'Na': 1}
        total_electrons = sum(valence[el] for el in element_types)
        fermi_idx = int(total_electrons / 2) # Spin degenerate
        
        e_fermi = eigenvalues[fermi_idx]
        band_gap = eigenvalues[fermi_idx + 1] - e_fermi if fermi_idx + 1 < len(eigenvalues) else 0.0
        
        # Total energy (sum of occupied states)
        total_energy = torch.sum(eigenvalues[:fermi_idx]) * 2.0
        
        return total_energy, band_gap, e_fermi, eigenvalues

class MassiveSyntheticDataEngine:
    def __init__(self):
        self.dft_surrogate = DensityFunctionalTheorySurrogate()
        
    def generate_lattice_variations(self, base_lattice, variations=10000):
        datasets = []
        for _ in range(variations):
            strain = torch.eye(3) + torch.randn(3, 3) * 0.05
            strain = 0.5 * (strain + strain.T) # Symmetric strain tensor
            
            coords = torch.matmul(base_lattice['coords'], strain)
            
            energy, gap, e_f, dos = self.dft_surrogate.solve_electronic_structure(coords, base_lattice['elements'])
            
            # Relate quantum energy to macroscopic OCV
            F = 96485.0
            voltage = - (energy / F) * 1000.0 # Approximation
            
            datasets.append({
                'strain_tensor': strain.detach().numpy(),
                'formation_energy': energy.item(),
                'band_gap': gap.item(),
                'fermi_level': e_f.item(),
                'ocv_proxy': voltage.item(),
                'density_of_states': dos.detach().numpy()
            })
        return datasets
