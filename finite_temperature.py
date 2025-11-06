"""
Finite Temperature BEC: Zaremba-Nikuni-Griffin (ZNG) Theory

This module implements the ZNG theory for finite temperature Bose-Einstein condensates,
which couples the condensate dynamics with the thermal cloud.

Theory:
-------
The system is described by:
1. Condensate: Modified GP equation with thermal cloud coupling
2. Thermal cloud: Bogoliubov-de Gennes quasi-particle distribution

Condensate equation:
    iℏ ∂ψ_c/∂t = [H_0 + 2g(n_c + 2ñ)]ψ_c - iR_{12}

where:
    n_c = |ψ_c|² (condensate density)
    ñ = thermal cloud density
    R_{12} = collision integral (condensate-thermal coupling)
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import simps
import matplotlib.pyplot as plt


class BogoliubovSolver1D:
    """
    Solves the Bogoliubov-de Gennes equations for quasi-particle excitations

    The BdG equations:
        [H_0 + 2gn]u_j + gn*v_j = E_j u_j
        -[H_0 + 2gn]v_j - gn*u_j = E_j v_j

    where u_j, v_j are quasi-particle amplitudes and E_j are excitation energies
    """

    def __init__(self, x, k, V, psi_c, m=1.0, hbar=1.0, g=1.0):
        """
        Initialize Bogoliubov solver

        Parameters:
        -----------
        x : array
            Position grid
        k : array
            Momentum grid
        V : array
            External potential
        psi_c : array
            Condensate wave function
        m, hbar, g : float
            Physical parameters
        """
        self.x = x
        self.k = k
        self.V = V
        self.psi_c = psi_c
        self.m = m
        self.hbar = hbar
        self.g = g
        self.N = len(x)
        self.dx = x[1] - x[0]

        # Condensate density
        self.n_c = np.abs(psi_c)**2

        # Quasi-particle spectrum (to be computed)
        self.energies = None
        self.u_modes = None
        self.v_modes = None

    def solve_spectrum(self, n_modes=None):
        """
        Solve for Bogoliubov quasi-particle spectrum

        Parameters:
        -----------
        n_modes : int
            Number of modes to compute (default: all)

        Returns:
        --------
        energies : array
            Excitation energies
        u_modes, v_modes : arrays
            Quasi-particle amplitudes
        """
        if n_modes is None:
            n_modes = self.N

        # Build BdG Hamiltonian in momentum space for efficiency
        # This is a simplified version using local density approximation

        # Kinetic energy
        T_k = (self.hbar**2 * self.k**2) / (2 * self.m)

        # Mean-field energy in momentum space
        n_mean = np.mean(self.n_c)
        epsilon_k = T_k + 2 * self.g * n_mean

        # Bogoliubov dispersion (homogeneous approximation)
        c = np.sqrt(self.g * n_mean / self.m)  # Speed of sound
        xi = self.hbar / (self.m * c)  # Healing length

        # Bogoliubov spectrum: E_k = sqrt(epsilon_k^2 - (g*n)^2)
        E_k = np.sqrt(epsilon_k * (epsilon_k + 2 * self.g * n_mean))

        # Sort by energy
        idx = np.argsort(E_k)
        E_k = E_k[idx]
        k_sorted = self.k[idx]
        epsilon_sorted = epsilon_k[idx]

        # Compute u and v amplitudes
        u_k = np.sqrt((epsilon_sorted + E_k) / (2 * E_k))
        v_k = np.sign(k_sorted) * np.sqrt((epsilon_sorted - E_k) / (2 * E_k))

        # Store results
        self.energies = E_k[:n_modes]

        # Transform to position space
        self.u_modes = np.zeros((n_modes, self.N), dtype=complex)
        self.v_modes = np.zeros((n_modes, self.N), dtype=complex)

        for i in range(n_modes):
            # Create mode in momentum space
            mode_k = np.zeros(self.N, dtype=complex)
            mode_k[idx[i]] = 1.0

            # u and v in position space
            self.u_modes[i] = ifft(u_k[i] * mode_k) * np.sqrt(self.N)
            self.v_modes[i] = ifft(v_k[i] * mode_k) * np.sqrt(self.N)

        return self.energies, self.u_modes, self.v_modes

    def thermal_density(self, temperature, mu=None, n_modes=None):
        """
        Compute thermal cloud density at given temperature

        Parameters:
        -----------
        temperature : float
            Temperature (in units where k_B = 1)
        mu : float
            Chemical potential (computed if not provided)
        n_modes : int
            Number of modes to include

        Returns:
        --------
        n_thermal : array
            Thermal cloud density ñ(x)
        """
        if self.energies is None:
            self.solve_spectrum(n_modes)

        if mu is None:
            mu = self.g * np.mean(self.n_c)

        kT = temperature
        n_thermal = np.zeros(self.N)

        # Sum over quasi-particle modes
        for i, E_i in enumerate(self.energies):
            if E_i > 0 and kT > 0:
                # Bose-Einstein distribution
                n_i = 1.0 / (np.exp(E_i / kT) - 1.0)

                # Contribution to thermal density
                n_thermal += n_i * np.abs(self.v_modes[i])**2

        return n_thermal

    def condensate_fraction(self, temperature, n_total=None):
        """
        Compute condensate fraction at given temperature

        Parameters:
        -----------
        temperature : float
            Temperature
        n_total : float
            Total particle density (computed from condensate if not given)

        Returns:
        --------
        f_c : float
            Condensate fraction N_c / N_total
        """
        n_c_integrated = simps(self.n_c, self.x)
        n_th = self.thermal_density(temperature)
        n_th_integrated = simps(n_th, self.x)

        if n_total is None:
            n_total = n_c_integrated + n_th_integrated

        return n_c_integrated / n_total


class ZNGSolver1D:
    """
    1D Zaremba-Nikuni-Griffin Solver for finite temperature BEC

    Couples condensate and thermal cloud dynamics
    """

    def __init__(self, x_min, x_max, N, m=1.0, hbar=1.0, g=1.0, temperature=0.0):
        """
        Initialize ZNG solver

        Parameters:
        -----------
        x_min, x_max : float
            Spatial boundaries
        N : int
            Number of grid points
        m, hbar, g : float
            Physical parameters
        temperature : float
            Temperature (in units where k_B = 1)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.m = m
        self.hbar = hbar
        self.g = g
        self.temperature = temperature

        # Grids
        self.x = np.linspace(x_min, x_max, N)
        self.dx = (x_max - x_min) / (N - 1)
        self.k = 2 * np.pi * fftfreq(N, self.dx)
        self.T_k = (self.hbar**2 * self.k**2) / (2 * self.m)

        # Potential
        self.V = np.zeros(N)

        # Condensate wave function
        self.psi_c = np.zeros(N, dtype=complex)

        # Thermal cloud density
        self.n_thermal = np.zeros(N)

        # Collision rate (simplified)
        self.gamma = 0.1  # Collision rate parameter

    def set_potential(self, V):
        """Set external potential"""
        if callable(V):
            self.V = V(self.x)
        else:
            self.V = np.array(V)

    def set_initial_state(self, psi0):
        """Set initial condensate wave function"""
        if callable(psi0):
            self.psi_c = psi0(self.x)
        else:
            self.psi_c = np.array(psi0, dtype=complex)
        self.normalize()

    def normalize(self):
        """Normalize condensate wave function"""
        norm = np.sqrt(simps(np.abs(self.psi_c)**2, self.x))
        if norm > 0:
            self.psi_c /= norm

    def update_thermal_cloud(self, n_modes=50):
        """
        Update thermal cloud density using Bogoliubov theory

        Parameters:
        -----------
        n_modes : int
            Number of Bogoliubov modes to include
        """
        if self.temperature > 0:
            # Solve Bogoliubov spectrum
            bog_solver = BogoliubovSolver1D(
                self.x, self.k, self.V, self.psi_c,
                m=self.m, hbar=self.hbar, g=self.g
            )

            # Compute thermal density
            self.n_thermal = bog_solver.thermal_density(
                self.temperature, n_modes=n_modes
            )
        else:
            self.n_thermal = np.zeros(self.N)

    def collision_rate(self):
        """
        Compute collision integral R_{12} (simplified model)

        This represents energy/particle exchange between condensate and thermal cloud

        Returns:
        --------
        R : array (complex)
            Collision rate
        """
        # Simplified collision term
        # Full theory requires detailed quasi-particle dynamics

        n_c = np.abs(self.psi_c)**2

        # Growth/damping rate
        # Positive: growth of condensate from thermal cloud
        # Negative: evaporation from condensate

        # Thermal equilibrium chemical potential
        mu_thermal = self.g * (n_c + 2 * self.n_thermal)
        mu_condensate = self.g * n_c

        # Simplified collision rate proportional to chemical potential difference
        R = self.gamma * (mu_thermal - mu_condensate) * self.psi_c

        return R

    def split_step_thermal(self, dt):
        """
        Time evolution step with thermal cloud coupling

        Modified split-step method including thermal effects

        Parameters:
        -----------
        dt : float
            Time step
        """
        # Effective potential including thermal cloud
        n_c = np.abs(self.psi_c)**2
        V_eff = self.V + self.g * (n_c + 2 * self.n_thermal)

        # Collision term
        R_12 = self.collision_rate()

        # Half step in position space with collision term
        self.psi_c *= np.exp(-1j * V_eff * dt / (2 * self.hbar))
        self.psi_c += -1j * R_12 * dt / self.hbar

        # Full step in momentum space
        psi_k = fft(self.psi_c)
        psi_k *= np.exp(-1j * self.T_k * dt / self.hbar)
        self.psi_c = ifft(psi_k)

        # Half step in position space
        n_c = np.abs(self.psi_c)**2
        V_eff = self.V + self.g * (n_c + 2 * self.n_thermal)
        self.psi_c *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        # Weak normalization (preserve particle number approximately)
        # In full ZNG theory, particle number can transfer between components
        current_norm = np.sqrt(simps(np.abs(self.psi_c)**2, self.x))
        if current_norm > 0.1:  # Avoid numerical issues
            self.psi_c *= np.sqrt(1.0 / current_norm)

    def evolve(self, t_total, dt, update_thermal_every=10):
        """
        Evolve the system in time

        Parameters:
        -----------
        t_total : float
            Total evolution time
        dt : float
            Time step
        update_thermal_every : int
            Update thermal cloud every N steps

        Returns:
        --------
        times : array
            Time points
        psi_history : array
            Condensate wave function history
        n_thermal_history : array
            Thermal cloud density history
        """
        n_steps = int(t_total / dt)
        times = np.linspace(0, t_total, n_steps + 1)

        psi_history = np.zeros((n_steps + 1, self.N), dtype=complex)
        n_thermal_history = np.zeros((n_steps + 1, self.N))

        psi_history[0] = self.psi_c.copy()
        n_thermal_history[0] = self.n_thermal.copy()

        for i in range(n_steps):
            # Update thermal cloud periodically
            if i % update_thermal_every == 0:
                self.update_thermal_cloud()

            # Time step
            self.split_step_thermal(dt)

            # Store
            psi_history[i + 1] = self.psi_c.copy()
            n_thermal_history[i + 1] = self.n_thermal.copy()

        return times, psi_history, n_thermal_history

    def find_thermal_equilibrium(self, max_iter=100, dt_imag=0.01, tol=1e-6):
        """
        Find thermal equilibrium state using imaginary time evolution

        Parameters:
        -----------
        max_iter : int
            Maximum iterations
        dt_imag : float
            Imaginary time step
        tol : float
            Convergence tolerance

        Returns:
        --------
        converged : bool
            Whether convergence was achieved
        """
        for iteration in range(max_iter):
            psi_old = self.psi_c.copy()

            # Update thermal cloud
            if iteration % 5 == 0:
                self.update_thermal_cloud()

            # Imaginary time step
            n_c = np.abs(self.psi_c)**2
            V_eff = self.V + self.g * (n_c + 2 * self.n_thermal)

            self.psi_c *= np.exp(-V_eff * dt_imag / (2 * self.hbar))

            psi_k = fft(self.psi_c)
            psi_k *= np.exp(-self.T_k * dt_imag / self.hbar)
            self.psi_c = ifft(psi_k)

            V_eff = self.V + self.g * (np.abs(self.psi_c)**2 + 2 * self.n_thermal)
            self.psi_c *= np.exp(-V_eff * dt_imag / (2 * self.hbar))

            self.normalize()

            # Check convergence
            diff = np.max(np.abs(self.psi_c - psi_old))
            if diff < tol:
                return True

        return False

    def compute_properties(self):
        """
        Compute physical properties

        Returns:
        --------
        properties : dict
            Dictionary of physical quantities
        """
        n_c = np.abs(self.psi_c)**2

        # Condensate number
        N_c = simps(n_c, self.x)

        # Thermal number
        N_th = simps(self.n_thermal, self.x)

        # Condensate fraction
        N_total = N_c + N_th
        f_c = N_c / N_total if N_total > 0 else 0

        # Energies
        psi_k = fft(self.psi_c)
        E_kin = simps(self.T_k * np.abs(psi_k)**2, self.k) / (2 * np.pi)
        E_pot = simps(self.V * n_c, self.x)
        E_int = 0.5 * self.g * simps((n_c + 2 * self.n_thermal) * n_c, self.x)
        E_total = E_kin + E_pot + E_int

        return {
            'N_condensate': N_c,
            'N_thermal': N_th,
            'N_total': N_total,
            'condensate_fraction': f_c,
            'E_kinetic': E_kin,
            'E_potential': E_pot,
            'E_interaction': E_int,
            'E_total': E_total,
            'temperature': self.temperature
        }

    def plot_state(self, title="Finite Temperature BEC", figsize=(14, 10)):
        """
        Plot condensate and thermal cloud

        Parameters:
        -----------
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        n_c = np.abs(self.psi_c)**2
        n_total = n_c + self.n_thermal

        # Density plot
        ax = axes[0]
        ax.plot(self.x, n_c, 'b-', linewidth=2, label='Condensate $n_c$')
        ax.plot(self.x, self.n_thermal, 'r--', linewidth=2, label='Thermal $\\tilde{n}$')
        ax.plot(self.x, n_total, 'k:', linewidth=2, label='Total')
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{title} (T = {self.temperature:.3f})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Condensate fraction
        ax = axes[1]
        f_c = n_c / n_total
        f_c[n_total < 1e-10] = 0  # Avoid division by zero
        ax.plot(self.x, f_c, 'g-', linewidth=2)
        ax.set_ylabel('Condensate Fraction', fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)

        # Wave function
        ax = axes[2]
        ax.plot(self.x, self.psi_c.real, 'b-', linewidth=2, label='Re($\\psi_c$)')
        ax.plot(self.x, self.psi_c.imag, 'r-', linewidth=2, label='Im($\\psi_c$)')
        ax.set_xlabel('Position x', fontsize=12)
        ax.set_ylabel('Condensate $\\psi_c$', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("Finite Temperature BEC - ZNG Theory")
    print("Import this module to use ZNGSolver1D and BogoliubovSolver1D")
