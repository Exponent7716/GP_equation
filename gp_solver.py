"""
Gross-Pitaevskii Equation Solver

This module implements a solver for the Gross-Pitaevskii (GP) equation,
which describes Bose-Einstein condensates (BEC).

The GP equation:
    iℏ ∂ψ/∂t = [-ℏ²/2m ∇² + V(r) + g|ψ|²]ψ

where:
    ψ = wave function
    ℏ = reduced Planck constant
    m = particle mass
    V(r) = external potential
    g = interaction strength parameter
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class GPSolver1D:
    """
    1D Gross-Pitaevskii Equation Solver

    Solves the time-dependent GP equation using split-step Fourier method
    """

    def __init__(self, x_min, x_max, N, m=1.0, hbar=1.0, g=1.0):
        """
        Initialize the GP solver

        Parameters:
        -----------
        x_min : float
            Minimum position
        x_max : float
            Maximum position
        N : int
            Number of grid points
        m : float
            Particle mass
        hbar : float
            Reduced Planck constant
        g : float
            Interaction strength parameter
        """
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.m = m
        self.hbar = hbar
        self.g = g

        # Spatial grid
        self.x = np.linspace(x_min, x_max, N)
        self.dx = (x_max - x_min) / (N - 1)

        # Momentum grid for Fourier space
        self.k = 2 * np.pi * fftfreq(N, self.dx)

        # Kinetic energy operator in momentum space
        self.T_k = (self.hbar**2 * self.k**2) / (2 * self.m)

        # External potential
        self.V = np.zeros(N)

        # Wave function
        self.psi = np.zeros(N, dtype=complex)

    def set_potential(self, V):
        """
        Set external potential V(x)

        Parameters:
        -----------
        V : array_like or callable
            External potential as array or function of x
        """
        if callable(V):
            self.V = V(self.x)
        else:
            self.V = np.array(V)

    def set_initial_state(self, psi0):
        """
        Set initial wave function

        Parameters:
        -----------
        psi0 : array_like or callable
            Initial wave function as array or function of x
        """
        if callable(psi0):
            self.psi = psi0(self.x)
        else:
            self.psi = np.array(psi0, dtype=complex)

        # Normalize
        self.normalize()

    def normalize(self):
        """Normalize the wave function"""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        if norm > 0:
            self.psi /= norm

    def split_step(self, dt):
        """
        Perform one time step using split-step Fourier method

        Parameters:
        -----------
        dt : float
            Time step
        """
        # Half step in position space (potential + nonlinear term)
        V_eff = self.V + self.g * np.abs(self.psi)**2
        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        # Full step in momentum space (kinetic energy)
        psi_k = fft(self.psi)
        psi_k *= np.exp(-1j * self.T_k * dt / self.hbar)
        self.psi = ifft(psi_k)

        # Half step in position space again
        V_eff = self.V + self.g * np.abs(self.psi)**2
        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

    def evolve(self, t_total, dt):
        """
        Evolve the wave function in time

        Parameters:
        -----------
        t_total : float
            Total evolution time
        dt : float
            Time step

        Returns:
        --------
        times : array
            Array of time points
        psi_history : array
            Wave function at each time point
        """
        n_steps = int(t_total / dt)
        times = np.linspace(0, t_total, n_steps + 1)
        psi_history = np.zeros((n_steps + 1, self.N), dtype=complex)
        psi_history[0] = self.psi.copy()

        for i in range(n_steps):
            self.split_step(dt)
            psi_history[i + 1] = self.psi.copy()

        return times, psi_history

    def find_ground_state(self, max_iter=1000, dt_imag=0.01, tol=1e-8):
        """
        Find ground state using imaginary time evolution

        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        dt_imag : float
            Imaginary time step
        tol : float
            Convergence tolerance

        Returns:
        --------
        energy : float
            Ground state energy
        converged : bool
            Whether the iteration converged
        """
        for iteration in range(max_iter):
            psi_old = self.psi.copy()

            # Imaginary time step (similar to split-step but with real exponentials)
            V_eff = self.V + self.g * np.abs(self.psi)**2
            self.psi *= np.exp(-V_eff * dt_imag / (2 * self.hbar))

            psi_k = fft(self.psi)
            psi_k *= np.exp(-self.T_k * dt_imag / self.hbar)
            self.psi = ifft(psi_k)

            V_eff = self.V + self.g * np.abs(self.psi)**2
            self.psi *= np.exp(-V_eff * dt_imag / (2 * self.hbar))

            # Normalize after each step
            self.normalize()

            # Check convergence
            diff = np.max(np.abs(self.psi - psi_old))
            if diff < tol:
                energy = self.compute_energy()
                return energy, True

        energy = self.compute_energy()
        return energy, False

    def compute_energy(self):
        """
        Compute total energy of current state

        Returns:
        --------
        energy : float
            Total energy (kinetic + potential + interaction)
        """
        # Kinetic energy
        psi_k = fft(self.psi)
        E_kin = np.sum(self.T_k * np.abs(psi_k)**2) * self.dx / (2 * np.pi)

        # Potential energy
        E_pot = np.sum(self.V * np.abs(self.psi)**2) * self.dx

        # Interaction energy
        E_int = 0.5 * self.g * np.sum(np.abs(self.psi)**4) * self.dx

        return E_kin + E_pot + E_int

    def compute_chemical_potential(self):
        """
        Compute chemical potential μ = dE/dN

        Returns:
        --------
        mu : float
            Chemical potential
        """
        # μ = ⟨H⟩ for the ground state
        psi_k = fft(self.psi)

        # Kinetic term
        mu = np.sum(self.T_k * np.abs(psi_k)**2) * self.dx / (2 * np.pi)

        # Potential term
        mu += np.sum(self.V * np.abs(self.psi)**2) * self.dx

        # Nonlinear interaction term
        mu += self.g * np.sum(np.abs(self.psi)**4) * self.dx

        return mu

    def plot_state(self, title="Wave Function", figsize=(10, 6)):
        """
        Plot the current wave function

        Parameters:
        -----------
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Probability density
        ax1.plot(self.x, np.abs(self.psi)**2, 'b-', linewidth=2, label='|ψ|²')
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Real and imaginary parts
        ax2.plot(self.x, self.psi.real, 'r-', linewidth=2, label='Re(ψ)')
        ax2.plot(self.x, self.psi.imag, 'b-', linewidth=2, label='Im(ψ)')
        ax2.set_xlabel('Position x', fontsize=12)
        ax2.set_ylabel('Wave Function', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        return fig


class GPSolver2D:
    """
    2D Gross-Pitaevskii Equation Solver

    Solves the time-dependent GP equation in 2D using split-step Fourier method
    """

    def __init__(self, x_min, x_max, y_min, y_max, Nx, Ny, m=1.0, hbar=1.0, g=1.0):
        """
        Initialize the 2D GP solver

        Parameters:
        -----------
        x_min, x_max : float
            x-axis boundaries
        y_min, y_max : float
            y-axis boundaries
        Nx, Ny : int
            Number of grid points in x and y directions
        m : float
            Particle mass
        hbar : float
            Reduced Planck constant
        g : float
            Interaction strength parameter
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.Nx = Nx
        self.Ny = Ny
        self.m = m
        self.hbar = hbar
        self.g = g

        # Spatial grids
        self.x = np.linspace(x_min, x_max, Nx)
        self.y = np.linspace(y_min, y_max, Ny)
        self.dx = (x_max - x_min) / (Nx - 1)
        self.dy = (y_max - y_min) / (Ny - 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Momentum grids
        self.kx = 2 * np.pi * fftfreq(Nx, self.dx)
        self.ky = 2 * np.pi * fftfreq(Ny, self.dy)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')

        # Kinetic energy operator in momentum space
        self.T_k = (self.hbar**2 * (self.KX**2 + self.KY**2)) / (2 * self.m)

        # External potential
        self.V = np.zeros((Nx, Ny))

        # Wave function
        self.psi = np.zeros((Nx, Ny), dtype=complex)

    def set_potential(self, V):
        """
        Set external potential V(x, y)

        Parameters:
        -----------
        V : array_like or callable
            External potential as 2D array or function of (X, Y)
        """
        if callable(V):
            self.V = V(self.X, self.Y)
        else:
            self.V = np.array(V)

    def set_initial_state(self, psi0):
        """
        Set initial wave function

        Parameters:
        -----------
        psi0 : array_like or callable
            Initial wave function as 2D array or function of (X, Y)
        """
        if callable(psi0):
            self.psi = psi0(self.X, self.Y)
        else:
            self.psi = np.array(psi0, dtype=complex)

        self.normalize()

    def normalize(self):
        """Normalize the wave function"""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx * self.dy)
        if norm > 0:
            self.psi /= norm

    def split_step(self, dt):
        """
        Perform one time step using split-step Fourier method

        Parameters:
        -----------
        dt : float
            Time step
        """
        # Half step in position space
        V_eff = self.V + self.g * np.abs(self.psi)**2
        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

        # Full step in momentum space
        psi_k = np.fft.fft2(self.psi)
        psi_k *= np.exp(-1j * self.T_k * dt / self.hbar)
        self.psi = np.fft.ifft2(psi_k)

        # Half step in position space
        V_eff = self.V + self.g * np.abs(self.psi)**2
        self.psi *= np.exp(-1j * V_eff * dt / (2 * self.hbar))

    def evolve(self, t_total, dt):
        """
        Evolve the wave function in time

        Parameters:
        -----------
        t_total : float
            Total evolution time
        dt : float
            Time step

        Returns:
        --------
        times : array
            Array of time points
        psi_history : array
            Wave function at each time point
        """
        n_steps = int(t_total / dt)
        times = np.linspace(0, t_total, n_steps + 1)
        psi_history = np.zeros((n_steps + 1, self.Nx, self.Ny), dtype=complex)
        psi_history[0] = self.psi.copy()

        for i in range(n_steps):
            self.split_step(dt)
            psi_history[i + 1] = self.psi.copy()

        return times, psi_history

    def find_ground_state(self, max_iter=1000, dt_imag=0.01, tol=1e-8):
        """
        Find ground state using imaginary time evolution

        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        dt_imag : float
            Imaginary time step
        tol : float
            Convergence tolerance

        Returns:
        --------
        energy : float
            Ground state energy
        converged : bool
            Whether the iteration converged
        """
        for iteration in range(max_iter):
            psi_old = self.psi.copy()

            # Imaginary time step
            V_eff = self.V + self.g * np.abs(self.psi)**2
            self.psi *= np.exp(-V_eff * dt_imag / (2 * self.hbar))

            psi_k = np.fft.fft2(self.psi)
            psi_k *= np.exp(-self.T_k * dt_imag / self.hbar)
            self.psi = np.fft.ifft2(psi_k)

            V_eff = self.V + self.g * np.abs(self.psi)**2
            self.psi *= np.exp(-V_eff * dt_imag / (2 * self.hbar))

            self.normalize()

            # Check convergence
            diff = np.max(np.abs(self.psi - psi_old))
            if diff < tol:
                energy = self.compute_energy()
                return energy, True

        energy = self.compute_energy()
        return energy, False

    def compute_energy(self):
        """
        Compute total energy of current state

        Returns:
        --------
        energy : float
            Total energy
        """
        # Kinetic energy
        psi_k = np.fft.fft2(self.psi)
        E_kin = np.sum(self.T_k * np.abs(psi_k)**2) * self.dx * self.dy / (4 * np.pi**2)

        # Potential energy
        E_pot = np.sum(self.V * np.abs(self.psi)**2) * self.dx * self.dy

        # Interaction energy
        E_int = 0.5 * self.g * np.sum(np.abs(self.psi)**4) * self.dx * self.dy

        return E_kin + E_pot + E_int

    def plot_state(self, title="Wave Function (2D)", figsize=(12, 5)):
        """
        Plot the current wave function

        Parameters:
        -----------
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Probability density
        density = np.abs(self.psi)**2
        im1 = ax1.contourf(self.X, self.Y, density, levels=50, cmap='viridis')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title(f'{title} - |ψ|²', fontsize=14)
        plt.colorbar(im1, ax=ax1)

        # Phase
        phase = np.angle(self.psi)
        im2 = ax2.contourf(self.X, self.Y, phase, levels=50, cmap='twilight')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title(f'{title} - Phase', fontsize=14)
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("Gross-Pitaevskii Equation Solver")
    print("Import this module to use GPSolver1D or GPSolver2D classes")
