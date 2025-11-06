"""
Example: 1D Gross-Pitaevskii Equation with Harmonic Trap

This script demonstrates:
1. Finding the ground state of a BEC in a harmonic trap
2. Time evolution of the wave function
"""

import numpy as np
import matplotlib.pyplot as plt
from gp_solver import GPSolver1D


def main():
    print("=" * 60)
    print("1D GP Equation: Harmonic Trap Example")
    print("=" * 60)

    # Parameters
    x_min, x_max = -10, 10
    N = 512
    m = 1.0
    hbar = 1.0
    omega = 1.0  # Harmonic oscillator frequency
    g = 1.0  # Interaction strength

    # Create solver
    solver = GPSolver1D(x_min, x_max, N, m=m, hbar=hbar, g=g)

    # Harmonic potential: V(x) = 0.5 * m * omega^2 * x^2
    def harmonic_potential(x):
        return 0.5 * m * omega**2 * x**2

    solver.set_potential(harmonic_potential)

    # Initial guess: Gaussian wave packet
    def initial_wavefunction(x):
        sigma = 1.0
        return np.exp(-x**2 / (2 * sigma**2)) / (np.pi * sigma**2)**0.25

    solver.set_initial_state(initial_wavefunction)

    print("\nFinding ground state using imaginary time evolution...")
    energy, converged = solver.find_ground_state(max_iter=2000, dt_imag=0.01, tol=1e-10)

    if converged:
        print(f"✓ Converged!")
    else:
        print("⚠ Did not fully converge (may need more iterations)")

    print(f"Ground state energy: {energy:.6f}")
    print(f"Chemical potential: {solver.compute_chemical_potential():.6f}")

    # Plot ground state
    fig = solver.plot_state(title="Ground State in Harmonic Trap")
    plt.savefig("ground_state_1d.png", dpi=150, bbox_inches='tight')
    print("\n✓ Ground state plot saved: ground_state_1d.png")

    # Time evolution
    print("\nPerforming time evolution...")
    t_total = 10.0
    dt = 0.01
    times, psi_history = solver.evolve(t_total, dt)

    # Create animation-like plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    time_indices = [0, len(times)//3, 2*len(times)//3, -1]

    for idx, ax in enumerate(axes.flat):
        t_idx = time_indices[idx]
        t = times[t_idx]
        density = np.abs(psi_history[t_idx])**2

        ax.plot(solver.x, density, 'b-', linewidth=2)
        ax.set_xlabel('Position x', fontsize=11)
        ax.set_ylabel('|ψ|²', fontsize=11)
        ax.set_title(f'Time t = {t:.2f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, np.max(np.abs(psi_history)**2) * 1.1])

    plt.tight_layout()
    plt.savefig("time_evolution_1d.png", dpi=150, bbox_inches='tight')
    print("✓ Time evolution plot saved: time_evolution_1d.png")

    # Energy vs time
    energies = []
    for psi in psi_history:
        solver.psi = psi
        energies.append(solver.compute_energy())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, energies, 'b-', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Conservation', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("energy_conservation_1d.png", dpi=150, bbox_inches='tight')
    print("✓ Energy conservation plot saved: energy_conservation_1d.png")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
