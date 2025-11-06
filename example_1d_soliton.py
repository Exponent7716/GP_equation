"""
Example: 1D Soliton in Gross-Pitaevskii Equation

This script demonstrates bright soliton solutions with attractive interactions
"""

import numpy as np
import matplotlib.pyplot as plt
from gp_solver import GPSolver1D


def main():
    print("=" * 60)
    print("1D GP Equation: Bright Soliton Example")
    print("=" * 60)

    # Parameters
    x_min, x_max = -20, 20
    N = 1024
    m = 1.0
    hbar = 1.0
    g = -1.0  # Negative (attractive) interaction for bright soliton

    # Create solver
    solver = GPSolver1D(x_min, x_max, N, m=m, hbar=hbar, g=g)

    # No external potential (V = 0)
    solver.set_potential(lambda x: 0.0 * x)

    # Soliton initial condition
    # Analytical soliton solution: ψ(x,t=0) = A * sech(x/ξ)
    A = 1.0
    xi = 2.0  # Soliton width
    v = 0.5   # Soliton velocity

    def soliton_initial(x):
        return A / np.cosh(x / xi) * np.exp(1j * m * v * x / hbar)

    solver.set_initial_state(soliton_initial)

    print("\nInitial state set: Bright soliton")
    print(f"Amplitude: {A}")
    print(f"Width: {xi}")
    print(f"Velocity: {v}")
    print(f"Interaction strength: {g}")

    # Plot initial state
    fig = solver.plot_state(title="Initial Soliton State")
    plt.savefig("soliton_initial.png", dpi=150, bbox_inches='tight')
    print("\n✓ Initial soliton plot saved: soliton_initial.png")

    # Time evolution
    print("\nPerforming time evolution...")
    t_total = 20.0
    dt = 0.01
    times, psi_history = solver.evolve(t_total, dt)

    # Create space-time plot
    fig, ax = plt.subplots(figsize=(12, 8))
    density = np.abs(psi_history)**2

    # Downsample for plotting
    plot_every = 10
    im = ax.contourf(solver.x, times[::plot_every], density[::plot_every],
                     levels=50, cmap='hot')
    ax.set_xlabel('Position x', fontsize=14)
    ax.set_ylabel('Time t', fontsize=14)
    ax.set_title('Soliton Propagation (Space-Time Plot)', fontsize=16)
    plt.colorbar(im, ax=ax, label='|ψ|²')
    plt.tight_layout()
    plt.savefig("soliton_spacetime.png", dpi=150, bbox_inches='tight')
    print("✓ Space-time plot saved: soliton_spacetime.png")

    # Snapshots at different times
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_indices = [0, len(times)//3, 2*len(times)//3, -1]

    for idx, ax in enumerate(axes.flat):
        t_idx = time_indices[idx]
        t = times[t_idx]
        density = np.abs(psi_history[t_idx])**2
        real_part = psi_history[t_idx].real
        imag_part = psi_history[t_idx].imag

        ax.plot(solver.x, density, 'r-', linewidth=2, label='|ψ|²')
        ax.plot(solver.x, real_part, 'b--', linewidth=1.5, alpha=0.7, label='Re(ψ)')
        ax.plot(solver.x, imag_part, 'g--', linewidth=1.5, alpha=0.7, label='Im(ψ)')
        ax.set_xlabel('Position x', fontsize=11)
        ax.set_ylabel('Wave Function', fontsize=11)
        ax.set_title(f'Time t = {t:.2f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("soliton_snapshots.png", dpi=150, bbox_inches='tight')
    print("✓ Soliton snapshots saved: soliton_snapshots.png")

    # Energy and norm conservation
    energies = []
    norms = []
    for psi in psi_history:
        solver.psi = psi
        energies.append(solver.compute_energy())
        norms.append(np.sum(np.abs(psi)**2) * solver.dx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(times, energies, 'b-', linewidth=2)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title('Energy Conservation', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, norms, 'r-', linewidth=2)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Norm', fontsize=12)
    ax2.set_title('Norm Conservation', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("soliton_conservation.png", dpi=150, bbox_inches='tight')
    print("✓ Conservation laws plot saved: soliton_conservation.png")

    print("\n" + "=" * 60)
    print("Soliton simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
