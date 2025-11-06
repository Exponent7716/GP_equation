"""
Example: 2D Gross-Pitaevskii Equation with Vortex

This script demonstrates vortex solutions in 2D BEC
"""

import numpy as np
import matplotlib.pyplot as plt
from gp_solver import GPSolver2D


def main():
    print("=" * 60)
    print("2D GP Equation: Vortex Example")
    print("=" * 60)

    # Parameters
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    Nx, Ny = 128, 128
    m = 1.0
    hbar = 1.0
    omega = 1.0
    g = 1.0  # Repulsive interaction

    # Create solver
    solver = GPSolver2D(x_min, x_max, y_min, y_max, Nx, Ny, m=m, hbar=hbar, g=g)

    # 2D Harmonic trap: V(x,y) = 0.5 * m * omega^2 * (x^2 + y^2)
    def harmonic_trap_2d(X, Y):
        return 0.5 * m * omega**2 * (X**2 + Y**2)

    solver.set_potential(harmonic_trap_2d)

    # Initial state: Vortex with winding number n=1
    def vortex_initial(X, Y):
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        winding_number = 1

        # Gaussian envelope with phase winding
        sigma = 2.0
        amplitude = r * np.exp(-r**2 / (2 * sigma**2))
        phase = winding_number * theta

        return amplitude * np.exp(1j * phase)

    solver.set_initial_state(vortex_initial)

    print("\nInitial state set: Vortex with winding number 1")
    print(f"Grid size: {Nx} × {Ny}")
    print(f"Interaction strength g = {g}")

    # Plot initial state
    fig = solver.plot_state(title="Initial Vortex State")
    plt.savefig("vortex_initial_2d.png", dpi=150, bbox_inches='tight')
    print("\n✓ Initial vortex plot saved: vortex_initial_2d.png")

    # Find ground state (relaxed vortex)
    print("\nRelaxing vortex state using imaginary time evolution...")
    energy, converged = solver.find_ground_state(max_iter=1000, dt_imag=0.01, tol=1e-8)

    if converged:
        print(f"✓ Converged!")
    else:
        print("⚠ Did not fully converge")

    print(f"Energy: {energy:.6f}")

    # Plot relaxed vortex
    fig = solver.plot_state(title="Relaxed Vortex State")
    plt.savefig("vortex_relaxed_2d.png", dpi=150, bbox_inches='tight')
    print("✓ Relaxed vortex plot saved: vortex_relaxed_2d.png")

    # Time evolution
    print("\nPerforming time evolution...")
    t_total = 5.0
    dt = 0.01
    times, psi_history = solver.evolve(t_total, dt)

    # Create snapshots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    time_indices = [0, len(times)//3, 2*len(times)//3, -1]

    for idx, ax in enumerate(axes.flat):
        t_idx = time_indices[idx]
        t = times[t_idx]
        density = np.abs(psi_history[t_idx])**2

        im = ax.contourf(solver.X, solver.Y, density, levels=50, cmap='viridis')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'|ψ|² at t = {t:.2f}', fontsize=12)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("vortex_evolution_2d.png", dpi=150, bbox_inches='tight')
    print("✓ Vortex evolution plot saved: vortex_evolution_2d.png")

    # Phase plot showing vortex core
    fig, ax = plt.subplots(figsize=(10, 9))
    phase = np.angle(solver.psi)
    density = np.abs(solver.psi)**2

    im = ax.contourf(solver.X, solver.Y, phase, levels=50, cmap='twilight')
    # Add density contours to show vortex core
    ax.contour(solver.X, solver.Y, density, levels=10, colors='white',
               alpha=0.4, linewidths=1)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('Phase Distribution (Vortex Core)', fontsize=16)
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Phase (radians)', fontsize=12)
    plt.tight_layout()
    plt.savefig("vortex_phase_2d.png", dpi=150, bbox_inches='tight')
    print("✓ Vortex phase plot saved: vortex_phase_2d.png")

    # Cross-section through vortex core
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Density cross-section
    mid_idx = Nx // 2
    density_final = np.abs(psi_history[-1])**2
    ax1.plot(solver.x, density_final[mid_idx, :], 'b-', linewidth=2, label='y = 0')
    ax1.plot(solver.x, density_final[:, mid_idx], 'r--', linewidth=2, label='x = 0')
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('|ψ|²', fontsize=12)
    ax1.set_title('Density Cross-Section Through Vortex Core', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Phase cross-section
    phase_final = np.angle(psi_history[-1])
    ax2.plot(solver.x, phase_final[mid_idx, :], 'b-', linewidth=2, label='y = 0')
    ax2.plot(solver.x, phase_final[:, mid_idx], 'r--', linewidth=2, label='x = 0')
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Phase (radians)', fontsize=12)
    ax2.set_title('Phase Cross-Section', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("vortex_crosssection_2d.png", dpi=150, bbox_inches='tight')
    print("✓ Vortex cross-section plot saved: vortex_crosssection_2d.png")

    print("\n" + "=" * 60)
    print("2D Vortex simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
