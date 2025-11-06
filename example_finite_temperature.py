"""
Example: Finite Temperature BEC using ZNG Theory

This script demonstrates:
1. Thermal equilibrium at different temperatures
2. Temperature dependence of condensate fraction
3. Condensate-thermal cloud dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from finite_temperature import ZNGSolver1D, BogoliubovSolver1D


def example_thermal_equilibrium():
    """
    Example 1: Thermal equilibrium at different temperatures
    """
    print("=" * 70)
    print("Example 1: Thermal Equilibrium at Different Temperatures")
    print("=" * 70)

    # Parameters
    x_min, x_max = -15, 15
    N = 256
    m = 1.0
    hbar = 1.0
    omega = 1.0
    g = 0.5

    # Temperatures to test
    temperatures = [0.0, 0.1, 0.3, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    condensate_fractions = []

    for idx, T in enumerate(temperatures):
        print(f"\nTemperature T = {T:.2f}")

        # Create solver
        solver = ZNGSolver1D(x_min, x_max, N, m=m, hbar=hbar, g=g, temperature=T)

        # Harmonic potential
        def harmonic_potential(x):
            return 0.5 * m * omega**2 * x**2

        solver.set_potential(harmonic_potential)

        # Initial state: Gaussian
        def initial_state(x):
            sigma = 2.0
            return np.exp(-x**2 / (2 * sigma**2))

        solver.set_initial_state(initial_state)

        # Find thermal equilibrium
        print("Finding thermal equilibrium...")
        converged = solver.find_thermal_equilibrium(max_iter=200, dt_imag=0.01, tol=1e-7)

        if converged:
            print("✓ Converged to thermal equilibrium")
        else:
            print("⚠ Reached max iterations")

        # Compute properties
        props = solver.compute_properties()
        condensate_fractions.append(props['condensate_fraction'])

        print(f"  Condensate number: {props['N_condensate']:.4f}")
        print(f"  Thermal number: {props['N_thermal']:.4f}")
        print(f"  Condensate fraction: {props['condensate_fraction']:.4f}")
        print(f"  Total energy: {props['E_total']:.6f}")

        # Plot
        ax = axes.flat[idx]
        n_c = np.abs(solver.psi_c)**2
        n_total = n_c + solver.n_thermal

        ax.plot(solver.x, n_c, 'b-', linewidth=2.5, label='Condensate')
        ax.plot(solver.x, solver.n_thermal, 'r--', linewidth=2.5, label='Thermal')
        ax.fill_between(solver.x, 0, n_c, alpha=0.3, color='blue')
        ax.fill_between(solver.x, n_c, n_total, alpha=0.3, color='red')

        ax.set_xlabel('Position x', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'T = {T:.2f}, $f_c$ = {props["condensate_fraction"]:.3f}',
                     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('thermal_equilibrium_temperatures.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved: thermal_equilibrium_temperatures.png")

    # Plot condensate fraction vs temperature
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temperatures, condensate_fractions, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('Temperature', fontsize=14)
    ax.set_ylabel('Condensate Fraction $f_c$', fontsize=14)
    ax.set_title('Condensate Fraction vs Temperature', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('condensate_fraction_vs_temperature.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: condensate_fraction_vs_temperature.png")


def example_thermal_dynamics():
    """
    Example 2: Time dynamics with thermal cloud
    """
    print("\n" + "=" * 70)
    print("Example 2: Dynamics with Thermal Cloud")
    print("=" * 70)

    # Parameters
    x_min, x_max = -15, 15
    N = 256
    m = 1.0
    hbar = 1.0
    omega = 1.0
    g = 0.5
    T = 0.2

    print(f"\nTemperature T = {T:.2f}")

    # Create solver
    solver = ZNGSolver1D(x_min, x_max, N, m=m, hbar=hbar, g=g, temperature=T)

    # Harmonic potential
    def harmonic_potential(x):
        return 0.5 * m * omega**2 * x**2

    solver.set_potential(harmonic_potential)

    # Initial state: Displaced Gaussian (to create oscillations)
    x0 = 3.0  # Displacement
    def initial_state(x):
        sigma = 2.0
        return np.exp(-(x - x0)**2 / (2 * sigma**2))

    solver.set_initial_state(initial_state)

    # First find thermal equilibrium shape
    print("Finding initial thermal equilibrium...")
    solver.find_thermal_equilibrium(max_iter=100, dt_imag=0.01)

    # Then displace it
    x0 = 2.0
    def displaced_state(x):
        sigma = 2.0
        return np.exp(-(x - x0)**2 / (2 * sigma**2))
    solver.set_initial_state(displaced_state)
    solver.update_thermal_cloud()

    print("Performing time evolution...")
    t_total = 15.0
    dt = 0.02
    times, psi_history, n_thermal_history = solver.evolve(
        t_total, dt, update_thermal_every=5
    )

    # Plot evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_indices = [0, len(times)//3, 2*len(times)//3, -1]

    for idx, ax in enumerate(axes.flat):
        t_idx = time_indices[idx]
        t = times[t_idx]

        n_c = np.abs(psi_history[t_idx])**2
        n_th = n_thermal_history[t_idx]
        n_total = n_c + n_th

        ax.plot(solver.x, n_c, 'b-', linewidth=2.5, label='Condensate')
        ax.plot(solver.x, n_th, 'r--', linewidth=2.5, label='Thermal')
        ax.fill_between(solver.x, 0, n_c, alpha=0.3, color='blue')
        ax.fill_between(solver.x, n_c, n_total, alpha=0.3, color='red')

        # Condensate fraction
        N_c = np.trapz(n_c, solver.x)
        N_th = np.trapz(n_th, solver.x)
        f_c = N_c / (N_c + N_th) if (N_c + N_th) > 0 else 0

        ax.set_xlabel('Position x', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f't = {t:.2f}, $f_c$ = {f_c:.3f}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('thermal_dynamics_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: thermal_dynamics_evolution.png")

    # Plot center of mass motion
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Compute center of mass for condensate and thermal cloud
    x_cm_condensate = []
    x_cm_thermal = []
    N_condensate = []
    N_thermal_arr = []

    for i in range(len(times)):
        n_c = np.abs(psi_history[i])**2
        n_th = n_thermal_history[i]

        N_c = np.trapz(n_c, solver.x)
        N_th = np.trapz(n_th, solver.x)

        if N_c > 1e-10:
            x_cm_c = np.trapz(solver.x * n_c, solver.x) / N_c
        else:
            x_cm_c = 0

        if N_th > 1e-10:
            x_cm_th = np.trapz(solver.x * n_th, solver.x) / N_th
        else:
            x_cm_th = 0

        x_cm_condensate.append(x_cm_c)
        x_cm_thermal.append(x_cm_th)
        N_condensate.append(N_c)
        N_thermal_arr.append(N_th)

    # Center of mass oscillations
    ax1.plot(times, x_cm_condensate, 'b-', linewidth=2, label='Condensate')
    ax1.plot(times, x_cm_thermal, 'r--', linewidth=2, label='Thermal')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Center of Mass', fontsize=12)
    ax1.set_title('Center of Mass Oscillations', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Particle number evolution
    ax2.plot(times, N_condensate, 'b-', linewidth=2, label='Condensate')
    ax2.plot(times, N_thermal_arr, 'r--', linewidth=2, label='Thermal')
    total = np.array(N_condensate) + np.array(N_thermal_arr)
    ax2.plot(times, total, 'k:', linewidth=2, label='Total')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Particle Number', fontsize=12)
    ax2.set_title('Particle Number (Condensate ↔ Thermal Exchange)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('thermal_dynamics_properties.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: thermal_dynamics_properties.png")


def example_bogoliubov_spectrum():
    """
    Example 3: Bogoliubov excitation spectrum
    """
    print("\n" + "=" * 70)
    print("Example 3: Bogoliubov Excitation Spectrum")
    print("=" * 70)

    # Parameters
    x_min, x_max = -20, 20
    N = 512
    m = 1.0
    hbar = 1.0
    omega = 1.0
    g = 1.0

    # Create ground state
    from gp_solver import GPSolver1D

    solver = GPSolver1D(x_min, x_max, N, m=m, hbar=hbar, g=g)

    def harmonic_potential(x):
        return 0.5 * m * omega**2 * x**2

    solver.set_potential(harmonic_potential)

    def initial_state(x):
        return np.exp(-x**2 / 4)

    solver.set_initial_state(initial_state)

    print("\nFinding ground state...")
    energy, converged = solver.find_ground_state(max_iter=1000)
    print(f"✓ Ground state energy: {energy:.6f}")

    # Compute Bogoliubov spectrum
    print("Computing Bogoliubov spectrum...")
    bog_solver = BogoliubovSolver1D(
        solver.x, solver.k, solver.V, solver.psi,
        m=m, hbar=hbar, g=g
    )

    n_modes = 100
    energies, u_modes, v_modes = bog_solver.solve_spectrum(n_modes=n_modes)

    print(f"✓ Computed {n_modes} Bogoliubov modes")
    print(f"  Lowest excitation energy: {energies[0]:.6f}")
    print(f"  Sound velocity c ≈ {energies[10] / abs(solver.k[10]):.4f}")

    # Plot spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Excitation energies vs mode number
    ax1.plot(range(n_modes), energies, 'bo-', markersize=4)
    ax1.set_xlabel('Mode Number', fontsize=12)
    ax1.set_ylabel('Excitation Energy $E_n$', fontsize=12)
    ax1.set_title('Bogoliubov Excitation Spectrum', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Dispersion relation
    # Sort by momentum
    k_vals = np.abs(solver.k)
    idx_sort = np.argsort(k_vals)
    k_sorted = k_vals[idx_sort]
    E_sorted = energies[idx_sort]

    # Plot only positive k
    mask = k_sorted > 0
    ax2.plot(k_sorted[mask][:50], E_sorted[mask][:50], 'ro', markersize=6, label='Bogoliubov')

    # Compare with phonon (low k) and free particle (high k) limits
    k_plot = k_sorted[mask][:50]
    n0 = np.max(np.abs(solver.psi)**2)
    c_sound = np.sqrt(g * n0 / m)
    E_phonon = hbar * c_sound * k_plot
    E_free = (hbar * k_plot)**2 / (2 * m)

    ax2.plot(k_plot, E_phonon, 'b--', linewidth=2, label=f'Phonon (c={c_sound:.2f})')
    ax2.plot(k_plot, E_free, 'g--', linewidth=2, label='Free particle')

    ax2.set_xlabel('Momentum $|k|$', fontsize=12)
    ax2.set_ylabel('Energy $E(k)$', fontsize=12)
    ax2.set_title('Dispersion Relation', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bogoliubov_spectrum.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: bogoliubov_spectrum.png")

    # Plot some mode shapes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    mode_indices = [0, 5, 10, 20]

    for idx, mode_idx in enumerate(mode_indices):
        ax = axes.flat[idx]

        # Plot u and v amplitudes
        ax.plot(solver.x, np.abs(u_modes[mode_idx])**2, 'b-',
                linewidth=2, label='$|u|^2$')
        ax.plot(solver.x, np.abs(v_modes[mode_idx])**2, 'r--',
                linewidth=2, label='$|v|^2$')

        # Also show condensate density
        n_c = np.abs(solver.psi)**2
        ax.plot(solver.x, n_c / np.max(n_c) * np.max(np.abs(u_modes[mode_idx])**2),
                'k:', linewidth=1, alpha=0.5, label='$n_c$ (scaled)')

        ax.set_xlabel('Position x', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'Mode {mode_idx}: $E$ = {energies[mode_idx]:.4f}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bogoliubov_modes.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: bogoliubov_modes.png")


def main():
    """Run all examples"""
    example_thermal_equilibrium()
    example_thermal_dynamics()
    example_bogoliubov_spectrum()

    print("\n" + "=" * 70)
    print("All finite temperature examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
