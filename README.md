# Gross-Pitaevskii Equation Solver

Gross-Pitaevskii（GP）方程式の数値ソルバー実装

A numerical solver for the Gross-Pitaevskii equation, which describes Bose-Einstein Condensates (BEC).

## 概要 / Overview

このプロジェクトは、ボース・アインシュタイン凝縮体（BEC）を記述するGross-Pitaevskii方程式の数値解法を提供します。

This project provides numerical solutions for the Gross-Pitaevskii equation that describes Bose-Einstein Condensates.

### GP方程式 / The GP Equation

```
iℏ ∂ψ/∂t = [-ℏ²/2m ∇² + V(r) + g|ψ|²]ψ
```

where:
- ψ = wave function (波動関数)
- ℏ = reduced Planck constant (換算プランク定数)
- m = particle mass (粒子質量)
- V(r) = external potential (外部ポテンシャル)
- g = interaction strength (相互作用強度)

## 特徴 / Features

- **1D Solver**: 1次元GP方程式ソルバー / One-dimensional GP equation solver
- **2D Solver**: 2次元GP方程式ソルバー / Two-dimensional GP equation solver
- **Split-Step Fourier Method**: 高速かつ正確な時間発展 / Fast and accurate time evolution
- **Imaginary Time Evolution**: 基底状態探索 / Ground state finding
- **Examples**: 調和ポテンシャル、ソリトン、渦など / Harmonic trap, solitons, vortices, etc.

## インストール / Installation

```bash
# リポジトリのクローン / Clone the repository
git clone https://github.com/yourusername/GP_equation.git
cd GP_equation

# 依存パッケージのインストール / Install dependencies
pip install -r requirements.txt
```

## 依存関係 / Requirements

- Python 3.7+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0

## 使い方 / Usage

### 基本的な使い方 / Basic Usage

```python
from gp_solver import GPSolver1D
import numpy as np

# ソルバーの初期化 / Initialize solver
solver = GPSolver1D(x_min=-10, x_max=10, N=512, m=1.0, hbar=1.0, g=1.0)

# ポテンシャルの設定 / Set potential
def harmonic_potential(x):
    return 0.5 * x**2

solver.set_potential(harmonic_potential)

# 初期状態の設定 / Set initial state
def initial_state(x):
    return np.exp(-x**2 / 2)

solver.set_initial_state(initial_state)

# 基底状態の探索 / Find ground state
energy, converged = solver.find_ground_state(max_iter=1000, dt_imag=0.01)

# 時間発展 / Time evolution
times, psi_history = solver.evolve(t_total=10.0, dt=0.01)
```

### サンプルの実行 / Running Examples

#### 1D 調和ポテンシャル / 1D Harmonic Trap

```bash
python example_1d_harmonic.py
```

このサンプルは以下を実行します / This example demonstrates:
- 調和ポテンシャル中のBECの基底状態を求める / Finding ground state in harmonic trap
- 時間発展とエネルギー保存 / Time evolution and energy conservation

生成されるファイル / Generated files:
- `ground_state_1d.png` - 基底状態の波動関数
- `time_evolution_1d.png` - 時間発展のスナップショット
- `energy_conservation_1d.png` - エネルギー保存則の確認

#### 1D ソリトン / 1D Bright Soliton

```bash
python example_1d_soliton.py
```

このサンプルは以下を実行します / This example demonstrates:
- 引力相互作用によるブライトソリトン / Bright soliton with attractive interaction
- ソリトンの伝播 / Soliton propagation

生成されるファイル / Generated files:
- `soliton_initial.png` - 初期状態
- `soliton_spacetime.png` - 時空間プロット
- `soliton_snapshots.png` - 時間スナップショット
- `soliton_conservation.png` - 保存則の確認

#### 2D 渦 / 2D Vortex

```bash
python example_2d_vortex.py
```

このサンプルは以下を実行します / This example demonstrates:
- 2次元BEC中の量子渦 / Quantum vortex in 2D BEC
- 渦の緩和と時間発展 / Vortex relaxation and evolution

生成されるファイル / Generated files:
- `vortex_initial_2d.png` - 初期渦状態
- `vortex_relaxed_2d.png` - 緩和後の渦
- `vortex_evolution_2d.png` - 時間発展
- `vortex_phase_2d.png` - 位相分布
- `vortex_crosssection_2d.png` - 断面図

## クラスリファレンス / Class Reference

### GPSolver1D

1次元GP方程式ソルバー / One-dimensional GP equation solver

**主要メソッド / Main Methods:**

- `set_potential(V)` - 外部ポテンシャルの設定 / Set external potential
- `set_initial_state(psi0)` - 初期波動関数の設定 / Set initial wave function
- `split_step(dt)` - 時間発展の1ステップ / One time evolution step
- `evolve(t_total, dt)` - 時間発展 / Time evolution
- `find_ground_state(max_iter, dt_imag, tol)` - 基底状態探索 / Find ground state
- `compute_energy()` - エネルギー計算 / Compute energy
- `compute_chemical_potential()` - 化学ポテンシャル計算 / Compute chemical potential
- `plot_state(title)` - 波動関数のプロット / Plot wave function

### GPSolver2D

2次元GP方程式ソルバー / Two-dimensional GP equation solver

**主要メソッド / Main Methods:**

同様のインターフェース（2D版） / Similar interface (2D version)

## アルゴリズム / Algorithm

### Split-Step Fourier Method

GP方程式は以下のように分解されます / The GP equation is split as:

```
∂ψ/∂t = (T̂ + V̂)ψ
```

where:
- T̂ = 運動エネルギー演算子（運動量空間で対角） / Kinetic energy operator (diagonal in momentum space)
- V̂ = ポテンシャル＋非線形項（位置空間で対角） / Potential + nonlinear term (diagonal in position space)

時間発展は以下の手順で行います / Time evolution follows:

1. 位置空間で半ステップ: ψ → exp(-iV̂dt/2)ψ
2. フーリエ変換で運動量空間へ
3. 運動量空間で完全ステップ: ψ̃ → exp(-iT̂dt)ψ̃
4. 逆フーリエ変換で位置空間へ
5. 位置空間で半ステップ: ψ → exp(-iV̂dt/2)ψ

### Imaginary Time Evolution

基底状態は虚時間発展により求めます / Ground state is found by imaginary time evolution:

```
∂ψ/∂τ = -(Ĥ - μ)ψ
```

各ステップ後に規格化することで、最低エネルギー状態に収束します。
By normalizing after each step, the wave function converges to the lowest energy state.

## 物理的応用 / Physical Applications

このソルバーは以下の物理現象のシミュレーションに使用できます:

This solver can be used to simulate:

- ボース・アインシュタイン凝縮体の基底状態 / Ground states of Bose-Einstein condensates
- ソリトンの形成と伝播 / Soliton formation and propagation
- 量子渦の動力学 / Quantum vortex dynamics
- BECの集団振動 / Collective oscillations of BECs
- 双極子BEC / Dipolar BECs
- スピノールBEC / Spinor BECs (拡張可能 / with extensions)

## 参考文献 / References

1. C.J. Pethick and H. Smith, "Bose-Einstein Condensation in Dilute Gases", Cambridge University Press (2008)
2. L.P. Pitaevskii and S. Stringari, "Bose-Einstein Condensation and Superfluidity", Oxford University Press (2016)
3. W. Bao and Y. Cai, "Mathematical theory and numerical methods for Bose-Einstein condensation", Kinetic and Related Models (2013)

## ライセンス / License

MIT License

## 貢献 / Contributing

プルリクエストを歓迎します / Pull requests are welcome!

## 作者 / Author

Created with Claude AI

## TODO / 今後の開発予定

- [ ] 3D solver implementation / 3次元ソルバーの実装
- [ ] Rotating BEC / 回転BECのサポート
- [ ] Multi-component BEC / 多成分BEC
- [ ] GPU acceleration / GPU加速
- [ ] Animation export / アニメーション出力機能
- [ ] Jupyter notebook examples / Jupyterノートブック例
