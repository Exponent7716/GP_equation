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
- **Finite Temperature**: ZNG理論による有限温度BEC / Finite temperature BEC with ZNG theory
- **Bogoliubov-de Gennes**: 準粒子励起スペクトル / Quasi-particle excitation spectrum
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

#### 有限温度BEC / Finite Temperature BEC

```bash
python example_finite_temperature.py
```

このサンプルは以下を実行します / This example demonstrates:
- ZNG理論による有限温度効果 / Finite temperature effects with ZNG theory
- 凝縮成分と熱成分の結合ダイナミクス / Coupled dynamics of condensate and thermal cloud
- Bogoliubov励起スペクトル / Bogoliubov excitation spectrum
- 温度依存の凝縮率 / Temperature-dependent condensate fraction

生成されるファイル / Generated files:
- `thermal_equilibrium_temperatures.png` - 異なる温度での熱平衡
- `condensate_fraction_vs_temperature.png` - 凝縮率の温度依存性
- `thermal_dynamics_evolution.png` - 熱雲を含む時間発展
- `thermal_dynamics_properties.png` - 粒子数の交換
- `bogoliubov_spectrum.png` - 励起スペクトル
- `bogoliubov_modes.png` - 準粒子モード

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

### ZNGSolver1D

有限温度BECソルバー（Zaremba-Nikuni-Griffin理論）/ Finite temperature BEC solver (ZNG theory)

**主要メソッド / Main Methods:**

- `set_potential(V)` - 外部ポテンシャルの設定 / Set external potential
- `set_initial_state(psi0)` - 初期凝縮波動関数の設定 / Set initial condensate wave function
- `update_thermal_cloud(n_modes)` - 熱雲密度の更新 / Update thermal cloud density
- `evolve(t_total, dt)` - 時間発展（凝縮+熱雲） / Time evolution (condensate + thermal)
- `find_thermal_equilibrium(max_iter, dt_imag, tol)` - 熱平衡状態探索 / Find thermal equilibrium
- `compute_properties()` - 物理量計算（凝縮率、エネルギーなど） / Compute physical properties

### BogoliubovSolver1D

Bogoliubov-de Gennes準粒子励起ソルバー / Bogoliubov-de Gennes quasi-particle solver

**主要メソッド / Main Methods:**

- `solve_spectrum(n_modes)` - 励起スペクトル計算 / Solve excitation spectrum
- `thermal_density(temperature)` - 熱雲密度計算 / Compute thermal cloud density
- `condensate_fraction(temperature)` - 凝縮率計算 / Compute condensate fraction

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

### Zaremba-Nikuni-Griffin (ZNG) Theory

有限温度では、凝縮成分と熱成分を分離して記述します:

At finite temperature, we separate the condensate and thermal components:

**凝縮成分 / Condensate:**
```
iℏ ∂ψ_c/∂t = [H_0 + 2g(n_c + 2ñ)]ψ_c - iR_{12}
```

**熱成分 / Thermal cloud:**
- Bogoliubov-de Gennes準粒子で記述 / Described by Bogoliubov-de Gennes quasi-particles
- 熱密度: ñ(x) = Σ_j n_j |v_j|² / Thermal density
- n_j = 1/(exp(E_j/kT) - 1) (Bose-Einstein分布 / distribution)

**衝突項 / Collision integral:**
- R_{12}: 凝縮と熱成分の粒子・エネルギー交換 / Particle and energy exchange
- 熱平衡への緩和を記述 / Describes relaxation to thermal equilibrium

### Bogoliubov Excitation Spectrum

準粒子励起エネルギー / Quasi-particle excitation energy:
```
E_k = √(ε_k(ε_k + 2gn₀))
```

where:
- ε_k = ℏ²k²/2m + V (単粒子エネルギー / single-particle energy)
- 低運動量極限: E_k ≈ ℏck (音速 / sound velocity c = √(gn₀/m))
- 高運動量極限: E_k ≈ ℏ²k²/2m (自由粒子 / free particle)

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
4. E. Zaremba, T. Nikuni, and A. Griffin, "Dynamics of Trapped Bose Gases at Finite Temperatures", J. Low Temp. Phys. 116, 277 (1999)
5. N.P. Proukakis and B. Jackson, "Finite-temperature models of Bose-Einstein condensation", J. Phys. B 41, 203002 (2008)

## ライセンス / License

MIT License

## 貢献 / Contributing

プルリクエストを歓迎します / Pull requests are welcome!

## 作者 / Author

Created with Claude AI

## TODO / 今後の開発予定

- [x] Finite temperature BEC (ZNG theory) / 有限温度BEC（ZNG理論）
- [x] Bogoliubov excitation spectrum / Bogoliubov励起スペクトル
- [ ] 3D solver implementation / 3次元ソルバーの実装
- [ ] Rotating BEC / 回転BECのサポート
- [ ] Multi-component BEC / 多成分BEC
- [ ] 2D finite temperature / 2次元有限温度
- [ ] Stochastic GP equation / 確率的GP方程式
- [ ] GPU acceleration / GPU加速
- [ ] Animation export / アニメーション出力機能
- [ ] Jupyter notebook examples / Jupyterノートブック例
