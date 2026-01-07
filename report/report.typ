#import "lib/template.typ": conf
#import "lib/helpers.typ": *

#set text(lang: "en")

#show: doc => conf(
  title: [Technical Report: Modernization and Optimization of Algorithms in Robótica Computacional],
  abstract: [
    This document details the software re-engineering process applied to the practical assignments of the "Robótica Computacional" course. It describes the evolution from basic procedural scripts towards interactive, modular applications built with modern Python. The covered areas include Forward Kinematics via web interfaces (Streamlit), Inverse Kinematics with physical constraints (CCD), and probabilistic localization methods (Grid Localization and Particle Filters). Furthermore, it discusses the software architecture choices, including dependency management and type safety. All source code and resources developed for this project are available in the repository @repo.
  ],
  affiliations: (
    (
      id: 1,
      name: "Universidad de La Laguna",
      full: [
        Robótica Computacional, Academic Year 2025-2026. Escuela Superior de Ingeniería y Tecnología,
        Universidad de La Laguna, Santa Cruz de Tenerife, Spain.
      ],
    ),
  ),
  authors: (
    (
      name: "Pablo Hernández Jiménez",
      affiliation: [1],
      email: "alu0101495934@ull.edu.es",
      equal-contributor: false,
    ),
  ),
  accent: rgb("#5c068c"),
  logo: image("data/images/logo.svg"),
  doc,
)

= Introduction

Robotics combines complex mathematical foundations with the need for clear visual representation to validate agent behavior. The original repository provided a functional base (`original/`), but it was limited in terms of interactivity, scalability, and software engineering best practices. For instance, kinematics were hard-coded, preventing rapid experimentation, and localization scripts relied on ideal conditions or lacked rigorous probabilistic definitions.

The objective of this project has been to rewrite these modules by applying modern paradigms:
1. **Object-Oriented Programming (OOP)** to encapsulate the state of robots and particles.
2. **Graphical User Interfaces (GUI)** using web and desktop libraries for real-time parameter manipulation.
3. **Static Typing and External Configuration** to improve code maintainability and robustness.
4. **Vectorization** using NumPy and SciPy to improve computational performance in probabilistic filters.

The following sections expose the technical differences and improvements implemented in each of the four fundamental assignments.

= Forward Kinematics

The original implementation (`cin_dir_1.py`) consisted of a rigid script where Denavit-Hartenberg (DH) parameters were *hard-coded* into the source code. Visualization was handled via a static blocking `matplotlib` window.

== New Implementation: `main.py`

A complete web application has been developed using the **Streamlit** framework @streamlit, transforming a static calculator into a dynamic visual editor.

=== Implementation Details

The core of the application relies on the `matriz_T` function, which computes the homogeneous transformation matrix based on the DH convention:

$
  T = mat(
    cos(theta), -sin(theta) cos(alpha), sin(theta) sin(alpha), a cos(theta);
    sin(theta), cos(theta) cos(alpha), -sin(alpha) cos(theta), a sin(theta);
    0, sin(alpha), cos(alpha), d;
    0, 0, 0, 1
  )
$

However, the significant engineering effort lies in the surrounding infrastructure:

- **Safe Expression Evaluation**:
  Users can input values like `L1 + 5` or `cosd(45)`. Using Python's `eval()` is a security risk. A custom parser `safe_eval` was implemented using the `ast` (Abstract Syntax Tree) module. It recursively visits nodes and only permits:
  - `ast.BinOp` (Binary operations: +, -, \*, /)
  - `ast.UnaryOp` (Unary operations: -)
  - `ast.Call` (Function calls restricted to a specific whitelist: `sin`, `cos`, `tan`, `pi`, `e`, etc.)

- **Recursive Tree Traversal**:
  The original script only supported serial chains ($T_"total" = T_0 dot T_1 ...$). The new `build_tree_transforms` function supports branching (e.g., a torso splitting into two arms). It uses a dictionary to map joint indices to transformation matrices.
  $ T_"global"^(i) = T_"global"^("parent"(i)) dot T_"local"^(i) $
  This allows defining complex robots where multiple joints share the same parent frame.

- **State Management**:
  Streamlit reruns the script on every interaction. To persist the robot configuration and presets, `st.session_state` is utilized, ensuring that the user's design is not lost during re-renders.

#figure(
  image("data/images/fk_streamlit.png", width: 90%),
  caption: [Interface of `main.py`. On the left, controls and the editable DH table; on the right, the dynamically generated 3D visualization using `matplotlib`'s 3D toolkit integrated into the web view.],
)

#pagebreak()

= Inverse Kinematics

The original script (`ccdp3.py`) provided a rudimentary implementation of the **Cyclic Coordinate Descent (CCD)** algorithm. It lacked support for prismatic joints and joint limits.

== Advanced CCD Implementation: `ik.py`

The rewritten module `ik.py` elevates the solver to a production-ready utility using **Typer** @typer for the Command Line Interface (CLI) and **TOML** for configuration management.

=== Mathematical Formulation

The solver iterates from the end-effector to the base (`reversed(range(n))`). For each joint $i$, it computes the vector from the joint to the end-effector ($arrow(e)$) and to the target ($arrow(t)$).

1. **Revolute Joints (R)**:
  The algorithm minimizes the angle $phi$ between $arrow(e)$ and $arrow(t)$.
  $ cos(phi) = (arrow(e) dot arrow(t)) / (norm(arrow(e)) norm(arrow(t))), quad "axis" = arrow(e) times arrow(t) $
  The new angle becomes $theta_i' = theta_i + eta dot phi$, where $eta$ is a damping factor.

2. **Prismatic Joints (P)**:
  Unlike the original script, `ik.py` handles sliding joints. The translation axis $arrow(z)_i$ is extracted from the rotation matrix up to joint $i$. The required displacement $Delta s$ is the projection of the error vector onto this axis:
  $ Delta s = arrow(z)_i dot (arrow(t)_"pos" - arrow(e)_"pos") $
  The new extension becomes $s_i' = s_i + eta dot Delta s$.

=== Constraints and Visualization

- **Joint Limits**: A `JointLimits` dataclass was introduced. After every update step, the `clamp` function ensures $theta_"min" <= theta_i <= theta_"max"$. This prevents the solver from reaching mathematically correct but physically impossible solutions (e.g., self-collision).

- **Onion Skinning**: To visualize the *process* of convergence, the `Animator` class stores the history of all joint positions during the iteration steps. It renders past frames with decreasing alpha values (transparency), creating a motion trail effect that visualizes the descent gradient.

#figure(
  image("data/images/ik_onion.png", width: 60%),
  caption: [Output of `ik.py`. The "onion skin" trails illustrate the iterative movement of the links. The transparency gradient indicates the temporal sequence of the iterations.],
)

= Localization (Grid Localization)

The original approach in `localizacion.py` compared odometry against an ideal path but did not implement a probabilistic filter.

== Histogram Filter: `localization.py`

The `GridLocalization` class implements a discrete Bayes filter. The workspace is represented as a matrix $P$, where $P_(i j)$ is the probability of the robot being at cell $(i, j)$.

=== Optimized Implementation via Vectorization

Instead of iterating over every cell with Python loops (which is computationally expensive), the implementation uses `scipy.ndimage` and `numpy`:

1. **Prediction (Motion Model)**:
  The robot's movement $(Delta x, Delta y)$ shifts the probability distribution. This is implemented as an image convolution:
  - **Shift**: `scipy.ndimage.shift(grid, shift=[dy, dx])` moves the probability mass.
  - **Diffusion**: `scipy.ndimage.gaussian_filter(grid, sigma)` applies Gaussian blur. The `sigma` parameter represents the motion noise; higher noise results in a more blurred (uncertain) distribution.

2. **Correction (Sensor Model)**:
  When a landmark is detected at distance $z$, we compute a **Likelihood Field**.
  - A grid of distances to the landmark is precomputed: $D_(i j) = ||(x_i, y_j) - (L_x, L_y)||$.
  - The probability of the measurement given the location is modeled as a Gaussian:
    $ P(z | x_(i j)) prop 1 / (sigma_z sqrt(2 pi)) exp(- (D_(i j) - z)^2 / (2 sigma_z^2)) $
  - The update step is a simple element-wise multiplication: `self.grid *= likelihood`.

This vectorized approach allows for high-resolution grids (e.g., 100x100) to be updated in milliseconds.

#figure(
  image("data/images/localization_grid.png", width: 60%),
  caption: [Probability heatmap. Dark areas indicate high certainty. The correction of the estimated position versus the erroneous odometry is clearly visible.],
)

= Particle Filter (Monte Carlo)

The `filtro.py` script consolidates the logic into a single cohesive module, replacing the fragmented structure of the original code.

== Improvements and Algorithms

1. **Non-Holonomic Motion Model**:
  The `robot` class now includes `move_triciclo(turn, forward, length)`. This implements the kinematics of a bicycle/car model (Ackermann steering approximation):
  $ theta' = theta + d/L tan(alpha) $
  $ x' = x + d cos(theta'), quad y' = y + d sin(theta') $
  This is crucial for simulating realistic mobile robots that cannot rotate in place.

2. **Low Variance Resampling**:
  Standard random resampling can lead to *particle depletion* (loss of diversity). The implementation uses the **Low Variance Resampling** algorithm @thrun2005probabilistic:
  - A single random number $r in [0, N^(-1)]$ is chosen.
  - Particles are selected by marching through the cumulative weight distribution with step size $N^(-1)$.
  - This method is $O(N)$ and systematically covers the space of particle weights, ensuring both high-probability particles are kept and some lower-probability ones survive to maintain diversity.

3. **Statistics**:
  Methods to compute the *spatial dispersion* (standard deviation of $x$ and $y$) and the *mean weight* were added to provide quantitative metrics on the filter's convergence confidence.

#figure(
  image("data/images/pf_particles.png", width: 75%),
  caption: [MCL Simulation. The particle cloud (green) approximates the robot's real position (red arrow). The spread of the cloud visually represents the covariance of the estimate.],
)

= Software Architecture

Beyond the algorithms, the project adopts professional software engineering practices to ensure reproducibility and quality.

== Dependency Management
The project utilizes `uv` (a modern Python package manager) and `pyproject.toml`. This replaces the legacy `requirements.txt` approach. The `pyproject.toml` file explicitly defines dependencies (`numpy`, `matplotlib`, `scipy`, `typer`, `streamlit`) and requires Python $>= 3.13$, ensuring a consistent environment for all developers.

== Modern Python Features
- **Type Hinting**: Functions use type hints (e.g., `def ccd(...) -> CCDResult:`) to enable static analysis and improve IDE autocompletion.
- **Dataclasses**: Used in `ik.py` (e.g., `JointLimits`, `CCDResult`) to structure data clearly instead of using unstructured dictionaries or tuples.
- **Modularization**: Code is split into logical units (e.g., separating the math helpers from the UI logic in `main.py` via `lib/` imports is suggested, though currently self-contained for portability).

= Conclusion

The modernization of these scripts results in a robust educational toolkit. The transition from procedural, hard-coded scripts to interactive, vectorized, and type-safe applications significantly lowers the barrier to entry for understanding complex robotics concepts like CCD convergence or Bayesian updates. The use of visualization—3D plots, heatmaps, and particle clouds—provides immediate feedback, bridging the gap between mathematical theory and practical implementation.

The complete source code, including dependency definitions and documentation, is available at the project repository @repo.

#pagebreak()
#bibliography("bibliography.bib")
