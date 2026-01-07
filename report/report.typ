#import "lib/template.typ": conf
#import "lib/helpers.typ": *

#set text(lang: "en")

#show: doc => conf(
  title: [Technical Report: Modernization and Optimization of Algorithms in Robótica Computacional],
  abstract: [
    This document details the software re-engineering process applied to the practical assignments of the Robótica Computacional course. It describes the evolution from basic procedural scripts towards interactive, modular applications built with modern Python. The covered areas include Forward Kinematics via web interfaces (Streamlit), Inverse Kinematics with physical constraints (CCD), and probabilistic localization methods (Grid Localization and Particle Filters). The result is a suite of tools that facilitates both didactic visualization and rigorous experimentation.
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
  doc,
)

= Introduction

Robotics combines complex mathematical foundations with the need for clear visual representation to validate agent behavior. The original repository provided a functional base (`original/`), but it was limited in terms of interactivity, scalability, and software engineering best practices.

The objective of this project has been to rewrite these modules by applying modern paradigms:
1. **Object-Oriented Programming (OOP)** to encapsulate the state of robots and particles.
2. **Graphical User Interfaces (GUI)** using web and desktop libraries for real-time parameter manipulation.
3. **Static Typing and External Configuration** to improve code maintainability and robustness.

The following sections expose the technical differences and improvements implemented in each of the four fundamental assignments.

= Forward Kinematics

The original implementation (`cin_dir_1.py`) consisted of a rigid script where Denavit-Hartenberg (DH) parameters were *hard-coded* into the source code. Visualization was handled via a static blocking `matplotlib` window.

== New Implementation: `main.py`

A complete web application has been developed using the **Streamlit** framework @streamlit, transforming a static calculator into a dynamic visual editor.

=== Key Features
- **Real-Time Tabular Editor**: `st.data_editor` has been integrated to allow modification of the DH table (parameters $d$, $theta$, $a$, $alpha$) directly from the browser.
- **Safe Expression Evaluation**: Unlike the dangerous use of `eval()`, a parser based on Abstract Syntax Trees (AST) (`ast.parse`) has been implemented. This allows the user to safely input mathematical expressions such as `pi/2` or `L1 + 5`.
- **Branching Support**: A `parent` column was introduced in the robot definition, enabling the modeling of tree-like structures rather than just serial kinematic chains.
- **Non-Blocking Visualization**: The 3D rendering updates reactively to any parameter change, displaying local coordinate systems ($O_n$) to facilitate debugging.

#figure(
  image("data/images/fk_streamlit.png", width: 90%),
  caption: [Interface of `main.py`. On the left, controls and the editable DH table; on the right, the dynamically generated 3D visualization.],
)

= Inverse Kinematics

The original script (`ccdp3.py`) provided a rudimentary implementation of the **Cyclic Coordinate Descent (CCD)** algorithm. While it demonstrated the mathematical concept, it lacked practical features such as joint limits or support for different joint types, and the visualization was minimal.

== Advanced CCD Implementation: `ik.py`

The rewritten module `ik.py` elevates the solver to a production-ready utility using **Typer** @typer for the Command Line Interface (CLI) and **TOML** for configuration management.

=== Algorithmic Enhancements
1. **Joint Type Abstraction**: The solver now explicitly handles both **Revolute (R)** and **Prismatic (P)** joints.
  - For *Revolute* joints, the algorithm minimizes the angle error using vector cross products.
  - For *Prismatic* joints, it projects the target error vector onto the joint's translation axis to calculate the required extension.
2. **Constraints Handling**: Physical robots have limits. The new implementation enforces `[min, max]` constraints for every joint at each iteration of the CCD loop, preventing unrealistic solutions (e.g., self-collisions or backward elbow bends).

=== Visual Debugging
To visualize the convergence, an **"Onion Skin"** animator class was developed. This tool draws semi-transparent trails of the robot's previous configurations, allowing the user to see exactly how the algorithm iteratively reaches the target.

// #figure(
//   image("data/images/ik_onion.png", width: 80%),
//   caption: [Output of `ik.py`. The "onion skin" trails illustrate the iterative movement of the links (colored segments) towards the target position (black star).],
// )

= Localization (Grid Localization)

The original approach in `localizacion.py` was a comparative odometry simulation without a true grid-based probabilistic sensor fusion implementation.

== Histogram Filter: `localization.py`

A `GridLocalization` class has been implemented, which discretizes the workspace into a probability matrix $P(x,y)$.

- **Prediction Step (Motion)**: Modeled via a convolution operation. When the robot reports movement, the probability matrix is shifted (`scipy.ndimage.shift`) and a Gaussian filter (`gaussian_filter`) is applied to represent the increase in uncertainty (motion noise).
- **Correction Step (Sensing)**: Upon perceiving a landmark, a *Likelihood Field* is generated based on the measured distance. This map is multiplied by the current belief (Bayes' Theorem), concentrating probability in areas compatible with the sensor measurement.

As a result, the system can recover the robot's actual position (magenta cross) even when the internal odometry (red line) has diverged significantly.

// #figure(
//   image("images/localization_grid.png", width: 75%),
//   caption: [Probability heatmap. Dark areas indicate high certainty. The correction of the estimated position versus the erroneous odometry is clearly visible.],
// )

= Particle Filter (Monte Carlo)

The final assignment combines all previous concepts. The script `filtro.py` rewrites the original logic, which was scattered across multiple files (`pfbase.py`, `robot.py`), into a single, cohesive, and optimized module.

== Implemented Optimizations

1. **Tricycle Motion Model**: Support for non-holonomic kinematics (Ackermann/Tricycle type) was added via the `move_triciclo` method, allowing for the simulation of car-like vehicles.
2. **Low Variance Resampling**: The *Low Variance Resampling* algorithm @thrun2005probabilistic was implemented. This method is linear in complexity $O(N)$ and preserves particle diversity better than simple random resampling, preventing the "robot kidnapping" problem (particle depletion).
3. **Real-Time Visualization**: `matplotlib` is used in interactive mode (`ion()`) to render the particle cloud at 10 FPS, allowing observation of the distribution's convergence (green points) onto the robot's real pose.

// #figure(
//   image("images/pf_particles.png", width: 75%),
//   caption: [MCL Simulation. The particle cloud (green) approximates the robot's real position (red arrow) based on landmarks (black squares).],
// )

= Conclusion

The modernization of these scripts not only improves the user experience but also reinforces the understanding of the underlying algorithms. Separation of concerns, the use of vectorized matrix calculations with `numpy`, and interactive visualization provide a robust environment for experimentation in mobile robotics.

#pagebreak()
#bibliography("bibliography.bib")
