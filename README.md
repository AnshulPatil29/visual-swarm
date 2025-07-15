# Visual Swarm

**A modern, visualization-first Particle Swarm Optimization (PSO) library for Python.**

Visual Swarm makes it easy to not only *run* PSO but to *see* it in action. It's designed for researchers, students, and developers who want to gain an intuitive understanding of how swarms explore a search space.

![2D Animation GIF](https://raw.githubusercontent.com/AnshulPatil29/visual-swarm/refs/heads/main/assets/2d_animation.gif)
> ! The constraint line in blue has been added by editing the generated videp and is not available through code.
> The constraints may take form that I cannot account for when coding the visualization function  

---
### Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start: 2D Optimization](#quick-start-2d-optimization)
- [Advanced Usage](#advanced-usage)
  - [Saving an Animation](#saving-an-animation)
  - [Local Best PSO (lbest)](#local-best-pso-lbest)
- [AI Assistance](#ai-assistance)
- [License](#license)


### Key Features

*   **Built-in Animations:** Generate 1D, 2D, and 3D animations of the optimization process with a single line of code.
*   **Save to Video:** Save your animations directly to `.mp4` files for presentations or reports.
*   **Modern API:** A clean, object-oriented interface that's easy to extend.
*   **Flexible Topologies:** Switch between global best (gbest) and local best (lbest) PSO with a simple boolean flag.
*   **Constraint Handling:** Easily define constraints and penalties for complex optimization problems.

### Installation

To save animations, you will need **FFmpeg** installed on your system.

```bash
# On macOS (using Homebrew):
brew install ffmpeg

# On Debian/Ubuntu:
sudo apt-get install ffmpeg

# On Windows (using Chocolatey):
choco install ffmpeg
```

Once FFmpeg is installed, you can install Visual Swarm and its dependencies:

There are two ways to install Visual Swarm:
**1. For Development :**
```bash
# Clone the repository and install in editable mode for development
git clone https://github.com/AnshulPatil29/visual-swarm.git
cd visual-swarm
pip install -e .
```

**2. For General Use :**
```bash
pip install visual-swarm
```

### Quick Start: 2D Optimization

Here's how easy it is to find the peak of a 2D function and watch the swarm converge.

```python
import numpy as np
import matplotlib.pyplot as plt
from visual_swarm import ParticleSwarm

# 1. Define a fitness function to maximize
def peak_function(x, y):
    return -((x - 3)**2 + (y - 2)**2) + 10

# 2. Define the search space bounds: [[x_min, x_max], [y_min, y_max]]
bounds = np.array([[0.0, 5.0], [0.0, 5.0]])

# 3. Initialize the optimizer
pso = ParticleSwarm(
    num_particles=50,
    fitness_function=peak_function,
    bounds=bounds
)

# 4. Create the animation! This also runs the optimization.
# show_grid=True helps visualize the fitness landscape.
ani = pso.create_animation(iterations=100, show_grid=True)

# 5. Show the final results and the animation plot
print(f"Best solution found: {pso.global_best_particle}")
print(f"Best fitness: {pso.global_best_fitness:.4f}")
plt.show()
```

### Advanced Usage

#### Saving an Animation

To save the animation to a file instead of viewing it interactively, use the `save` and `save_path` arguments.

```python
# Creates a file named '2d_peak_optimization.mp4' in the current directory
pso.create_animation(
    iterations=100,
    show_grid=True,
    save=True,
    save_path='2d_peak_optimization.mp4'
)
```

You can save animations directly to a file. The library supports `.mp4`, `.gif`, and self-contained `.html` formats.

*   **MP4:** Requires **FFmpeg** to be installed on your system. If FFmpeg is not found, the library will automatically fall back to saving an `.html` file.
*   **HTML:** Works out-of-the-box with no extra dependencies. The file can be opened in any web browser.
*   **GIF:** To enable GIF support, you must install the library with the `[gif]` extra:
    ```bash
    pip install "visual-swarm[gif]"
    ```


#### Local Best PSO (lbest)

To help avoid premature convergence on complex problems, you can use the local best topology. Each particle is influenced by the best particle in its immediate neighborhood, not the entire swarm.

```python
pso_local = ParticleSwarm(
    num_particles=50,
    fitness_function=peak_function,
    bounds=bounds,
    is_global=False,          # Set to False for local best
    neighborhood_size=5       # Define how many neighbors to consider
)
```
---

### AI Assistance

To enhance productivity and ensure high-quality documentation, this project utilized AI assistance (via Google's Gemini) for the following specific tasks:

*   Generating initial docstrings based on function signatures.
*   Creating boilerplate for test case scenarios.
*   Refining grammar and phrasing in the documentation.

While AI tools helped with documentation, all core algorithmic logic, class structure, and final implementation were authored independently by the project maintainer.  

To gain insights in the learning/development process, my notes are available at [Notes](https://github.com/AnshulPatil29/Notes/blob/main/PSO.md)

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.