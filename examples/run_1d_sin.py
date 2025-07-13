import numpy as np
import matplotlib.pyplot as plt
import argparse
from visual_swarm import ParticleSwarm

def run_1d_example(save_path=None):
    """
    Runs a 1D optimization example using the sine function.

    Args:
        save_path (str, optional): If provided, the animation is saved to this
                                   file path instead of being shown interactively.
                                   Defaults to None.
    """
    def sin_function(x):
        return np.sin(x)

    num_particles_1d = 20
    bounds_1d = np.array([[-np.pi, np.pi]])

    pso_1d = ParticleSwarm(
        num_particles=num_particles_1d,
        fitness_function=sin_function,
        bounds=bounds_1d,
        inertia=0.5,
        c1=1.5,
        c2=1.5,
        is_global=False,
        neighborhood_size=5
    )

    print("Running 1D PSO Animation...")
    
    should_save = save_path is not None
    
    ani_1d = pso_1d.create_animation(
        iterations=100, 
        fps=24, 
        save=should_save, 
        save_path=save_path
    )

    print("\n1D Optimization Finished.")
    print(f"Best solution found: {pso_1d.global_best_particle[0]:.4f}")
    print(f"Best fitness: {pso_1d.global_best_fitness:.4f}")

    if not should_save:
        # If not saving, show the animation interactively.
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 1D PSO example and optionally save the animation.")
    parser.add_argument(
        '--save', 
        type=str, 
        default=None,
        help="Path to save the animation file (e.g., '1d_animation.mp4')."
    )
    args = parser.parse_args()
    
    run_1d_example(save_path=args.save)