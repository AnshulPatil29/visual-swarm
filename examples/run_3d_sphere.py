import numpy as np
import matplotlib.pyplot as plt
import argparse
from visual_swarm import ParticleSwarm

def run_3d_example(save_path=None):
    """
    Runs a 3D optimization example using the negative sphere function.

    Args:
        save_path (str, optional): If provided, the animation is saved to this
                                   file path instead of being shown interactively.
                                   Defaults to None.
    """
    def negative_sphere_function(x, y, z):
        return -(x**2 + y**2 + z**2)

    num_particles_3d = 100
    bounds_3d = np.array([[-5, 5], [-5, 5], [-5, 5]])

    pso_3d = ParticleSwarm(
        num_particles=num_particles_3d,
        fitness_function=negative_sphere_function,
        bounds=bounds_3d,
        inertia=0.7,
        c1=1.5,
        c2=2.0
    )

    print("Running 3D PSO Animation...")
    
    should_save = save_path is not None

    ani_3d = pso_3d.create_animation(
        iterations=150, 
        fps=20, 
        save=should_save, 
        save_path=save_path
    )

    print("\n3D Optimization Finished.")
    print(f"Best solution found: ({pso_3d.global_best_particle[0]:.4f}, {pso_3d.global_best_particle[1]:.4f}, {pso_3d.global_best_particle[2]:.4f})")
    print(f"Best fitness: {pso_3d.global_best_fitness:.4f}")
    
    if not should_save:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D PSO example and optionally save the animation.")
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help="Path to save the animation file (e.g., '3d_animation.mp4')."
    )
    args = parser.parse_args()
    
    run_3d_example(save_path=args.save)