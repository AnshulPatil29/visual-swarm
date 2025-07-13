import numpy as np
import matplotlib.pyplot as plt
import argparse
from visual_swarm import ParticleSwarm

def run_2d_example(save_path=None):
    """
    Runs a 2D constrained optimization example.

    Args:
        save_path (str, optional): If provided, the animation is saved to this
                                   file path instead of being shown interactively.
                                   Defaults to None.
    """
    def fitness_function(x: float, y: float) -> float:
        return -((x - 3)**2 + (y - 2)**2) + 10

    def constraint1(x: float, y: float) -> bool:
        return x + y <= 4 


    bounds = np.array([[0.0, 5.0], [0.0, 5.0]])
    pso_optimizer = ParticleSwarm(
        num_particles=50,
        fitness_function=fitness_function,
        bounds=bounds,
        constraints=[constraint1],
        constraint_penalty=1000.0,
        c1=1.5,
        c2=1.5,
        inertia=0.7
    )

    print("Starting 2D PSO optimization...")
    
    should_save = save_path is not None

    ani = pso_optimizer.create_animation(
        iterations=100, 
        fps=24, 
        show_grid=True, 
        save=should_save, 
        save_path=save_path
    )

    print("\nOptimization Finished.")
    print(f"Global Best Particle: {pso_optimizer.global_best_particle}")
    print(f"Global Best Fitness: {pso_optimizer.global_best_fitness}")
    print("\nExpected constrained optimum is around x=2.5, y=1.5 with fitness 9.5")
    gbp = pso_optimizer.global_best_particle
    print(f"Constraint check for global best: x+y = {gbp[0] + gbp[1]:.4f} (should be <= 4)")

    if not should_save:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D constrained PSO example and optionally save the animation.")
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help="Path to save the animation file (e.g., '2d_animation.mp4')."
    )
    args = parser.parse_args()
    
    run_2d_example(save_path=args.save)