import numpy as np
from typing import Callable, List, Union
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path

def validate_save_path(path_str:str) -> tuple[bool,str]:
    """
    Validates the save path for an animation.
    Checks for valid extension, directory existence, and write permissions.
    Args:
        path_str (str): The path to validate.
    Returns:
        (bool, str): A tuple containing a boolean for validity and a message.
    """
    supported_formats=['.mp4', '.gif', '.html']
    path = Path(path_str)
    if path.suffix.lower() not in supported_formats:
        return False, f"Unsupported format '{path.suffix}'. Please use one of {supported_formats}."
    if not path.parent.exists():
        return False, f"Directory '{path.parent}' does not exist."
    if path.exists():
        return False, f"File '{path}' already exists. Please choose a different name or delete it."
    if not os.access(path.parent, os.W_OK):
        return False, f"Cannot write to directory '{path.parent}' due to permissions."
    return True,"Valid path."

class ParticleSwarm:
    """
    An implementation of the Particle Swarm Optimization (PSO) algorithm.

    This class supports both global best (gbest) and local best (lbest) PSO,
    handles constraints with penalties, and can generate 1D, 2D, and 3D
    animations of the optimization process.
    """
    def __population_initializer(self)->np.ndarray:
        """Initializes the swarm population within the specified bounds.

        Returns:
            np.ndarray: A numpy array of shape (num_particles, num_parameters)
                        containing the initial positions of all particles.
        """
        lower_bounds,upper_bounds=self.bounds.T
        difference=upper_bounds-lower_bounds
        population=(np.random.random(size=(self.num_particles,self.num_parameters))*difference)+lower_bounds
        return population.astype(np.float32)

    def __constraint_param_checker(self,constraints:List[Callable]):
        """Checks if constraint functions have the correct number of arguments.

        Args:
            constraints (List[Callable]): A list of constraint functions.

        Returns:
            bool: True if all constraint functions are valid, False otherwise.
        """
        for constraint in constraints:
            if (constraint.__code__.co_argcount!=self.num_parameters):
                return False
        return True

    @staticmethod
    def __apply_function_rowwise(population:np.ndarray,func:Callable):
        """Applies a function to each row (particle) of the population.

        Args:
            population (np.ndarray): The population of particles.
            func (Callable): The function to apply to each particle.

        Returns:
            np.ndarray: An array containing the result of the function for each particle.
        """
        return np.array([func(*particle) for particle in population],dtype=np.float32)

    def __init__(self,
                 num_particles:int,
                 fitness_function:Callable,
                 bounds:np.ndarray,
                 constraints:List[Callable]=None,
                 constraint_penalty:Union[float, list, np.ndarray] =1000.0,
                 c1:float=1.0,
                 c2:float=1.0,
                 inertia:float=0.5,
                 is_global:bool = True,
                 neighborhood_size:int = None
                 ):
        """Initializes the Particle Swarm Optimization algorithm.

        Args:
            num_particles (int): The number of particles in the swarm.
            fitness_function (Callable): The function to be maximized.
            bounds (np.ndarray): A numpy array of shape (num_parameters, 2)
                                 defining the search space bounds for each parameter.
            constraints (List[Callable], optional): A list of functions that
                                 particles must satisfy. Each function should return
                                 True if the constraint is met, False otherwise. Defaults to None.
            constraint_penalty (Union[float, list, np.ndarray], optional):
                                 The penalty subtracted from fitness for violating
                                 constraints. Defaults to 1000.0.
            c1 (float, optional): The cognitive coefficient, influencing the pull
                                  towards a particle's personal best. Defaults to 1.0.
            c2 (float, optional): The social coefficient, influencing the pull
                                  towards the swarm's best known position. Defaults to 1.0.
            inertia (float, optional): The inertia weight, controlling the momentum
                                     of the particles. Defaults to 0.5.
            is_global (bool, optional): If True, uses the global best PSO (gbest).
                                      If False, uses the local best PSO (lbest). Defaults to True.
            neighborhood_size (int, optional): The size of the neighborhood for
                                             lbest PSO. Required if is_global is False.
                                             Defaults to None.
        """
        self.num_particles=num_particles
        self.fitness_function=fitness_function
        self.num_parameters=fitness_function.__code__.co_argcount
        assert self.num_parameters>0, "The fitness function must have atleast 1 parameter to optimize"
        assert bounds.ndim>1 and bounds.shape==(self.num_parameters,2),f"The bounds must be of shape {(self.num_parameters,2)}."
        assert num_particles>0,"Number of particles must be a positive integer."
        assert c1>=0,"Cognitive coefficient (c1) cannot be negative."
        assert c2>=0,"Social coefficient (c2) cannot be negative."
        assert inertia>=0,"Inertia cannot be negative."
        lower_bounds,upper_bounds=bounds.T
        assert np.all(lower_bounds<upper_bounds),"All lower bounds must be strictly less than upper bounds."


        self.is_global = is_global
        if not self.is_global:
            assert neighborhood_size is not None and isinstance(neighborhood_size, int), "Neighborhood size must be an integer for local PSO."
            assert 0 < neighborhood_size <= self.num_particles, "Neighborhood size must be between 1 and the number of particles."
            self.neighborhood_size = neighborhood_size

        self.bounds=bounds
        self.population=self.__population_initializer()

        try:
        # Test if the function can process a batch of particles
            self.fitness_function(*self.population.T)
            self._is_vectorized = True
        except TypeError:
            self._is_vectorized = False
        self.constraints=None
        if constraints:
            assert self.__constraint_param_checker(constraints) , "Incorrect argument count for constraints"
            self.constraints=constraints
            if isinstance(constraint_penalty,(int,float)):
                self.constraint_penalty=np.array([constraint_penalty]*len(constraints))
            elif len(constraints)==len(constraint_penalty) and isinstance(constraint_penalty,(list,tuple,set,np.ndarray)):
                self.constraint_penalty=np.array(constraint_penalty)
            else:
                raise AssertionError(f'Number of constraints and constraint penalties do not match {len(constraint_penalty)}!={len(constraints)}, or incorrect data type {type(constraint_penalty)}')
            if self.constraints:
                assert np.all(self.constraint_penalty>=0),"Constraint penalties cannot be negative."
        else:
            self.constraints=None
        self.velocity=np.zeros_like(self.population,dtype=np.float32)
        self.c1=c1
        self.c2=c2
        self.inertia=inertia
        # local state
        self.personal_best=self.population.copy()
        self._update_fitness()
        self.personal_best_fitness=self.fitness.copy()
        _best_idx=np.argmax(self.personal_best_fitness)
        self.global_best_fitness=self.personal_best_fitness[_best_idx]
        self.global_best_particle=self.population[_best_idx]

        if not self.is_global:
            self.local_best_particles = np.zeros_like(self.population)
            self._update_local_best()

    def _update_velocity(self):
        """Updates the velocity of each particle."""
        shape=self.population.shape
        inertial_component=self.velocity*self.inertia
        cognitive_component=(self.c1*(self.personal_best-self.population))*np.random.rand(*shape)
        
        if self.is_global:
            social_attractor = self.global_best_particle
        else:
            social_attractor = self.local_best_particles

        social_component=(self.c2*(social_attractor-self.population))*np.random.rand(*shape)
        self.velocity=inertial_component+cognitive_component+social_component
        return

    def _update_fitness(self):
        """Calculates the fitness of each particle in the swarm."""
        # unconstrainted fitness score
        if self._is_vectorized:
            fitness = self.fitness_function(*self.population.T)
        else:
            fitness=ParticleSwarm.__apply_function_rowwise(self.population,self.fitness_function)
        # penalizing for constraints
        if self.constraints:
            for constraint,penalty_multiplier in zip(self.constraints,self.constraint_penalty):
                failed=ParticleSwarm.__apply_function_rowwise(self.population,constraint)
                # since this returns a numpy array I need to turn 1s into 0s and vice versa
                failed=-(failed-1.0)
                fitness-=(failed*penalty_multiplier)
        self.fitness=fitness
        return

    def _update_local_best(self):
        """Updates the local best for each particle based on its neighborhood."""
        half_k=self.neighborhood_size//2
        indices=np.arange(self.num_particles)

        for i in range(self.num_particles):
            neighbor_indices=np.mod(np.arange(i-half_k,i+half_k+1), self.num_particles)
            best_neighbor_local_idx=np.argmax(self.personal_best_fitness[neighbor_indices])
            best_neighbor_global_idx=neighbor_indices[best_neighbor_local_idx]
            self.local_best_particles[i]=self.personal_best[best_neighbor_global_idx]

    def _update_best_particles(self):
        """Updates the personal, local, and global best positions."""
        # updating personal bests
        for idx,(particle_fitness,personal_best_fitness) in enumerate(zip(self.fitness,self.personal_best_fitness)):
            if particle_fitness>personal_best_fitness:
                self.personal_best[idx]=self.population[idx]
                self.personal_best_fitness[idx]=self.fitness[idx]
        
        if not self.is_global:
            self._update_local_best()

        iter_best_idx=np.argmax(self.fitness)
        if self.fitness[iter_best_idx]>self.global_best_fitness:
            self.global_best_fitness=self.fitness[iter_best_idx]
            self.global_best_particle=self.population[iter_best_idx]
        return

    def _update_position(self):
        """Updates particle positions and enforces search space bounds."""
        self.population+=self.velocity
        lower_bounds,upper_bounds=self.bounds.T
        self.population=np.clip(self.population,lower_bounds,upper_bounds)
        return

    def update(self, verbose=False):
        """Performs a single iteration of the PSO algorithm.

        Args:
            verbose (bool, optional): If True, prints the average fitness and
                                      global best fitness for the iteration.
                                      Defaults to False.
        """
        self._update_velocity()
        self._update_position()
        self._update_fitness()
        self._update_best_particles()
        if verbose:
            average_fitness=np.average(self.fitness)
            print(f"Average Fitness: {average_fitness} \t Global Best: {self.global_best_fitness}")

    def run(self,iterations:int=50,verbose:bool=False,verbose_interval:int=10)->None:
        """Runs the PSO algorithm for a specified number of iterations.

        Args:
            iterations (int, optional): The number of iterations to run the
                                        algorithm. Defaults to 50.
            verbose (bool, optional): If True, prints periodic status updates.
            verbose_interval (int, optional): The interval of iterations between
                                              status updates.
        """
        assert iterations>=0, "Number of iterations cannot be negative."
        assert verbose_interval>0,"Verbose interval must be a positive integer."
        for it in range(iterations):
            if verbose and it%verbose_interval==0:
                self.update(verbose=True)
            else:
                self.update()
        return


    def __create_1D_animation(self, iterations, fps, particle_size):
        """Creates a 1D animation of the PSO process.

        Args:
            iterations (int): The number of animation frames (iterations).
            fps (int): The frames per second for the animation.
            particle_size (float): The size of the plotted particles.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        fig, ax=plt.subplots()
        ax.set_xlim(self.bounds[0, 0],self.bounds[0, 1])
        ax.set_title("Particle Swarm Optimization 1D")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Fitness")

        # Plot the fitness function curve
        x_coords = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 200)
        try:
            # Try to compute fitness in a vectorized way
            y_coords=self.fitness_function(x_coords)
        except TypeError:
            # If function is not vectorized, compute point-by-point
            y_coords=np.array([self.fitness_function(x) for x in x_coords])
        ax.plot(x_coords,y_coords,label='Fitness Function',color='gray',linestyle='--',zorder=1)
        ax.set_ylim(np.min(y_coords)*1.1,np.max(y_coords)*1.1)

        # Initial scatter plots
        scatterplot_reg=ax.scatter(self.population, self.fitness,s=particle_size,c='blue',alpha=0.7,label='Particles',zorder=5)
        scatterplot_best=ax.scatter(self.global_best_particle,self.global_best_fitness,s=particle_size*2,c='red',marker="*",label='Global Best',zorder=10)
        ax.legend()

        def animation_update(frame):
            self.update()
            # Update particle positions (x=parameter, y=fitness)
            particle_data = np.c_[self.population.flatten(),self.fitness]
            scatterplot_reg.set_offsets(particle_data)
            # Update global best position
            best_particle_data=np.c_[self.global_best_particle,self.global_best_fitness].flatten()
            scatterplot_best.set_offsets(best_particle_data)
            return (scatterplot_reg, scatterplot_best)

        ani = animation.FuncAnimation(fig, animation_update, frames=iterations, interval=1000/fps, blit=True)
        return ani

    def __create_2D_animation(self,iterations,fps,particle_size,show_grid=False):
        """Creates a 2D animation of the PSO process.

        Args:
            iterations (int): The number of animation frames (iterations).
            fps (int): The frames per second for the animation.
            particle_size (float): The size of the plotted particles.
            show_grid (bool, optional): If True, displays the fitness landscape
                                        as a contour plot. Defaults to False.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(self.bounds[0,0],self.bounds[0,1])
        ax.set_ylim(self.bounds[1,0],self.bounds[1,1])
        ax.set_title("Particle Swarm Optimization 2D")
        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")
        # first frame
        if show_grid:
            try:
                grid_resolution = 100
                x_coords=np.linspace(self.bounds[0, 0],self.bounds[0, 1],grid_resolution)
                y_coords=np.linspace(self.bounds[1, 0],self.bounds[1, 1],grid_resolution)
                X_grid, Y_grid=np.meshgrid(x_coords, y_coords)
                Z_grid=self.fitness_function(X_grid, Y_grid)
                contour_plot=ax.contourf(X_grid,Y_grid,Z_grid,levels=20,cmap='viridis',alpha=0.7)
                fig.colorbar(contour_plot,ax=ax,label='Fitness Value')
            except:
                print("The fitness function was not vectorized, cannot show grid.")
                pass

        scatterplot_reg=ax.scatter(*self.population.T,s=particle_size,c='blue',alpha=0.7,label='Particles',zorder=5)
        scatterplot_best=ax.scatter(*self.global_best_particle.T,s=particle_size*1.5,c='red',alpha=1,label='Global Best',marker="*",zorder=10)
        ax.legend()

        def animation_update(frame):
            self.update()
            # this method is designed differently as PathCollections.set_offsets expects data in N,2 shape
            scatterplot_reg.set_offsets(self.population)
            scatterplot_best.set_offsets(self.global_best_particle)
            return (scatterplot_reg,scatterplot_best)

        ani=animation.FuncAnimation(fig,
                                    animation_update,
                                    frames=iterations,
                                    interval=1000/fps,
                                    blit=True)
        return ani

    def __create_3D_animation(self, iterations, fps, particle_size):
        """Creates a 3D animation of the PSO process.

        Args:
            iterations (int): The number of animation frames (iterations).
            fps (int): The frames per second for the animation.
            particle_size (float): The size of the plotted particles.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(self.bounds[0,0],self.bounds[0,1])
        ax.set_ylim(self.bounds[1,0],self.bounds[1,1])
        ax.set_zlim(self.bounds[2,0],self.bounds[2,1])
        ax.set_title("Particle Swarm Optimization 3D")
        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")
        ax.set_zlabel("Parameter 3")

        # Initial scatter plots for particles and global best
        scatterplot_reg=ax.scatter(*self.population.T, s=particle_size, c='blue', alpha=0.7, label='Particles')
        scatterplot_best=ax.scatter(*self.global_best_particle.T, s=particle_size*2, c='red', marker="*", label='Global Best')
        ax.legend()

        def animation_update(frame):
            self.update()
            # For 3D scatter plots, update the _offsets3d property
            scatterplot_reg._offsets3d=self.population.T
            scatterplot_best._offsets3d=self.global_best_particle.T.reshape(3,1)
            return (scatterplot_reg,scatterplot_best)

        ani=animation.FuncAnimation(fig, animation_update,frames=iterations,interval=1000/fps,blit=False)
        return ani

    def create_animation(self, iterations: int = 50, fps: int = 24, show_grid:bool=False,save:bool=False,save_path:str='swarm_animation.mp4'):
        """Creates and returns an animation of the PSO process.

        The dimensionality of the animation (1D, 2D, or 3D) is determined
        automatically from the number of parameters in the fitness function.

        Args:
            iterations (int, optional): The number of animation frames. Defaults to 50.
            fps (int, optional): The frames per second. Defaults to 24.
            show_grid (bool, optional): If True and dimensionality is 2D,
                                        a contour plot of the fitness landscape
                                        is shown. Defaults to False.
            save (bool, optional): If True, saves the animation to a file. Defaults to False.
            save_path (str, optional): The path to save the animation file (must be .mp4).
                                       Defaults to 'swarm_animation.mp4'.

        Returns:
            matplotlib.animation.FuncAnimation: The generated animation object,
                                                or None if dimensionality is > 3.
        """
        assert iterations>=0, "Number of iterations cannot be negative."
        assert fps>0, "FPS must be a positive integer."
        if self.num_parameters>3:
            print("Animation is only supported for 1, 2, or 3 dimensions.")
            return

        base_size=100
        # Adjust particle size based on the number of particles for better visibility
        particle_size=max(5,base_size/np.sqrt(self.num_particles))

        if self.num_parameters==1:
            ani=self.__create_1D_animation(iterations,fps,particle_size)
        elif self.num_parameters==2:
            ani=self.__create_2D_animation(iterations,fps,particle_size, show_grid)
        elif self.num_parameters==3:
            # Note: Blitting can be problematic with 3D plots in matplotlib.
            # It's often disabled for stability. And idk how to make 3d contour grid
            ani=self.__create_3D_animation(iterations,fps,particle_size)
        
        if save:
            is_valid, message = validate_save_path(save_path)
            if not is_valid:
                print(f"Error: Could not save animation. {message}")
                return ani

            output_path = Path(save_path)
            file_suffix = output_path.suffix.lower()

            try:
                if file_suffix == '.mp4':
                    if FFMpegWriter.isAvailable():
                        print(f"Saving MP4 animation to {output_path} (using FFmpeg)...")
                        ani.save(str(output_path),writer='ffmpeg',fps=fps)
                        print("Save complete.")
                    else:
                        html_path = output_path.with_suffix('.html')
                        print("Warning: FFmpeg is not installed. Cannot save as .mp4.")
                        print(f"Falling back to save as a self-contained HTML file at: {html_path}")
                        if html_path.exists():
                             print(f"Error: Fallback path '{html_path}' also exists. Please clear it and try again.")
                             return ani
                        ani.save(str(html_path),writer='html',fps=fps)
                        print("Save complete. Open the .html file in a web browser.")
                elif file_suffix=='.gif':
                    print(f"Saving GIF animation to {output_path} (using Pillow)...")
                    ani.save(str(output_path),writer='pillow',fps=fps)
                    print("Save complete.")
                else: # .html
                    print(f"Saving HTML animation to {output_path}")
                    ani.save(str(output_path),writer='html',fps=fps)
            except Exception as e:
                print(f"An error occurred while saving the animation: {e}")
        return ani