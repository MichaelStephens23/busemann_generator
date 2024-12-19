from busemann_flow import Inlet
from functools import partial
from scipy.optimize import differential_evolution
import warnings
import sys
import os


class InletDesign:
    def __init__(self, freestream_mach = 6.0, inlet_diameter = 0.30302, isolator_diameter = 0.1016, offset = 0.9):
        self.freestream_mach = freestream_mach 
        self.inlet_diameter = inlet_diameter
        self.isolator_diameter = isolator_diameter
        self.offset = offset
        
        self.inlet = Inlet(mach_2=0, theta_23_degrees=0, gamma=0, isolator_radius=0, offset=0, truncation_angle=0)

    def calculate(self):
        # Set up bounds for the optimizer to search within
        bounds = [(1.5, self.freestream_mach*0.75), # bounds for mach_2
                (5, 35),              # bounds for shock_angle
                (0, 10)]              # bounds for truncation_angle

        print('\nCalculating wavetrapper solution...')
        
        # Simplify the objective function so the differential evolution function can accept it as an argument
        simplified_objective = partial(objective, isolator_diameter=self.isolator_diameter, offset=self.offset, freestream_mach=self.freestream_mach, inlet_diameter=self.inlet_diameter)

        # Solve the problem
        solution = differential_evolution(simplified_objective, bounds=bounds, strategy='best1bin', workers=-1, updating='deferred', disp=True, atol=1e-6)
        if solution.success:
            # Display the successful design
            print(f'Message: {solution.message}')

            self.inlet = Inlet(solution.x[0], solution.x[1], 1.4, self.isolator_diameter/2, self.offset*self.isolator_diameter/2, solution.x[2])
            self.inlet.generate()
            self.inlet.test_design()

            print('')


def objective(x, isolator_diameter, offset, freestream_mach, inlet_diameter):
    """
    Objective function for the differential evolution optimization problem.

    Parameters
    ----------
    x : np.ndarray
        Array of optimization parameters.
    isolator_diameter : float
        Isolator diameter.
    offset : float
        Percentage of offset.
    freestream_mach : float
        Freestream Mach number.
    inlet_diameter : float
        Inlet diameter.

    Returns
    -------
    float
        Sum of squared errors between the observed and target inlet design parameters.

    """

    # Unpack the inputs
    mach_2 = x[0]
    theta_23_degrees = x[1]
    truncation_angle = x[2]

    # Turn off warnings for weird designs
    warnings.simplefilter('ignore')
    try:
        design = Inlet(mach_2, theta_23_degrees, 1.4, isolator_diameter/2, offset*isolator_diameter/2, truncation_angle)
        design.wavetrapper_inlet_simple()
        observed_mach = design.freestream_mach
        observed_capture_diameter = design.inlet_diameter
    except:
        # If the design fails to produce a solution, give the optimizer a large penalty for the input values
        observed_capture_diameter = 1e6
        observed_mach = 1e6

    warnings.simplefilter('default')

    # Calculate the sum of squared errors
    sse = (observed_mach - freestream_mach)**2 + (observed_capture_diameter - inlet_diameter)**2

    # Add in a penalty for truncation angle to drive the optimizer to a smoother inlet
    truncation_angle_penalty = truncation_angle * 0.01

    return sse + truncation_angle_penalty


if __name__ == '__main__':
    design = InletDesign()
    design.calculate()