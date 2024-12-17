from busemann_flow import wavetrapper_inlet_simple, wavetrapper_inlet 
from functools import partial
import numpy as np # type: ignore
from scipy.optimize import differential_evolution # type: ignore
import warnings
import sys
import os


def main():
    """ 
    Main function of the code. Runs Scipy's differential evolution algorithm to produce an inlet which matches the desired parameters.
    Freestream mach, inlet diameter, isolator diameter, and offset percentage should be modified to the target design parameter.
    """
    global freestream_mach
    global inlet_diameter
    global isolator_diameter
    global offset

    # Target design parameters
    freestream_mach = 6.0
    inlet_diameter = 0.30302
    isolator_diameter = 0.1016
    offset = 0.9

    # Display the target design
    print(f'Target Design:\n\tFreestream Mach: {freestream_mach}\n\tInlet Diameter: {inlet_diameter}')
    print(f'Given Parameters:\n\tIsolator Diameter: {isolator_diameter}\n\tOffset Percentage: {offset*100}%')

    # Set up bounds for the optimizer to search within
    bounds = [(1.5, freestream_mach*0.75), # bounds for mach_2
              (5, 35),              # bounds for shock_angle
              (0, 10)]              # bounds for truncation_angle

    print('\nCalculating wavetrapper solution...')
    
    # Simplify the objective function so the differential evolution function can accept it as an argument
    simplified_objective = partial(objective, isolator_diameter=isolator_diameter, offset=offset, freestream_mach=freestream_mach, inlet_diameter=inlet_diameter)

    # Solve the problem
    solution = differential_evolution(simplified_objective, bounds=bounds, strategy='best1bin', workers=-1, updating='deferred', disp=True, atol=1e-6)

    print('\nDone.\n')

    if solution.success:
        # Display the successful design
        print(f'Message: {solution.message}')
        print(f'Input Parameters:\n\tMach 2: {solution.x[0]}\n\tShock Angle: {solution.x[1]}\n\tTruncation Angle: {solution.x[2]}')

        design = wavetrapper_inlet_simple(solution.x[0], solution.x[1], 1.4, isolator_diameter/2, offset*isolator_diameter/2, solution.x[2])
        print(f'Design Properties:\n\tFreestream Mach: {design[0]}\n\tInlet Diameter: {design[1]}\n\tIsolator Mach: {design[2]}\n\tStagnation Pressure Ratio: {design[3]}')

        print('')

        # Generate the full inlet CSM file
        sys.stdout.write('Generating CSM file... \n')
        wavetrapper_inlet(mach_2=solution.x[0], theta_23_degrees=solution.x[1], gamma=1.4, isolator_radius=isolator_diameter/2, offset=offset*isolator_diameter/2, truncation_angle=solution.x[2])
        sys.stdout.write('done.\n')

        # Display where the CSM file is at
        cwd = os.getcwd()
        print(f'CSM located at: {cwd}\\busemann.csm')

    else:
        # The solver failed to find a solution that met the constraints
        print("Could not find a solution.")


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
        design = wavetrapper_inlet_simple(mach_2, theta_23_degrees, 1.4, isolator_diameter/2, offset*isolator_diameter/2, truncation_angle)
        observed_mach = design[0]
        observed_capture_diameter = design[1]
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
    main()