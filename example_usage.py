from busemann_flow import Inlet
from meshing import InletMesh
from wavetrapper_design import InletDesign


def main():
    ## Example 1: Prescribing design objectives and finding a wavetrapper that meets them
    design = InletDesign(freestream_mach = 6.0, inlet_diameter = 0.30302, isolator_diameter = 0.1016, offset = 0.9)

    # Run the differential evolution algorithm to find one that meets objectives
    design.calculate()

    # Show the wavetrapper inlet
    design.inlet.wavetrapper_inlet(show_plot=True)
    


if __name__ == '__main__':
    main()