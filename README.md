# Busemann & Wavetrapper Inlet Generator
**WARNING: This project is a work in progress and may produce incorrect results.**

## Description
This project provides an open source implementation of recent research in the field of hypersonic Busemann inlet design -- specifically the work done by Sannu Mölder in "The Busemann Air Intake for Hypersonic Speeds" (DOI: 10.5772/intechopen.82736). The overarching goal is to produce types of Busemann inlet modules Mölder calls "wavetrapper" inlets. This class of inlet is more suited for integration into supersonic aircraft due to their self starting nature and off design performance. 

## Working Principles
The code in this project begins by solving the Busemann flow field using the Taylor-Maccoll equations, which have been reduced to a first-order state space representation:

$$y=\begin{bmatrix} y_{1} \\ y_{2} \end{bmatrix} = \begin{bmatrix}
V_r \\
V_r'
\end{bmatrix}$$
$$y' = f(\theta, y) = \begin{bmatrix}
y_2 \\
\frac{y_{2}^{2}y_{1}- \frac{{\gamma-1}}{2}(1-y_{1}^{2} - y_{2}^{2})(2y_{1}+y_{2} \cot(\theta))}{\frac{{\gamma-1}}{2}(1-y_{1}^{2}-y_{2}^{2})-y_{2}^{2}}
\end{bmatrix}$$

These equations are contained in "taylor_maccoll.py", where scipy's solve_ivp function is used to perform 4th Order Runge-Kutta integration from the perscribed conic shock angle until the flowfield is parallel with the axial direction. The result of this integration is the Busemann flow field. A single streamtrace is then created following the velocity vector described by the flowfield. 

<p align="center">
  <img src="https://github.com/tycho-0/busemann_inlet_generator/blob/main/images/streamtracing.png" alt="Image showing the Busemann inlet streamtrace"/>
</p>

This process is performed in the busemann_flow.py - InletDesign.wavetrapper_inlet_simple method. The objective of this method is to create the top and bottom surfaces of a wavetrapper inlet based on 5 parameters and calculate properties of the inlet which are of more importance to aircraft design: 

*Inputs*
- mach_2: The mach number in front of the conic shock
- theta_23_degrees: The conic shock angle
- gamma: The ratio of specific heats for the working fluid
- isolator_radius: The radius of the outlet for the wavetrapper, which essentially controls the scale of the inlet
- offset: The percentage distance that the conic shock's apex is moved toward the bottom wall. Limits are [0, 1]
- truncation_angle: the flow angle difference from freestream which the streamtrace integration is halted. Reduces the length of the inlet at the cost of a stronger shock at the lip of the inlet

*Outputs*
- freestream_mach: Freestream mach number
- inlet_diameter: Diameter of the front of the inlet
- isolator_mach: Mach number after the inlet's conic shock
- stagnation_pressure_ratio: Ratio of the stagnation pressure after the conic shock to freestream

This single method forms the basis for the rest of the code.



## Dependencies
- [Python 3.10](https://www.python.org/downloads/release/python-31016/)
- [Numpy](https://pypi.org/project/numpy/)
- [Scipy](https://pypi.org/project/scipy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Open3D](https://pypi.org/project/open3d/)

## How to Use This Project


## References
