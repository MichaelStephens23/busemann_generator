# Busemann & Wavetrapper Inlet Generator
**WARNING: This project is a work in progress and may produce incorrect results.**

## Description
This project provides an open source implementation of recent research in the field of hypersonic Busemann inlet design -- specifically the work done by Sannu Mölder in "The Busemann Air Intake for Hypersonic Speeds" (DOI: 10.5772/intechopen.82736). The overarching goal is to produce Busemann inlet modules which Mölder calls "wavetrapper" inlets which are more suited for integration into supersonic aircraft due to their self starting nature and off design performance. 



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

![Image showing the Busemann inlet streamtrace](https://cdnintech.com/media/chapter/66839/1512345123/media/F3.png)


## Dependencies
- [Python 3.10](https://www.python.org/downloads/release/python-31016/)
- [Numpy](https://pypi.org/project/numpy/)
- [Scipy](https://pypi.org/project/scipy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Open3D](https://pypi.org/project/open3d/)

## How to Use This Project


## References
