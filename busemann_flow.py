import math
import numpy as np  # type: ignore
from scipy.integrate import solve_ivp # type: ignore
from taylor_maccoll import taylor_maccoll
import matplotlib.pyplot as plt

class Inlet:
    def __init__(self, mach_2:float, theta_23_degrees:float, gamma:float, isolator_radius:float, offset:float, truncation_angle:float):
        self.mach_2 = mach_2
        self.theta_23_degrees = theta_23_degrees
        self.gamma = gamma
        self.isolator_radius = isolator_radius
        self.offset = offset
        self.truncation_angle = truncation_angle

        # Resample point count for wavetrapper mesh
        self.streamtrace_point_count = 50

        # Design Outputs
        self.wavetrapper_x = []
        self.wavetrapper_y = []
        self.wavetrapper_z = []
        self.rotation = []
        self.scale = []

        self.freestream_mach = 0 
        self.inlet_diameter = 0
        self.isolator_mach = 0
        self.stagnation_pressure_ratio = 0

    # Methods
    
    def test_design(self):
        self.wavetrapper_inlet_simple()
        print('Design Inputs:')
        print('----------------------------------')
        print(f'\tmach_2 = {self.mach_2}')
        print(f'\ttheta_23_degrees = {self.theta_23_degrees}')
        print(f'\tgamma = {self.gamma}')
        print(f'\tisolator_radius = {self.isolator_radius}')
        print(f'\toffset = {self.offset}')
        print(f'\ttruncation_angle = {self.truncation_angle}')

        print('\nDesign Outputs:')
        print('----------------------------------')
        print(f'\tfreestream_mach = {self.freestream_mach}') 
        print(f'\tinlet_diameter = {self.inlet_diameter}')
        print(f'\tisolator_mach = {self.isolator_mach}')
        print(f'\tstagnation_pressure_ratio = {self.stagnation_pressure_ratio}')
        

    def generate(self):
        self.wavetrapper_inlet_simple()
        self.wavetrapper_inlet()


    def wavetrapper_inlet(self, show_plot=False):
        """
        Calculates the flow field and streamlines of a wavetrapper inlet and writes a CSM file representing the inlet.

        Parameters
        ----------
        mach_2 : float
            The Mach number in front of the conic shock.
        theta_23_degrees : float
            The conic shock angle.
        gamma : float
            The ratio of specific heats.
        isolator_radius : float
            The radius of the isolator in meters.
        offset : float
            The percent distance from the center of the isolator to the bottom inlet wall in meters.
        truncation_angle : float
            Angle between freestream and the start of the inlet in degrees.

        Returns
        -------
        None

        """

        mach_2 = self.mach_2
        theta_23_degrees = self.theta_23_degrees
        gamma = self.gamma
        isolator_radius = self.isolator_radius
        offset = self.offset
        truncation_angle = self.truncation_angle

        theta_23 = np.radians(theta_23_degrees)
        tan_delta = 2 * (1/np.tan(theta_23)) * (((mach_2**2 * np.sin(theta_23)**2) - 1) / (mach_2**2 * (gamma + np.cos(2 * theta_23)) + 2))
        delta_23 = np.arctan(tan_delta)
        theta_2 = theta_23 - delta_23
        u2 = mach_2 * np.cos(theta_23)
        v2 = -mach_2 * np.sin(theta_23)

        stagnation_pressure_ratio = (((gamma+1)*mach_2**2 * np.sin(theta_23)**2) / ((gamma-1)*mach_2**2 * np.sin(theta_23)**2 + 2))**(gamma/(gamma-1)) * ((gamma+1) / (2*gamma*mach_2**2 * np.sin(theta_23)**2 - gamma + 1))**(1 / (gamma-1))
        mach_n3 = np.sqrt((1 + ((gamma-1)/2) * mach_2**2 * np.sin(theta_23)**2) / (gamma*mach_2**2 * np.sin(theta_23)**2 - ((gamma-1)/2)))
        mach_3 = mach_n3 / np.sin(theta_23 - delta_23)
        print(f'Isolator Mach Number: {mach_3:.4f}')
        print(f'Stagnation Pressure Ratio: {stagnation_pressure_ratio:.4f}')

        # Solve Taylor-Maccoll Equations
        opts = {'rtol': 1e-12, 'atol': 1e-12}
        sol = solve_ivp(lambda t, y: taylor_maccoll(t, y, gamma), [theta_2, np.pi-0.01], [u2, v2], method='RK45', events=check_busemann_limit, **opts)

        theta = sol.t
        vr = sol.y[0]
        vtheta = sol.y[1]
        m = np.sqrt(vr**2 + vtheta**2)
        print(f'Freestream Mach: {m[-1]:.3f}')

        # Streamtrace
        r2 = isolator_radius / np.sin(theta_2)
        r = np.zeros(len(theta))
        r[0] = r2
        surf_angle = np.zeros(len(theta)-1)
        for i in range(1, len(theta)):
            d_theta = theta[i] - theta[i-1]
            dr_dtheta = r[i-1] * vr[i] / vtheta[i]
            r[i] = r[i-1] + dr_dtheta * d_theta
            dx = r[i] * np.cos(theta[i]) - r[i-1] * np.cos(theta[i-1])
            dy = r[i] * np.sin(theta[i]) - r[i-1] * np.sin(theta[i-1])
            surf_angle[i-1] = 180 - np.degrees(np.arctan2(dy, dx))
            if surf_angle[i-1] < truncation_angle:
                r = r[:i+1]
                theta = theta[:i+1]
                break

        # Mark the inflection point index for plotting purposes
        inflection_point = np.argmin(np.abs(vr))

        # Create wavetrapper
        phi = np.radians(np.arange(0, 360, 2))  # Convert degrees to radians, match MATLAB's 0:2:360
        wavetrapper_x = np.zeros((len(r), len(phi)))
        wavetrapper_y = np.zeros((len(r), len(phi)))
        wavetrapper_z = np.zeros((len(r), len(phi)))
        wavetrapper_r = np.zeros((len(r), len(phi)))
        wavetrapper_mach = np.ones((len(r), len(phi)))  # Initialize Mach number array
        radial_dist = np.zeros((len(r), len(phi)))
        axial_dist = np.zeros((len(r), len(phi)))

        # Initial conditions for wavetrapper based on isolator radius and offset
        wavetrapper_y[0, :] = isolator_radius * np.cos(phi) + offset
        wavetrapper_z[0, :] = isolator_radius * np.sin(phi)
        scale = []
        rotation = []

        for j in range(len(phi)):
            azimuth = np.arctan2(wavetrapper_z[0, j], wavetrapper_y[0, j])
            wavetrapper_x[0, j] = np.sqrt(wavetrapper_y[0, j]**2 + wavetrapper_z[0, j]**2) / np.tan(theta_2)
            radial_distance = np.sqrt(wavetrapper_y[0, j]**2 + wavetrapper_z[0, j]**2)
            axial_distance = wavetrapper_x[0, j]
            hyp = np.sqrt(radial_distance**2 + axial_distance**2)
            wavetrapper_r[:, j] = r * hyp / r2
            
            radial_dist[:, j] = wavetrapper_r[:, j] * np.sin(theta)
            axial_dist[:, j] = wavetrapper_r[:, j] * np.cos(theta)

            wavetrapper_y[1:, j] = radial_dist[1:, j] * np.cos(azimuth)
            wavetrapper_z[1:, j] = radial_dist[1:, j] * np.sin(azimuth)
            wavetrapper_x[1:, j] = axial_dist[1:, j]
            wavetrapper_mach[:, j] = m[:len(r)]

            scale.append(wavetrapper_r[0,j] / (wavetrapper_r[0,0]))
            rotation.append(azimuth)
        
        n_samples = self.streamtrace_point_count
        theta_resample_indices = np.round(np.linspace(0, len(theta) - 1, n_samples)).astype(int)
        self.wavetrapper_x_resample = np.zeros((n_samples, len(phi)))
        self.wavetrapper_y_resample = np.zeros((n_samples, len(phi)))
        self.wavetrapper_z_resample = np.zeros((n_samples, len(phi)))



        for j in range(len(phi)):
            self.wavetrapper_x_resample[:, j] = wavetrapper_x[theta_resample_indices, j]
            self.wavetrapper_y_resample[:, j] = wavetrapper_y[theta_resample_indices, j]
            self.wavetrapper_z_resample[:, j] = wavetrapper_z[theta_resample_indices, j]

        if show_plot:
            # Show a 2D plot illustrating the features of this wavetrapper
            plt.figure(1)
            bottom_index = math.ceil(len(wavetrapper_x[0]) / 2)
            plt.plot(wavetrapper_x[:,0], wavetrapper_y[:,0], 'k', 
                    [0, wavetrapper_x[0,0]], [0, wavetrapper_y[0,0]], 'r', 
                    [0, wavetrapper_x[inflection_point,0]], [0, wavetrapper_y[inflection_point,0]], 'b', 
                    [0, wavetrapper_x[-1,0]], [0, wavetrapper_y[-1,0]], 'g')
            plt.plot(wavetrapper_x[:,bottom_index], wavetrapper_y[:,bottom_index], 'k', 
                    [0, wavetrapper_x[0,bottom_index]], [0, wavetrapper_y[0,bottom_index]], 'r', 
                    [0, wavetrapper_x[inflection_point,bottom_index]], [0, wavetrapper_y[inflection_point,bottom_index]], 'b', 
                    [0, wavetrapper_x[-1,bottom_index]], [0, wavetrapper_y[-1,bottom_index]], 'g')
            plt.axis('equal')
            plt.grid(True)
            plt.legend(['Busemann Inlet Surface', 'Conic Shock', 'Inflection Point', 'Termination Point'])
            plt.show()

            # Plot all streamtraces showing a 3D representation of the wavetrapper
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.clear()

            for j in range(len(phi)):
                ax.plot(wavetrapper_x[:, j], wavetrapper_y[:, j], wavetrapper_z[:, j], color='k', alpha=0.2)  # Adjust alpha for line transparency
                ax.plot(wavetrapper_x[:, j], wavetrapper_y[:, j], -wavetrapper_z[:, j], color='k', alpha=0.2)  # Adjust alpha for line transparency

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.grid(True)
            ax.set_aspect('equal')
            plt.show()

        self.wavetrapper_x = wavetrapper_x
        self.wavetrapper_y = wavetrapper_y
        self.wavetrapper_z = wavetrapper_z
        self.rotation = rotation
        self.scale = scale

        # Write CSM File
        self.write_csm()


    def wavetrapper_inlet_simple(self):
        """
        Calculates the flow field and streamlines of a wavetrapper inlet.

        Parameters
        ----------
        mach_2 : float
            The Mach number in front of the conic shock.
        theta_23_degrees : float
            The conic shock angle.
        gamma : float
            The ratio of specific heats.
        isolator_radius : float
            The radius of the isolator in meters.
        offset : float
            The percent distance from the center of the isolator to the bottom inlet wall in meters.
        truncation_angle : float
            Angle between freestream and the start of the inlet in degrees.

        Returns
        -------
        list
            [freestream_mach, inlet_diameter, isolator_mach, stagnation_pressure_ratio]
        """
        # Import parameters from the inlet class
        mach_2 = self.mach_2
        theta_23_degrees = self.theta_23_degrees
        gamma = self.gamma
        isolator_radius = self.isolator_radius
        offset = self.offset
        truncation_angle = self.truncation_angle
        
        # Calculate the required conic shock deflection angle, and corresponding velocity components 
        theta_23 = np.radians(theta_23_degrees)
        tan_delta = 2 * (1/np.tan(theta_23)) * (((mach_2**2 * np.sin(theta_23)**2) - 1) / (mach_2**2 * (gamma + np.cos(2 * theta_23)) + 2))
        delta_23 = np.arctan(tan_delta)
        theta_2 = theta_23 - delta_23
        u2 = mach_2 * np.cos(theta_23)
        v2 = -mach_2 * np.sin(theta_23)

        # Calculate stagnation pressure ratio, and outflow Mach number
        stagnation_pressure_ratio = (((gamma+1)*mach_2**2 * np.sin(theta_23)**2) / ((gamma-1)*mach_2**2 * np.sin(theta_23)**2 + 2))**(gamma/(gamma-1)) * ((gamma+1) / (2*gamma*mach_2**2 * np.sin(theta_23)**2 - gamma + 1))**(1 / (gamma-1))
        mach_n3 = np.sqrt((1 + ((gamma-1)/2) * mach_2**2 * np.sin(theta_23)**2) / (gamma*mach_2**2 * np.sin(theta_23)**2 - ((gamma-1)/2)))
        mach_3 = mach_n3 / np.sin(theta_23 - delta_23)

        # Solve Taylor-Maccoll Equations
        opts = {'rtol': 1e-12, 'atol': 1e-12}
        sol = solve_ivp(lambda t, y: taylor_maccoll(t, y, gamma), [theta_2, np.pi-0.01], [u2, v2], method='RK45', events=check_busemann_limit, **opts)

        # Extract radial and tangential velocity components along each ray marching ahead of the conic shock apex
        theta = sol.t
        vr = sol.y[0]
        vtheta = sol.y[1]
        cone_angle = np.degrees(theta[-1])
        m = np.sqrt(vr**2 + vtheta**2)

        # Create the streamtrace for the inlet
        r2 = isolator_radius / np.sin(theta_2)
        r = np.zeros(len(theta))
        r[0] = r2
        surf_angle = np.zeros(len(theta)-1)
        for i in range(1, len(theta)):
            d_theta = theta[i] - theta[i-1]
            dr_dtheta = r[i-1] * vr[i] / vtheta[i]
            r[i] = r[i-1] + dr_dtheta * d_theta
            dx = r[i] * np.cos(theta[i]) - r[i-1] * np.cos(theta[i-1])
            dy = r[i] * np.sin(theta[i]) - r[i-1] * np.sin(theta[i-1])
            surf_angle[i-1] = 180 - np.degrees(np.arctan2(dy, dx))
            if surf_angle[i-1] < truncation_angle:
                r = r[:i+1]
                theta = theta[:i+1]
                break

        # Set up variables for storing the wavetrapper upper and lower streamline coordinates
        phi = np.array([0,np.radians(180)]) # Convert degrees to radians, match MATLAB's 0:2:360
        wavetrapper_x = np.zeros((len(r), len(phi)))
        wavetrapper_y = np.zeros((len(r), len(phi)))
        wavetrapper_z = np.zeros((len(r), len(phi)))
        wavetrapper_r = np.zeros((len(r), len(phi)))
        wavetrapper_mach = np.ones((len(r), len(phi)))  # Initialize Mach number array
        radial_dist = np.zeros((len(r), len(phi)))
        axial_dist = np.zeros((len(r), len(phi)))

        # Initial conditions for wavetrapper based on isolator radius and offset
        wavetrapper_y[0, :] = isolator_radius * np.cos(phi) + offset
        wavetrapper_z[0, :] = isolator_radius * np.sin(phi)

        # Calculate the polar coordinates of the streamtraces, then convert to cartesian coordinates
        for j in range(len(phi)):
            azimuth = np.arctan2(wavetrapper_z[0, j], wavetrapper_y[0, j])
            wavetrapper_x[0, j] = np.sqrt(wavetrapper_y[0, j]**2 + wavetrapper_z[0, j]**2) / np.tan(theta_2)
            radial_distance = np.sqrt(wavetrapper_y[0, j]**2 + wavetrapper_z[0, j]**2)
            axial_distance = wavetrapper_x[0, j]
            hyp = np.sqrt(radial_distance**2 + axial_distance**2)
            wavetrapper_r[:, j] = r * hyp / r2
            
            radial_dist[:, j] = wavetrapper_r[:, j] * np.sin(theta)
            axial_dist[:, j] = wavetrapper_r[:, j] * np.cos(theta)

            wavetrapper_y[1:, j] = radial_dist[1:, j] * np.cos(azimuth)
            wavetrapper_z[1:, j] = radial_dist[1:, j] * np.sin(azimuth)
            wavetrapper_x[1:, j] = axial_dist[1:, j]
            wavetrapper_mach[:, j] = m[:len(r)]

        # Store design parameters
        freestream_mach = m[-1]
        inlet_diameter = wavetrapper_y[-1][0] + wavetrapper_y[-1][1]
        isolator_mach = mach_3
    
        # Update the Inlet class with the design parameters
        self.freestream_mach = freestream_mach 
        self.inlet_diameter = inlet_diameter
        self.isolator_mach = isolator_mach
        self.stagnation_pressure_ratio = stagnation_pressure_ratio


    def write_csm(self):    
        wavetrapper_x = self.wavetrapper_x
        wavetrapper_y = self.wavetrapper_y
        wavetrapper_z = self.wavetrapper_z
        rotation = self.rotation
        scale = self.scale

        inlet_center_y = (wavetrapper_y[-1] - (wavetrapper_y[-1] * scale[-1]))/2
        inlet_diameter = (wavetrapper_y[-1] + wavetrapper_y[-1] * scale[-1])

        isolator_center_y = (wavetrapper_y[0] - (wavetrapper_y[0] * scale[-1]))/2
        isolator_diameter = (wavetrapper_y[0] + wavetrapper_y[0] * scale[-1])

        inlet_length = wavetrapper_x[-1] - wavetrapper_x[0]

        with open('busemann.csm','w') as f:

            f.writelines('# busemann.csm written by busemann_flow.py\n\n')
            f.writelines('# Branches:\n\n')

            f.writelines(f'CYLINDER   {wavetrapper_x[0] + 0.5}   {isolator_center_y}   0   {inlet_length*2}   {isolator_center_y}   0   {isolator_diameter/1.999}\n')
            f.writelines(f'ROTATEX   90   {isolator_center_y}   0\n')

            f.writelines('STORE   $body1   0\n')

            f.writelines(f'SKBEG     {wavetrapper_x[0]}   {wavetrapper_y[0]}   {wavetrapper_z[0]}   0\n')

            for i in range(len(wavetrapper_x)):
                f.writelines(f'   SPLINE   {wavetrapper_x[i]}   {wavetrapper_y[i]}   {wavetrapper_z[i]}\n')

            f.writelines('SKEND     0\n\n')
            f.writelines('STORE   $name   $root1\n')

            for i in range(0, len(rotation)):
                f.writelines(f'\nRESTORE $name   $root1\n')
                f.writelines(f'ROTATEX   {(180.0/np.pi) * rotation[i]}   0   0\n')
                f.writelines(f'SCALE   {scale[i]}   0   0   0\n')
                
            f.writelines('\nBLEND   0   0   0   0   0   0\n')
            f.writelines(f'EXTRUDE   {inlet_length*2}   0   0\n')
            f.writelines('STORE   $name   $inlet   1\n')
            f.writelines('MIRROR   0   0   1\n')
            f.writelines('RESTORE   $name   $inlet\n')
            f.writelines('UNION\n')

            f.writelines('RESTORE   $body1\n')
            f.writelines('UNION   0   0   0.01\n')

            f.writelines(f'CYLINDER   {inlet_length}   0   0   {inlet_length*5}   0   0   {inlet_diameter}\n')
            f.writelines('SUBTRACT\n')
            f.writelines('DUMP      busemann.step   0   0   0   .\n')
            f.writelines('\nEND\n')
    

def check_busemann_limit(theta: float, y: np.ndarray) -> float:
    """
    Calculates the crossflow velocity at a given angle theta based on the given velocity components y.

    Parameters
    ----------
    theta : float
        The angle between the x-axis and the velocity vector.
    y : np.ndarray
        The velocity components in the x- and y-directions, respectively.

    Returns
    -------
    float
        The crossflow velocity at the given angle theta.

    """
    crossflow_velocity = y[0] * np.sin(theta) + y[1] * np.cos(theta)
    # The event occurs when crossflow_velocity changes sign (crosses zero)
    # Since solve_ivp stops the integration when the event function returns zero,
    # we return the crossflow_velocity itself as the event condition.
    return crossflow_velocity


## Test code for generating an example inlet
if __name__ == '__main__':
    mach_2 = 3.57
    theta_23_degrees = 21.34
    gamma = 1.4
    isolator_radius = 0.1 / 2
    offset = 0.9 * isolator_radius
    truncation_angle = 8e-15
    inlet = Inlet(mach_2=mach_2, theta_23_degrees=theta_23_degrees, gamma=gamma, isolator_radius=isolator_radius, offset=offset, truncation_angle=truncation_angle)
    inlet.test_design()
    inlet.wavetrapper_inlet(show_plot=True)