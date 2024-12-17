from busemann_flow import InletDesign
import open3d as o3d  # type: ignore
import numpy as np  # type: ignore
import math


class InletMesh:
    def __init__(self, inlet: InletDesign):
        self.inlet = inlet
        self.inlet.streamtrace_point_count = 90
        self.inlet.generate()
        vertex_x = np.array(inlet.wavetrapper_x_resample)
        vertex_y = np.array(inlet.wavetrapper_y_resample)
        vertex_z = np.array(inlet.wavetrapper_z_resample)
        
        n = len(vertex_x)
        m = len(vertex_x[0])

        self.num_vertices = n * m
        self.num_faces = m * (n-1)

        self.vertices = np.zeros([n * m, 3])
        for i in range(n):
            for j in range(m):
                self.vertices[i + j * n] = [vertex_x[i][j], vertex_y[i][j], vertex_z[i][j]]

        self.faces = np.zeros([self.num_faces, 4])

        for J in range(m):
            for I in range(n-1):
                self.faces[I + J*n] = [I + J*n, I+1 + J*n, I + ((J*n) + 1)%(n*m), I+1 + ((J*n) + 1)%(n*m)]     
        
        pass
                


    def visualize_points(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vertices)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        # vis.get_render_option().background_color = [0,0,1]
        vis.add_geometry(pcd)
        vis.run()
        pass
        


if __name__ == '__main__':    
    inlet = InletDesign(3.5755891619371436, 21.392019580965947, 1.4, 0.3028737410348979, 0.3, 1.106489353652762e-05)
    
    mesh = InletMesh(inlet)
    mesh.visualize_points()

    pass