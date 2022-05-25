import moderngl
from pyrr import Matrix44, Vector3
import numpy as np

from base_window import BaseWindowConfig

class RobotWindow(BaseWindowConfig):

    def __init__(self, **kwargs):
        super(RobotWindow, self).__init__(**kwargs)

    def model_load(self):
        self.box = self.load_scene('cube.obj')
        self.box = self.box.root_nodes[0].mesh.vao.instance(self.program)

        self.sphere = self.load_scene('sphere.obj')
        self.sphere = self.sphere.root_nodes[0].mesh.vao.instance(self.program)

    def init_shaders_variables(self):
        self.color = self.program['color']
        self.pvm_matrix = self.program['pvm_matrix']

    def render(self, time: float, frame_time: float):
        self.ctx.clear(0.8, 0.8, 0.8, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        #Perspective set up

        projection = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            (-20.0, -15.0, 5.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        )

########################################################################

        # Head

        # Color
        self.color.value = (0.0, 1.0, 0.0)

        # Translation by vector(0, 0, 5)
        move = Matrix44.from_translation((0.0, 0.0, 5.0))

        self.pvm_matrix.write((projection * lookat * move).astype('f4'))
        self.sphere.render()

########################################################################

        # Body

        # Color
        self.color.value = (0.0, 1.0, 1.0)

        # Axis scale by (1, 1, 2)
        scale = Matrix44.from_scale((1.0, 1.0, 2.0))

        # Translation by vector(0, 0, 2)

        move = Matrix44.from_translation((0.0, 0.0, 2.0))

        self.pvm_matrix.write((projection * lookat * move * scale).astype('f4'))
        self.box.render()

########################################################################

        # Arms
        
        # Color
        self.color.value = (0.0, 0.0, 1.0)

        # Axis scale by (0.5, 0.5, 1.25)
        scale = Matrix44.from_scale(Vector3([0.5, 0.5, 1.25]))

        # 45 degrees rotation on the x axis
        rotate = Matrix44.from_x_rotation(-np.pi/4)

        # Translation by vector(0, 3, 3)
        move = Matrix44.from_translation(Vector3([0.0, 3.0, 3.0]))

        self.pvm_matrix.write((projection * lookat * move * rotate * scale).astype('f4'))
        self.box.render()

        # 45 degrees rotation on the x axis
        rotate = Matrix44.from_x_rotation(np.pi/4)

        # Translation by vector(0, -3, 3)
        move = Matrix44.from_translation(Vector3([0.0, -3.0, 3.0]))

        self.pvm_matrix.write((projection * lookat * move * rotate * scale).astype('f4'))
        self.box.render()

########################################################################

        # Legs

        # Color
        self.color.value = (0.0, 0.5, 0.5)

        # Axis scale by (0.5, 0.5, 1.75)
        scale = Matrix44.from_scale(Vector3([0.5, 0.5, 1.75]))

        # 30 degrees rotation on the x axis
        rotate = Matrix44.from_x_rotation(-np.pi/6)

        # Translation by vector(0, 2, -1.5)
        move = Matrix44.from_translation(Vector3([0.0, 2.0, -1.5]))

        self.pvm_matrix.write((projection * lookat * move * rotate * scale).astype('f4'))
        self.box.render()

        # 30 degrees rotation on the x axis
        rotate = Matrix44.from_x_rotation(np.pi/6)

        # Translation by vector(0, -2, -1.5)
        move = Matrix44.from_translation(Vector3([0.0, -2.0, -1.5]))

        self.pvm_matrix.write((projection * lookat * move * rotate * scale).astype('f4'))
        self.box.render()
