"""
2024.03.08  Trial: Dynamics of arrow.
In this version, only gravity and air resistance (with a simple form) are considered. 
The trace of the arrow is solved using velocity Verlet algorithm.
"""
import numpy as np 
import dataclasses
import typing as tp 

# The position, velocity is expressed using 2-d array
# vec_t = tp.TypeAlias[np.ndarray]

# physical constants
g_const = 9.8   # m/s^2
rho_air = 1.29  # kg/m^3


@dataclasses.dataclass
class ArrowModel:
    mass: float     # g
    radius: float   # mm
    resistance_coeff: float  # dimensionless


class ArrowDynamics:
    """
    Simulator of arrow dynamics.
    """
    def __init__(self, model:ArrowModel) -> None:
        self.model = model
        self.pos = np.zeros(2, float)   # m
        self.velocity = np.zeros(2, float)   # m/s
        self.force = np.zeros(2, float)   # N

    def cal_force(self):
        """
        Compute current force.
        """
        self.force[0] = 0.
        self.force[1] = -self.model.mass * 1e-3 * g_const   # gravity
        v2 = np.dot(self.velocity, self.velocity)
        if v2 > 0:
            # air resistance
            v_dir = self.velocity / np.sqrt(v2)
            area = np.pi * self.model.radius**2 * 1e-6   # m^2
            self.force += -v_dir * area * self.model.resistance_coeff * v2 * rho_air / 2


    def verlet_step(self, dt:float)->None:
        """
        Verlet MD step.
        See https://zhuanlan.zhihu.com/p/75210090
        Note, here duplicate force computation is called. However, we do not consider it in this simple program
        """
        self.cal_force()
        acc = self.force / self.model.mass * 1e3   # m/s^2
        # first half step
        self.velocity += acc * dt / 2.0
        self.pos += self.velocity * dt 
        # second half step
        self.cal_force()
        acc = self.force / self.model.mass * 1e3  # m/s^2
        self.velocity += acc * dt / 2.0

    def get_trace_for_time(self, time:float, dt:float)->tp.List[np.ndarray]:
        """
        2024.03.11  Returns the trace of the arrow 
        in the time range given by `time` with time step `dt`.
        The position and velocity should be assigned BEFORE calling this. This is same for other similar functions.
        """
        res = []
        t = 0.0
        while t < time:
            self.verlet_step(dt)
            res.append(self.pos.copy())
            t += dt
        return res
    
    def get_trace_for_condition(self, dt:float, cont_func:tp.Callable[['ArrowDynamics'], bool], max_time=None)->tp.List[np.ndarray]:
        """
        2024.03.11  Get the trace UNTIL the `cont_func` returns false. 
        the optional `max_time` is the limitation of time.
        """
        res = []
        t = 0.0
        while cont_func(self):
            if max_time is not None and t >= max_time:
                break
            self.verlet_step(dt)
            res.append(self.pos.copy())
            t += dt
        return res
    
    def get_trace_for_distance(self, distance:float, dt:float, max_time=None)->tp.List[np.ndarray]:
        """
        Returns the trace until the specified distance (the x coord) is reached.
        """
        return self.get_trace_for_condition(dt, lambda s: s.pos[0] < distance, max_time)
    
    def get_trace_for_height(self, dt:float, max_time=None)->tp.List[np.ndarray]:
        return self.get_trace_for_condition(dt, lambda s: s.pos[1] >= 0., max_time)
    
    def set_height(self, h: float):
        self.pos = np.array([0.0, h])
    
    def set_horizontal_speed(self, vx: float):
        self.velocity = np.array([vx, 0.0])
