import sys
sys.path.append("/home/msun/Code/rt_ergodic_control")
from rt_erg_lib.double_integrator import DoubleIntegrator
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation import simulation
import autograd.numpy as np
import matplotlib.pyplot as plt

"""initialization"""
size = 25.0
env = DoubleIntegrator(size=size)
model = DoubleIntegrator(size=size)
means = [np.array([10.5, 12.8]), np.array([10.5,12.8]), np.array([10.5, 12.8]), np.array([15.5, 5.5]), np.array([5.5, 20.5]), np.array([5.5, 20.5])]
vars = [np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
erg_ctrl = RTErgodicControl(model, t_dist, horizon=50, num_basis=15, batch_size=-1)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 500
erg_ctrl_sim = simulation(size, t_dist, model, erg_ctrl, env, tf)
erg_ctrl_sim.start(report=True)
erg_ctrl_sim.animate(point_size=5, show_traj=False)
erg_ctrl_sim.plot(point_size=1)
erg_ctrl_sim.path_reconstruct()