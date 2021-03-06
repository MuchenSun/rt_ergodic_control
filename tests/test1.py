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
size = 3.0
env = DoubleIntegrator(size=size)
model = DoubleIntegrator(size=size)
means = [np.array([0.5, 1.8]), np.array([2.6,0.6]), np.array([2.5, 2.5])]
vars = [np.array([0.2,0.2])**2, np.array([0.2,0.2])**2, np.array([0.2,0.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
erg_ctrl = RTErgodicControl(model, t_dist, horizon=50, num_basis=15, batch_size=-1)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 3000
erg_ctrl_sim = simulation(size, t_dist, model, erg_ctrl, env, tf)
erg_ctrl_sim.start(report=True)
erg_ctrl_sim.animate(point_size=5, show_traj=False)
erg_ctrl_sim.plot(point_size=1)
erg_ctrl_sim.path_reconstruct()