import sys
sys.path.append("/home/msun/Code/rt_ergodic_control")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation_slam import simulation_slam
import autograd.numpy as np

"""initialization"""
size = 10.0
# size = 25.0
noise = 0.005
init_state = np.array([2., 2., 0.0])
env = IntegratorSE2(size=size)
model = IntegratorSE2(size=size)
means = [np.array([7.5, 2.5]), np.array([2.5, 7.5])]
vars = [np.array([1.,1.])**2, np.array([0.7,0.7])**2]
# means = [np.array([5.5, 16.8]), np.array([5.5,16.8]), np.array([16.5, 9.8]), np.array([16.5, 9.8]), np.array([5.5, 5.5]), np.array([5.5, 5.5]), np.array([22.5, 21.5]), np.array([22.5, 21.5])]
# vars = [np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2, np.array([1.2,1.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
erg_ctrl = RTErgodicControl(model, t_dist, horizon=30, num_basis=10, batch_size=100)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 2000
# landmarks = np.array([
#     [4.2, 3.3],
#     [5.4, 8.2],
#     [7.4, 3.3],
#     [6.3, 4.9],
#     [3.2, 1.2]
# ])
landmarks1 = np.random.uniform(0.2, 3.8, size=(6,2))
landmarks2 = np.random.uniform(8.2, 9.8, size=(4,2))
landmarks = np.concatenate((landmarks1, landmarks2))
sensor_range = 2
erg_ctrl_sim = simulation_slam(size, init_state, t_dist, model, erg_ctrl, env, tf, landmarks, sensor_range)
erg_ctrl_sim.start(report=True)
erg_ctrl_sim.animate(point_size=1, show_label=True, show_traj=True)
erg_ctrl_sim.plot(point_size=1, save=None)
erg_ctrl_sim.path_reconstruct(save=None)