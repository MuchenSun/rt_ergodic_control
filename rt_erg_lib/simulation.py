import matplotlib.pyplot as plt
from matplotlib import animation
import autograd.numpy as np
from .utils import convert_ck2dist, convert_traj2ck
from tqdm import tqdm
import time

class simulation():
    def __init__(self, size, t_dist, model, erg_ctrl, env, tf):
        self.size = size
        self.erg_ctrl = erg_ctrl
        self.env = env
        self.tf = tf
        self.t_dist = t_dist
        self.model = model
        self.exec_times = np.zeros(tf)

    def start(self, report=False):
        self.log = {'trajectory': []}
        init_state = np.array([3., 3., 0.2, 0.2])
        state = self.env.reset(init_state)
        # state = self.env.reset(None)
        for t in tqdm(range(self.tf)):
            start_time = time.time()
            ctrl = self.erg_ctrl(state)
            state = self.env.step(ctrl)
            self.log['trajectory'].append(state)
            self.exec_times[t] = time.time()-start_time
        print("simulation finished.")
        if report:
            print(self.exec_times[1:10])
            print("mean execution time: {0:.6f}(s), standard deviation: {1:.6f}(s)".format(np.mean(self.exec_times), np.std(self.exec_times)))

    def plot(self, point_size=1):
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)
        xt = np.stack(self.log['trajectory'])
        plt.scatter(xt[:self.tf, 0], xt[:self.tf, 1], s=point_size, c='red')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.show()
        return plt.gcf()

    def animate(self, point_size=1, show_traj=True, rate=50):
        [xy, vals] = self.t_dist.get_grid_spec()
        xt = np.stack(self.log['trajectory'])
        plt.contourf(*xy, vals, levels=20)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        fig = plt.gcf()
        points = ax.scatter([], [], s=point_size, c='red')

        def sub_animate(i):
            if(show_traj):
                points.set_offsets(np.array([xt[:i, 0], xt[:i, 1]]).T)
            else:
                points.set_offsets(np.array([[xt[i, 0]], [xt[i, 1]]]).T)
            return [points]

        anim = animation.FuncAnimation(fig, sub_animate, frames=self.tf, interval=(1000/rate), blit=True)
        plt.show()
        return anim

    def path_reconstruct(self):
        xy, vals = self.t_dist.get_grid_spec()
        path = np.stack(self.log['trajectory'])[:self.tf, self.model.explr_idx]
        ck = convert_traj2ck(self.erg_ctrl.basis, path)
        val = convert_ck2dist(self.erg_ctrl.basis, ck, size=self.size)
        plt.contourf(*xy, val.reshape(50,50), levels=20)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.show()