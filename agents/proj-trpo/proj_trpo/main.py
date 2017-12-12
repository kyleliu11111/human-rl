from __future__ import print_function, absolute_import, division

import logging
import time
from sys import argv

import gym
import prettytensor as pt
from gym.spaces import Box
from proj_trpo.tb_logger import TBLogger
from proj_trpo.utils import *
# from proj_trpo.vectorized_env import VectorizedEnv

class TRPO(object):
    def __init__(self, env, config):
        self.config = config
        self.env = env
        if not isinstance(env.observation_space, Box) or\
                not isinstance(env.action_space, Box):
            print("Both the input space and the output space should be continuous.")
            print("(Probably OK to remove the requirement for the input space).")
            exit(-1)
        self.train = True
        self.session = tf.Session()
        self.obs = obs = tf.placeholder(
            # dtype, shape=[
            #     None, 2 * env.observation_space.shape[0] + env.action_space.shape[0]])
            dtype, shape=[
                None, env.observation_space.shape[0]])
        act_dim = np.prod(env.action_space.shape)
        self.prev_obs = np.zeros((1, env.observation_space.shape[0]))
        self.prev_action = np.zeros((1, env.action_space.shape[0]))
        self.action = action = tf.placeholder(tf.float32, shape=[None, act_dim])
        self.advant = advant = tf.placeholder(dtype, shape=[None])
        self.old_action_dist = old_action_dist = tf.placeholder(dtype, shape=[None, act_dim])
        self.old_action_dist_logstd = old_action_dist_logstd = tf.placeholder(dtype, shape=[None, act_dim])

        # Create neural network.
        h1, h1_vars = make_fully_connected("policy_h1", self.obs, 64)
        h2, h2_vars = make_fully_connected("policy_h3", h1, act_dim, final_op=None)

        action_dist = h2
        # action_dist, _ = (pt.wrap(self.obs).
        #     fully_connected(64, activation_fn=tf.nn.relu).
            # fully_connected(64, activation_fn=tf.nn.relu).
            # fully_connected(act_dim))  # output means and logstd's

        action_dist_logstd_param = tf.Variable((.01 * np.random.randn(1, act_dim)).astype(np.float32))
        action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(action_dist)[0], 1)))

        self.action_dist = action_dist
        self.action_dist_logstd = action_dist_logstd
        N = tf.shape(obs)[0]

        # compute probabilities of current actions and old action
        # p_n = slice_2d(action_dist, tf.range(0, N), action)
        # oldp_n = slice_2d(old_action_dist, tf.range(0, N), action)
        log_p_n = gauss_log_prob(action_dist, action_dist_logstd, action)
        log_oldp_n = gauss_log_prob(old_action_dist, old_action_dist_logstd, action)

        # proceed as before, good.
        ratio_n = tf.exp(log_p_n - log_oldp_n)
        # ratio_n = p_n / oldp_n #tf.exp(log_p_n - log_oldp_n)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        var_list = tf.trainable_variables()[6:]

        eps = 1e-8
        # Introduced the change into here:
        # kl = tf.reduce_sum(old_action_dist * tf.log((old_action_dist + eps) / (action_dist + eps))) / Nf
        # ent = -tf.reduce_sum(action_dist * tf.log(action_dist + eps)) / Nf
        kl = gauss_KL(old_action_dist, old_action_dist_logstd,
            action_dist, action_dist_logstd) / Nf
        ent = gauss_ent(action_dist, action_dist_logstd) / Nf

        self.losses = [surr, kl, ent]
        # print(surr)
        # print(var_list)
        self.pg = flatgrad(surr, var_list)

        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        # kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
        #     action_dist) * tf.log(tf.stop_gradient(action_dist + eps) / (action_dist + eps))) / Nf
        kl_firstfixed = gauss_selfKL_firstfixed(action_dist, action_dist_logstd) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.get_flat = GetFlat(self.session, var_list)
        self.set_from_flat = SetFromFlat(self.session, var_list)
        self.session.run(tf.variables_initializer(var_list))
        self.vf = LinearVF()

    def act(self, obs):
        # obs = np.expand_dims(obs, 0)
        # action_dist, = self.session.run([self.action_dist], {self.obs: obs})

        # act = action_dist

        # # return act.ravel(),\
        # # if self.train:
        #     # action = int(cat_sample(action_dist_n)[0])
        # # else:
        # # print("BLAH", act.ravel())
        # return act.ravel(),\
        #     ConfigObject(action_dist=action_dist)
        obs = np.expand_dims(obs, 0)
        self.prev_obs = obs
        obs_new = np.concatenate([obs, self.prev_obs, self.prev_action], 1)

        # action_dist_n = self.session.run(self.action_dist, {self.obs: obs})
        action_dist_n, action_dist_logstd =\
            self.session.run([self.action_dist, self.action_dist_logstd], {self.obs: obs})
        # if self.train:
        #     action = int(cat_sample(action_dist_n)[0])
        # else:
        #     action = int(np.argmax(action_dist_n))
        action = action_dist_n + np.exp(action_dist_logstd) * np.random.randn(*action_dist_logstd.shape)
        # self.prev_action *= 0.0
        # self.prev_action[0, action] = 1.0
        self.prev_action = action
        return action.ravel(), action_dist_n, obs.ravel(), ConfigObject(action_dist=action_dist_n, action_dist_logstd=action_dist_logstd)
        # return action, action_dist_n, np.squeeze(obs_new)#ConfigObject(action_dist=action_dist_n)#np.squeeze(obs_new)

    def learn(self):
        config = self.config
        start_time = time.time()
        timesteps_elapsed = 0
        episodes_elapsed = 0
        tb_logger = TBLogger(config.env_id, self.config.name)

        for i in range(1, config.n_iter):
            # Generating paths.
            paths = vectorized_rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch,
                config.predictor,
                render=False)  # (i % render_freq) == 0)

            # Computing returns and estimating advantage function.
            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config.gamma)
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_dist = np.concatenate([path["action_dists"] for path in paths])
            action_dist_logstd = np.concatenate([path["action_dists_logstd"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            # print("BLEGH", action_n.shape)
            # action_n = np.argmax(action_n, axis=1)
            print(action_n.shape)
            print(action_dist.shape)

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()
            advant_n /= (advant_n.std() + 1e-8)

            # Computing baseline function for next iter.
            self.vf.fit(paths)

            # import ipdb; ipdb.set_trace()
            print("BLEH 1", action_dist.shape)
            print("BLEH 2", action_dist_logstd.shape)
            print(action_dist.shape)
            feed = {self.obs: obs_n,
                self.action: action_n,
                self.advant: advant_n,
                self.old_action_dist: action_dist,
                self.old_action_dist_logstd: action_dist_logstd}

            theta_prev = self.get_flat()

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.session.run(self.fvp, feed) + p * config.cg_damping

            g = self.session.run(self.pg, feed_dict=feed)
            stepdir = conjugate_gradient(fisher_vector_product, -g)
            shs = (.5 * stepdir.dot(fisher_vector_product(stepdir)))
            print(shs)
            assert shs > 0

            lm = np.sqrt(shs / config.max_kl)

            fullstep = stepdir / lm

            # neggdotstepdir = -g.dot(stepdir)
            # theta = theta_prev + fullstep
            # def loss(th):
            #     self.set_from_flat(th)
            #     return self.session.run(self.losses[0], feed_dict=feed)
            # theta = linesearch(loss, theta_prev, fullstep, neggdotstepdir / lm)
            self.set_from_flat(theta_prev + fullstep)

            surrogate_loss, kl_old_new, entropy = self.session.run(self.losses, feed_dict=feed)
            if kl_old_new > 2.0 * config.max_kl:
                self.set_from_flat(theta_prev)
            ep_rewards = np.array([path["rewards"].sum() for path in paths])

            stats = {}
            timesteps_elapsed += sum([len(path["rewards"]) for path in paths])
            episodes_elapsed += len(paths)
            stats["timesteps_elapsed"] = timesteps_elapsed
            stats["episodes_elapsed"] = episodes_elapsed
            stats["Average sum of true rewards per episode"] = np.array([path["original_rewards"].sum() for path in paths]).mean()#ep_rewards.mean()
            stats["reward_mean_per_episode"] = ep_rewards.mean()
            stats["entropy"] = entropy
            stats["kl_difference_between_old_and_new"] = kl_old_new
            stats["surrogate_loss"] = surrogate_loss

            for k, v in stats.items():
                tb_logger.log(k, v)
            tb_logger.summary_step += 1

            stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
            print("\n********** Iteration {} ************".format(i))
            for k, v in stats.items():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if entropy != entropy:
                exit(-1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default='Hopper-v1')
    parser.add_argument("--name", type=str, default='unnamed_experiment')
    parser.add_argument("--timesteps_per_batch", type=int, default=8000)
    parser.add_argument("--max_pathlength", type=int, default=2000)
    parser.add_argument("--n_iter", type=int, default=3000)
    parser.add_argument('-s', "--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--max_kl", type=float, default=.001)
    parser.add_argument("--cg_damping", type=float, default=1e-3)
    args = parser.parse_args()
    print('python main.py {}'.format(' '.join(argv)))

    config = ConfigObject(
        timesteps_per_batch=args.timesteps_per_batch,
        max_pathlength=args.max_pathlength,
        gamma=args.gamma,
        n_iter=args.n_iter,
        max_kl=args.max_kl,
        env_id=args.env_id,
        name=args.name,
        cg_damping=args.cg_damping)

    logging.getLogger().setLevel(logging.DEBUG)

    # env = gym.make(args.env_id)
    env_id = args.env_id

    def seeded_env_fn(seed):
        def env_fn():
            from rl_teacher.envs import make_with_torque_removed
            env = make_with_torque_removed(env_id)
            env.seed(seed)
            return env
        return env_fn

    env_fns = [seeded_env_fn(seed) for seed in range(4)]
    # env = VectorizedEnv(env_fns)

    env = gym.make(args.env_id)

    agent = TRPO(env, config)
    agent.learn()

    print('python main.py {}'.format(' '.join(argv)))
