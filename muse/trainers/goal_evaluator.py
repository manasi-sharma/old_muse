import numpy as np

from muse.experiments import logger
from muse.trainers.base_goal_trainer import BaseGoalTrainer
from muse.utils.general_utils import timeit, is_next_cycle

from attrdict import AttrDict


class GoalEvaluator(BaseGoalTrainer):
    """
    No training, just does offline evaluations
    """

    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        assert self._no_data_saving, "Saving data not implemented"
        assert not self._load_statistics_initial, "Cannot load statistics on evaluation!"

        # specify either max steps or max eps
        self._max_steps = get_with_default(params, "max_steps", None, map_fn=int)
        self._max_eps = get_with_default(params, "max_eps", None, map_fn=int)
        assert (self._max_eps is None) ^ (self._max_steps is None), "Must set EXACTLY one of max_steps and max_eps!"

        if self._max_eps is None:
            assert self._max_steps > 0
        else:
            assert self._max_eps > 0

        """ env STEP (one step) """
        # how often to do N steps (in terms of current_eval_steps)
        self._step_train_env_every_n_steps = get_with_default(params, "step_train_env_every_n_steps", 1)
        # how many steps to take when the above is true
        self._step_train_env_n_per_step = get_with_default(params, "step_train_env_n_per_step", 1)
        # how often to do N holdout env steps (in terms of current_eval_steps)
        self._step_holdout_env_every_n_steps = get_with_default(params, "step_holdout_env_every_n_steps", 0)
        # how many holdout env steps to take when the above is true
        self._step_holdout_env_n_per_step = get_with_default(params, "step_holdout_env_n_per_step", 1)
        self._log_every_n_steps = int(params["log_every_n_steps"])

        """ env ROLLOUT (step until done) """
        # how often to do rollouts (in terms of current_eval_steps)
        self._rollout_train_env_every_n_steps = get_with_default(params, "rollout_train_env_every_n_steps", 1)
        # how many rollouts to do when the above is true
        self._rollout_train_env_n_per_step = get_with_default(params, "rollout_train_env_n_per_step", 1)
        # how often to do holdout env rollouts (in terms of current_eval_steps)
        self._rollout_holdout_env_every_n_steps = get_with_default(params, "rollout_holdout_env_every_n_steps", 0)
        # how many holdout env rollouts to do when the above is true
        self._rollout_holdout_env_n_per_step = get_with_default(params, "rollout_holdout_env_n_per_step", 1)

        assert self._rollout_holdout_env_every_n_steps == 0 or self._step_train_env_every_n_steps == 0, \
            "Cannot both rollout AND step train env."
        assert self._rollout_holdout_env_every_n_steps == 0 or self._step_holdout_env_every_n_steps == 0, \
            "Cannot both rollout AND step holdout env."

        # TODO data saving

        # add to the training dataset every N completed goals, or if episode terminates.
        #   This helps if resets are infrequent, or if we do not care about the episode boundary.
        self._add_to_data_train_every_n_goals = 0  # params["add_to_data_train_every_n_goals"]
        self._add_to_data_holdout_every_n_goals = 0  # params["add_to_data_holdout_every_n_goals"]

        # # data train save freq, per dataset.
        # self._save_data_train_every_n_steps = listify(params["save_data_train_every_n_steps"],
        #                                               len(self._datasets_train))
        # # data holdout save freq, per dataset.
        # self._save_data_holdout_every_n_steps = listify(params["save_data_holdout_every_n_steps"],
        #                                                 len(self._datasets_holdout))

        # how often in steps to log things to tensorboard (it overwrites itself if a positive number), -1 = only at end
        self._write_to_tensorboard_every_n = int(
            get_with_default(params, "write_to_tensorboard_every_n_train_steps", -1))

        self._current_eval_step = 0


        if self._max_eps is None:
            logger.info(f"Evaluating for {self._max_steps} steps")
            logger.info(f"Env Train: Step Eval w/ every_n={self._step_train_env_every_n_steps}, n_per={self._step_train_env_n_per_step}")
            logger.info(f"Env Holdout: Step Eval w/ every_n={self._step_holdout_env_every_n_steps}, n_per={self._step_holdout_env_n_per_step}")

        else:
            logger.info(f"Evaluating for {self._max_eps} eps")
            logger.info(f"Env Train: Rollout Eval w/ every_n={self._rollout_train_env_every_n_steps}, n_per={self._rollout_train_env_n_per_step}")
            logger.info(f"Env Holdout: Rollout Eval w/ every_n={self._rollout_holdout_env_every_n_steps}, n_per={self._rollout_holdout_env_n_per_step}")

        logger.info(f"Writing every {self._write_to_tensorboard_every_n} eval steps")

    def _init_optimizers(self, params):
        self._optimizer = None

    def _log(self):  # , AVG RETURN: {}±{}' TODO
        logger.info(
            f'[{self._current_eval_step}] env (steps={self._current_env_train_step}, eps={self._current_env_train_ep}) '
            f'| env holdout (steps={self._current_env_holdout_step}, eps={self._current_env_holdout_ep})')

        if any(tracker.has_data() for tracker in self._trackers.leaf_values()):
            tracker_str = "Trackers:"
            for tracker_name, tracker in self._trackers.leaf_items():
                if tracker.has_data():
                    #  the tracker has time series output (e.g. buffered returns), which we will average
                    ts_outputs = tracker.get_time_series().leaf_apply(lambda arr: np.asarray(arr)[None])  # T
                    writing_types = self._tracker_write_types[tracker_name]
                    for key, arr in ts_outputs.leaf_items():
                        if arr.size > 0:
                            tracker_str += f"\n------- {tracker_name}, len={arr.size} --------"
                            if 'mean' in writing_types:
                                tracker_str += f'\n{key + "_mean"}:  {arr.mean()}'
                            if 'max' in writing_types:
                                tracker_str += f'\n{key + "_max"}:  {arr.max()}'
                            if 'min' in writing_types:
                                tracker_str += f'\n{key + "_min"}:  {arr.min()}'
                            if 'std' in writing_types:
                                tracker_str += f'\n{key + "_std"}:  {arr.std()}'
            logger.debug(tracker_str)

        logger.debug(str(timeit))
        timeit.reset()

    # RUNNING SCRIPTS #

    def eval_step(self, obs_train, goal_train, obs_holdout, goal_holdout):

        return obs_train, goal_train, obs_holdout, goal_holdout

    def run(self):
        """
        This is the main loop:
            - eval either rollouts or steps.
        """
        self.run_preamble(required_checkpoint=True)

        obs_train = AttrDict()
        goal_train = AttrDict()
        obs_holdout = AttrDict()
        goal_holdout = AttrDict()

        def finished():
            if self._max_eps is None:
                return self._current_env_train_step >= self._max_steps
            else:
                return self._current_env_train_ep >= self._max_eps

        logger.debug("-------------- EVAL STARTING --------------")

        # loop
        while not finished():
            # NOTE: always have some form of timing so that you can find bugs
            with timeit('total_loop'):

                if self._max_steps is not None:
                    """ STEP by STEP"""
                    if is_next_cycle(self._current_eval_step, self._step_train_env_every_n_steps):
                        with timeit('step train env'):
                            for i in range(self._step_train_env_n_per_step):
                                obs_train, goal_train, _ = self.env_step(self._model, self._env_train,
                                                                         [],
                                                                         obs_train, goal_train, self._env_train_memory,
                                                                         self._policy, self._goal_policy,
                                                                         reward=self._reward,
                                                                         eval=True,
                                                                         trackers=self._trackers < ["env_train"],
                                                                         curr_step=self._current_env_train_step,
                                                                         add_data_every_n=self._add_to_data_train_every_n_goals)
                                self._current_env_train_step += 1
                                self._current_env_train_ep += int(self._env_train_memory.is_empty())

                    if is_next_cycle(self._current_eval_step, self._step_holdout_env_every_n_steps):
                        with timeit('step holdout env'):
                            for i in range(self._step_holdout_env_n_per_step):
                                obs_holdout, goal_holdout, _ = self.env_step(self._model, self._env_holdout,
                                                                             [],
                                                                             obs_holdout, goal_holdout,
                                                                             self._env_holdout_memory,
                                                                             self._policy_holdout,
                                                                             self._goal_policy_holdout,
                                                                             reward=self._reward, eval=True,
                                                                             trackers=self._trackers < ["env_holdout"],
                                                                             curr_step=self._current_env_holdout_step,
                                                                             add_data_every_n=self._add_to_data_holdout_every_n_goals)
                                self._current_env_holdout_step += 1
                                self._current_env_holdout_ep += int(self._env_holdout_memory.is_empty())
                else:
                    """ EPISODIC ROLLOUT """
                    if is_next_cycle(self._current_eval_step, self._rollout_train_env_every_n_steps):
                        with timeit('rollout train env'):
                            for i in range(self._rollout_train_env_n_per_step):
                                step_wrapper = AttrDict(step=self._current_env_train_step)
                                obs_holdout, goal_holdout, _ = self.env_rollout(self._model, self._env_train,
                                                                                [],
                                                                                obs_train, goal_train,
                                                                                self._env_train_memory,
                                                                                self._policy, self._goal_policy,
                                                                                reward=self._reward,
                                                                                trackers=self._trackers < ["env_train"],
                                                                                curr_step_wrapper=step_wrapper,
                                                                                add_to_data_every_n=self._add_to_data_train_every_n_goals)
                                self._current_env_train_step = step_wrapper["step"]
                                self._current_env_train_ep += 1

                            if is_next_cycle(self._current_eval_step, self._write_to_tensorboard_every_n):
                                # also force a write step here
                                self._tracker_write_step(self._trackers < ["env_train"], self._current_env_train_step,
                                                         force=True, debug=True)

                    if is_next_cycle(self._current_eval_step, self._rollout_holdout_env_every_n_steps):
                        with timeit('rollout holdout env'):
                            for i in range(self._rollout_holdout_env_n_per_step):
                                step_wrapper = AttrDict(step=self._current_env_holdout_step)
                                obs_holdout, goal_holdout, _ = self.env_rollout(self._model, self._env_holdout,
                                                                                [],
                                                                                obs_holdout, goal_holdout,
                                                                                self._env_holdout_memory,
                                                                                self._policy_holdout,
                                                                                self._goal_policy_holdout,
                                                                                reward=self._reward,
                                                                                trackers=self._trackers < [
                                                                                    "env_holdout"],
                                                                                curr_step_wrapper=step_wrapper,
                                                                                add_to_data_every_n=self._add_to_data_holdout_every_n_goals)
                                self._current_env_holdout_step = step_wrapper["step"]
                                self._current_env_holdout_ep += 1

                            if is_next_cycle(self._current_eval_step, self._write_to_tensorboard_every_n):
                                # also force a holdout write step here
                                self._tracker_write_step(self._trackers < ["env_holdout"], self._current_env_holdout_step,
                                                         force=True, debug=True)

                # TODO saving data
                # skip the first cycle.
                # if self._current_eval_step > 0:
                #     # SAVE DATA
                #     for i in range(len(self._datasets_train)):
                #         if is_next_cycle(self._current_eval_step, self._save_data_train_every_n_steps[i]):
                #             with timeit('save_data_train'):
                #                 self._datasets_train[i].save()
                #
                #     for i in range(len(self._datasets_holdout)):
                #         if is_next_cycle(self._current_eval_step, self._save_data_holdout_every_n_steps[i]):
                #             with timeit('save_data_holdout'):
                #                 self._datasets_holdout[i].save()

            # update step
            self._current_eval_step += 1

            if is_next_cycle(self._current_eval_step, self._log_every_n_steps):
                self._log()

        if self._write_to_tensorboard_every_n != 0:
            # also force a write step at the end
            self._tracker_write_step(self._trackers < ["env_train"], self._current_env_train_step,
                                     force=True, debug=True)

        logger.debug("-------------- EVAL FINISHED --------------")
        self._log()
