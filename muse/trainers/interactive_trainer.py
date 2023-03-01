from typing import Tuple

import numpy as np
import torch

from muse.datasets.preprocess.data_augmentation import DataAugmentation
from muse.experiments import logger
from muse.trainers.base_goal_trainer import BaseGoalTrainer
from muse.trainers.optimizers.optimizer import SingleOptimizer

from muse.utils.python_utils import timeit, is_next_cycle

from attrdict import AttrDict
from attrdict.utils import get_with_default, get_cls_param_instance

class InteractiveGoalTrainer(BaseGoalTrainer):

    def __init__(self, params, file_manager,
                 model,
                 policy,
                 goal_policy,
                 offline_dataset,
                 online_dataset,
                 env_train,
                 reward=None,
                 optimizer=None):

        self._offline_dataset = offline_dataset
        self._online_dataset = online_dataset

        # online is first, since that's the one we are usually saving data to.
        self._datasets = [self._online_dataset, self._offline_dataset]

        # samplers
        self._offline_dataset_sampler = self._offline_dataset.sampler
        self._online_dataset_sampler = self._online_dataset.sampler

        super(InteractiveGoalTrainer, self).__init__(params, file_manager, model, policy, goal_policy, env_train,
                                                     None, policy_holdout=None, goal_policy_holdout=None,  # no holdout
                                                     reward=reward, optimizer=optimizer)

        self._env_train_memory.intervention_history = []

    def _init_params_to_attrs(self, params):
        super(InteractiveGoalTrainer, self)._init_params_to_attrs(params)
        # ENVIRONMENT steps.
        self._max_steps = int(params >> "max_steps")

        # data augmentation, used for both online and offline datasets.
        self.data_augmentation_params: AttrDict = get_with_default(params, "data_augmentation_params", AttrDict())
        if isinstance(self.data_augmentation_params, DataAugmentation):
            self.data_augmentation: DataAugmentation = self.data_augmentation_params  # allow passing in data aug
        elif not self.data_augmentation_params.is_empty():
            self.data_augmentation: DataAugmentation = get_cls_param_instance(self.data_augmentation_params,
                                                                              "cls", "params", DataAugmentation)
        else:
            logger.info("Using no data augmentation.")
            self.data_augmentation = None
        self._train_do_data_augmentation = get_with_default(params, "train_do_data_augmentation", True)

        self._train_every_n_steps = params >> "train_every_n_steps"
        self._train_n_per_step = get_with_default(params, "train_n_per_step", 1)
        self._block_train_on_first_n_steps = int(get_with_default(params, "block_train_on_first_n_steps", 0))
        self._write_to_tensorboard_every_n = int(
            get_with_default(params, "write_to_tensorboard_every_n_train_steps", 20))

        # training state
        self._current_train_step = 0
        self._current_train_loss = np.inf
        self._log_every_n_steps = int(params >> "log_every_n_steps")
        self._save_every_n_steps = int(params >> "save_every_n_steps")

        self._save_checkpoint_every_n_steps = get_with_default(params, "save_checkpoint_every_n_steps", 0)
        if self._save_checkpoint_every_n_steps > 0:
            assert self._save_checkpoint_every_n_steps % self._save_every_n_steps == 0, "Checkpointing steps should be a multiple of model save steps."

        # manually disabled.
        self._add_to_data_train_every_n_goals = 0
        assert self._no_data_saving, "Data storing outside of online_dataset not supported yet!"

    def _init_optimizers(self, params):
        if len(list(self._model.parameters())) > 0:
            if self._optimizer is None:
                self._optimizer = SingleOptimizer(params >> "optimizer", self._model, datasets=self._datasets)
            elif isinstance(self._optimizer, AttrDict):
                self._optimizer = self._optimizer.cls(self._optimizer.params, self._model,
                                                      datasets=self._datasets)
        else:
            logger.warn("Model has no parameters...")

    def _optimizer_step(self, loss, inputs, outputs, meta: AttrDict = AttrDict(), i=0, ti=0, writer=None,
                        writer_prefix=""):
        # loss might be a dictionary potentially. TODO
        return self._optimizer.step(loss, inputs, outputs, dataset_idx=0, meta=meta, i=i, ti=ti, writer=writer,
                                    writer_prefix=writer_prefix)

    def _train_step(self, model):
        """
        Trains the model with the given offline and online data.

        :param offline_data:
        :param online_data:
        :return:
        """

        # (B x H x ...)
        with torch.no_grad():
            with timeit('train/get_offline_batch'):
                off_indices = self._offline_dataset_sampler.get_indices()
                res = self._offline_dataset.get_batch(indices=off_indices,
                                                      torch_device=model.device)
                offline_inputs, offline_outputs = res[:2]
                offline_meta = res[2] if len(res) == 3 else AttrDict()

            if self._train_do_data_augmentation and self.data_augmentation is not None:
                with timeit('train/data_offline_augmentation'):
                    offline_inputs, offline_outputs = self.data_augmentation.forward(offline_inputs, offline_outputs)

            with timeit('train/get_online_batch'):
                on_indices = self._online_dataset_sampler.get_indices()
                res = self._online_dataset.get_batch(indices=on_indices,
                                                      torch_device=model.device)
                online_inputs, online_outputs = res[:2]
                online_meta = res[2] if len(res) == 3 else AttrDict()

            if self._train_do_data_augmentation and self.data_augmentation is not None:
                with timeit('train/data_online_augmentation'):
                    online_inputs, online_outputs = self.data_augmentation.forward(online_inputs, online_outputs)

        model.train()

        sw = None
        if is_next_cycle(self._current_train_step, self._write_to_tensorboard_every_n):
            sw = self._summary_writer

        self._train_dual_batches(model, offline_inputs, offline_outputs, offline_meta,
                                 online_inputs, online_outputs, online_meta, writer=sw)

    def _restore_meta_data(self, checkpoint):
        # loading interactive checkpoint vs. regular checkpoint.
        if 'interactive_step' in checkpoint.keys():
            # override step
            self._current_step = checkpoint['interactive_step']  # override's default "step" field.
            self._current_train_step = checkpoint['interactive_train_step']
            self._current_train_loss = checkpoint['interactive_train_loss']
        else:
            self._current_step = self._current_train_step = 0
            self._current_train_loss = checkpoint['train_loss']

    def _get_save_meta_data(self):
        return {
            'interactive_step': self._current_step,
            'interactive_train_step': self._current_train_step,
            'interactive_train_loss': self._current_train_loss,
        }

    def _train_dual_batches(self, model, offline_inputs, offline_outputs, offline_meta,
                            online_inputs, online_outputs, online_meta, writer=None):
        """
        Trains on two batches, from offline and online data respectively.

        :param model:
        :param offline_inputs:
        :param offline_outputs:
        :param offline_meta:
        :param online_inputs:
        :param online_outputs:
        :param online_meta:
        :param writer:
        :return:
        """
        raise NotImplementedError

    def intervene_condition(self, model, env, env_memory):
        """
        Determines if intervention is necessary at the current step. Might depend on the model.

        :param model:
        :param env:
        :param env_memory:
        :return: bool
        """
        raise NotImplementedError

    def get_intervention(self, env, env_memory) -> Tuple[AttrDict, bool, bool]:
        """
        Get the actual intervention from the user, TODO: how?
        :param env:
        :param env_memory:
        :return: the intervention (AttrDict), intervention_done (intervention is over), and terminate_early (episode ends)
        """
        raise NotImplementedError

    def update_intervention_history(self, intervention_history, intervention: AttrDict):
        intervention_history.append(intervention)

    def aggregate(self, env, env_memory, intervention_history) -> Tuple[AttrDict, AttrDict]:
        """
        Combines all the interventions and updates the env memory with them,
        :param env:
        :param env_memory:
        :param intervention_history:
        :return: intervention episode (ins, outs)
        """
        raise NotImplementedError

    def run(self):
        """
        Steps the environment, etc
        :return:
        """
        self.run_preamble()

        # current state of the env.
        obs = AttrDict()
        goal = AttrDict()

        while self._current_step < self._max_steps:
            # step the environment. will update history as well.
            obs, goal, done = self.env_step(self._model, self._env_train, self._datasets, obs, goal,
                                            self._env_train_memory, self._policy, self._goal_policy,
                                            eval=True, reward=self._reward, curr_step=self._current_step,
                                            trackers=self._trackers < ["env_train"],
                                            add_data_every_n=self._add_to_data_train_every_n_goals)
            self._current_env_train_ep += int(done)

            # training
            if is_next_cycle(self._current_step, self._train_every_n_steps) and self._current_step >= self._block_train_on_first_n_steps:
                with timeit('train'):
                    for i in range(self._train_n_per_step):
                        self._train_step(self._model)
                        self._current_train_step += 1

            # saving things, skip the first cycle.
            if self._current_step > 0 and is_next_cycle(self._current_step, self._save_every_n_steps):
                with timeit('save'):
                    do_best = False
                    # compare if tracker name is valid and new episodes have been rolled out.
                    if self._track_best_name in self._trackers.leaf_keys() and \
                            self._current_env_train_ep > self._last_best_env_train_ep:
                        tracker = self._trackers[self._track_best_name]
                        # if there's data, save if this is the best one yet.
                        if tracker.has_data():
                            curr_tracked_val = self._track_best_reduce_fn(
                                np.asarray(tracker.get_time_series() >> self._track_best_key))
                            do_best = curr_tracked_val is not None and curr_tracked_val > self._last_best_tracked_val

                    if do_best:
                        self._last_best_tracked_val = curr_tracked_val
                        self._last_best_env_train_ep = self._current_env_train_ep
                        logger.info(
                            f"Saving best model: tracker = {self._track_best_name}, {self._track_best_key} = {curr_tracked_val}")

                        # also writing this only on saves.
                        if self._summary_writer is not None:
                            self._summary_writer.add_scalar(f"{self._track_best_name}/{self._track_best_key}_BEST", self._last_best_tracked_val, self._current_step)

                    self._save(chkpt=is_next_cycle(self._current_step, self._save_checkpoint_every_n_steps),
                               best=do_best)

            self._current_step += 1

    def clear_env_memory(self, obs, goal, env_memory, no_reset=False):
        super(InteractiveGoalTrainer, self).clear_env_memory(obs, goal, env_memory, no_reset=no_reset)

        # record with env_memory
        if env_memory.has_leaf_key('intervention_history'):
            env_memory.intervention_history.clear()

    def _env_post_step(self, model, env, env_memory, datasets, obs, goal, done):
        if self.intervene_condition(model, env, env_memory):
            intervention, intervention_done, done_early = self.get_intervention(env, env_memory)
            self.update_intervention_history(env_memory.intervention_history, intervention)
            done[:] |= done_early

            if done.item() or intervention_done:
                iep_ins, iep_outs = self.aggregate(env, env_memory, env_memory.intervention_history)
                iep_outs.done[-1] = True
                self._online_dataset.add_episode(iep_ins, iep_outs)

        return obs, goal, done
