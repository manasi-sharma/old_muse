from typing import List

from attrdict.utils import get_with_default

from muse.experiments import logger


class Writer(object):
    """
    Writers are abstractions for logging experiment information

    TODO
    """

    def __init__(self, exp_name: object, params: object, file_manager: object,
                 resume: object = False) -> object:
        self.exp_name = exp_name
        self.params = params.leaf_copy()
        self._file_manager = file_manager
        self.resume = resume

        self._init_params_to_attrs(params)

    def _init_params_to_attrs(self, params):
        self.project_name = get_with_default(params, "project_name", 'muse')
        self.config = dict(get_with_default(params, "config", {}))

    def open(self):
        """
        Opens a channel for whatever type of writer this is

        Returns
        -------

        """
        raise NotImplementedError

    def update_config(self, cfg_dict):
        """
        Adds a config to the logger after the open() call.
        Parameters
        ----------
        cfg_dict: dict

        Returns
        -------

        """
        raise NotImplementedError

    def add_scalar(self, name, value, step, **kwargs):
        """
        Logs a scalar.

        Parameters
        ----------
        name
        value
        step
        kwargs

        Returns
        -------
        """
        raise NotImplementedError


class TensorboardWriter(Writer):

    def open(self):
        from torch.utils.tensorboard import SummaryWriter
        self._summary_writer = SummaryWriter(self._file_manager.exp_dir)

    def update_config(self, cfg_dict):
        # tensorboard does not have a config option I think
        pass

    def add_scalar(self, name, value, step, **kwargs):
        self._summary_writer.add_scalar(name, value, step, **kwargs)


class WandbWriter(Writer):

    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)
        self.tags = params << "tags"
        if self.tags:
            assert isinstance(self.tags, List), f"Tags {self.tags} must be a list!"
            logger.debug(f'[wandb] Using tags: {self.tags}')

    def open(self):
        import wandb
        self.run = wandb.init(project=self.project_name, name=self.exp_name,
                              config=self.config, resume=self.resume, tags=self.tags)

    def update_config(self, cfg_dict):
        self.run.config.update(cfg_dict)

    def add_scalar(self, name, value, step, **kwargs):
        self.run.log({name: value}, step=step)
