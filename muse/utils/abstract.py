import abc as _abc
import json
from typing import List, Dict, Tuple

from muse.experiments import logger
from attrdict.utils import get_with_default


class Argument:
    """ For easily converting between parser kwargs and param, so is intentionally similar to argparse.Action

    each argument has a name (same as dest), as well as the option to add other names

    This only works for non-positional args
    """

    def __init__(self, name, *option_names, nargs=None, const=None, default=None, action=None,
                 type=None, choices=None, required=False, help=None, metavar=None):
        assert '-' not in name, f"{name} must not contain '-'"
        # this is where
        self.name = name
        # first option (will get '--{prefix}' before)
        self.dest = name
        # other option names
        self.option_names = option_names
        self.nargs = nargs
        self.const = const
        self.action = action
        self.type = type
        self.choices = choices
        self.required = required
        self.help = help
        if help is None:
            if self.type is not None:
                self.help = "type = %(type)s"
            elif self.action is not None:
                self.help = f"action = {self.action}"

        self.metavar = metavar

        # special case for filling in boolean value (in case user did not specify default bool val)
        if default is None and self.action in ['store_true', 'store_false']:
            default = self.action == 'store_false'
        self.default = default

    def set_default(self, value):
        """ Sets the default value for the argument.

        Parameters
        ----------
        value

        Returns
        -------

        """
        if self.action in ['store_true', 'store_false']:
            assert isinstance(value, bool)
            if self.default != value:
                self.default = value
                # toggle the action
                self.action = 'store_false' if self.action == 'store_true' else 'store_true'
                # make sure the name reflects the new action
                if self.dest.startswith('no-'):
                    self.dest = self.dest[3:]
                else:
                    self.dest = f'no-{self.name}'
        elif self.action is None:
            assert isinstance(value, self.type), f"Arg={self.name}: value={value} not of type={self.type}"
        else:
            raise NotImplementedError(f'action default unimplemented: {self.action}')

        self.default = value

    def add_to_parser(self, parser, added_option_names=(), update_kwargs=None, prefix="") -> str:
        """ For configs to use, to automatically read arg """
        kwargs = {
            "action": self.action, "default": self.default, "required": self.required, "help": self.help
        }
        if self.nargs is not None:
            kwargs["nargs"] = self.nargs
        if self.const is not None:
            kwargs["const"] = self.const
        if self.type is not None:
            kwargs["type"] = self.type
        if self.choices is not None:
            kwargs["choices"] = self.choices
        if self.metavar is not None:
            kwargs["metavar"] = self.metavar

        if isinstance(self.default, Argument):
            kwargs['default'] = None  # default will be filled in later

        if update_kwargs is not None:
            kwargs.update(update_kwargs)

        parser.add_argument(f"--{prefix}{self.dest}", *self.option_names, *added_option_names, **kwargs)
        return self.name

    def setattr_from_params(self, obj, params, values, prefix=""):
        """ For class to use when instantiating params.
        The params might not have been created using the add_to_parser above, hence double checking required.

        if self.default is another argument, will try to read it from values (order of declaration matters)
        """
        my_name = f"{prefix}{self.name}"
        if self.required:
            val = params[my_name]
        else:
            default = self.default
            # argument must be at the same "level" as this argument, otherwise prefix won't make sense.
            if isinstance(self.default, Argument):
                def_name = f"{prefix}{self.default.name}"
                assert def_name in values.keys(), f"external value for {def_name} " \
                                                           f"must be provided for {my_name} " \
                                                           f"(make sure external argument is defined first so it is read first)"
                default = values[def_name]

            val = get_with_default(params, my_name, default)

        setattr(obj, self.name, val)

        return val

    def __repr__(self):
        return f"Argument[name={self.name}, type={self.type}, action={self.action}]"


def resolve_arguments(*classes):
    # will remove duplicate arguments (by instance, not by name)
    # will error if names conflict but instances aren't the same
    arguments = []
    corr_names = []
    corr_classes = []
    for cls in classes:
        local_args_to_keep = []
        for arg in cls.predefined_arguments:
            # add if name does not conflict
            if arg not in arguments:
                for a, c in zip(arguments, corr_classes):
                    if arg.name == a.name:
                        raise ValueError(f'{cls}.{arg.name} = existing {c}.{a.name}, for resolving classes: {classes}')

                local_args_to_keep.append(arg)
                corr_names.append(arg.name)
                corr_classes.append(cls)

        arguments.extend(local_args_to_keep)

    return arguments


# class ArgumentSpec:
#
#     def __init__(self, predefined_arguments: List[Argument]=(), subclasses: List[Tuple[str, str, type]]=(), subclass_delim='_'):
#         # parameters that follow the argparse format, for easily defining configs for a given class.
#         self.predefined_arguments = list(predefined_arguments)
#
#         # strong typing on sub fields that also derive from base class
#         # args will be prefixed by the [name][subclass_delim]
#         # (attr_name, prefix_name, type)
#         self.subclasses = list(subclasses)
#
#         # delimiter between sub class name and its args.
#         self.subclass_delim = subclass_delim


class BaseClass(_abc.ABC):  # , _overrides.EnforceOverrides):

    # parameters that follow the argparse format, for easily defining configs for a given class.
    predefined_arguments: List[Argument] = [

    ]

    # strong typing on sub fields that also derive from base class
    # args will be prefixed by the [name][subclass_delim]
    # (attr_name, prefix_name, type)
    subclasses: List[Tuple[str, str, type]] = [

    ]

    # delimiter between sub class name and its args.
    subclass_delim = '_'

    @classmethod
    def search_arg(cls, name, default=None, required=False):
        for arg in cls.predefined_arguments:
            if name == arg.name:
                return arg

        if required:
            raise ValueError(f'missing argument in {cls}: {name}')
        else:
            assert default is not None
            return default

    @classmethod
    def predefined_arg_names(cls):
        # recursively get the names of arguments
        names = [a.name for a in cls.predefined_arguments]
        for _, n, sub_cls in cls.subclasses:
            assert issubclass(sub_cls,
                              BaseClass), f"{sub_cls} does not derive from base class but was listed in sub-classes!"
            sub_names = sub_cls.predefined_arg_names
            names.extend(f"{n}{cls.subclass_delim}{sn}" for sn in sub_names)
        return names

    @classmethod
    def declare_arguments(cls, parser, extra_option_names=None, update_kwargs=None,
                          skip_args=(), prefix="", defaults=None) -> List[str]:
        """ Fills in a parser

        Parameters
        ----------
        parser
        extra_option_names: dict[name -> List[str]]: local names with some additional naming options
        update_kwargs: dict[name -> dict[str->any]]: local names with some new kwargs to parser
        skip_args: list[name]: local names to avoid.
        prefix: the prefix to add to the local names before adding to the global arg parser.
        defaults: optional dict, new arg defaults.

        Returns
        -------
        list of names that were declared.

        """
        skip_args = list(skip_args)

        if update_kwargs is None:
            update_kwargs = {}
        if extra_option_names is None:
            extra_option_names = {}

        names = []
        skipped = []
        # local
        for arg in cls.predefined_arguments:
            if arg.name not in skip_args:
                extra_opnames = extra_option_names.get(arg.name, {})
                kwargs = update_kwargs.get(arg.name, {})
                if defaults is not None and arg.name in defaults.keys():
                    arg.set_default(defaults[arg.name])

                names.append(
                    arg.add_to_parser(parser, added_option_names=extra_opnames, update_kwargs=kwargs, prefix=prefix))
            else:
                skipped.append(arg.name)

        # log before doing nested args
        logger.debug(f"[{cls}] will declare arguments: {names} with prefix \"{prefix}\"")

        # we will iteratively remove from this as we declare
        node_skip_args = list(skip_args)

        # add in nested args
        for _, n, sub_cls in cls.subclasses:
            assert issubclass(sub_cls,
                              BaseClass), f"{sub_cls} does not derive from base class but was listed in sub-classes!"
            # prefix for indexing the various dicts and lists
            local_prefix = f"{n}{cls.subclass_delim}"
            # de-prefix the extra / update names
            local_predef_names = sub_cls.predefined_arg_names()
            is_subclass_arg = lambda k: (k.starts_with(local_prefix) and k[len(local_prefix):] in local_predef_names)
            local_extra_opnames = {k[len(local_prefix):]: v for k, v in extra_option_names.items()
                                   if is_subclass_arg(k)}
            local_update_kwargs = {k[len(local_prefix):]: v for k, v in update_kwargs.items()
                                   if is_subclass_arg(k)}
            local_skip_args = [k[len(local_prefix):] for k in skip_args
                         if is_subclass_arg(k)]

            # we have to declare in the global prefix, not just local.
            added_local_names = sub_cls.declare_arguments(parser, extra_option_names=local_extra_opnames,
                                                          update_kwargs=local_update_kwargs, skip_args=local_skip_args,
                                                          prefix=f"{prefix}{local_prefix}")
            names.extend(f"{local_prefix}{n}" for n in added_local_names)

            # remove the local skip args so we know these were handled
            for n in local_skip_args:
                node_skip_args.remove(n)

        assert len(skipped) == len(node_skip_args), f"[{cls}] Node skip {node_skip_args}, but only could remove: {skipped}"

        return names

    def read_predefined_params(self, params, prefix=""):
        """
        Now we actually read the arguments from params, prefixed by something

        Parameters
        ----------
        params: AttrDict

        Returns
        -------
        values for the params.
        """
        values = {}

        # set all of the attributes on this object first.
        for arg in self.predefined_arguments:
            prefixed_name = f"{prefix}{arg.name}"
            values[prefixed_name] = arg.setattr_from_params(self, params, values, prefix=prefix)

        # now traverse locally
        for attr, name, _ in self.subclasses:
            try:
                local_obj = getattr(self, attr)
            except Exception:
                raise Exception(f"[{type(self)}] missing attr: {attr}!")
            local_pref = f"{prefix}{self.subclass_delim}{name}"
            inner_values = local_obj.read_predefined_params(params, prefix=local_pref)
            # these will have the global keys
            values.update(inner_values)

        logger.debug(f"[{type(self)}] Read class fields: {json.dumps(values, indent=4, sort_keys=True)}")
        return values
