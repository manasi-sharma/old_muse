import argparse
import re
from typing import Dict

from muse.experiments import current_macros, logger


class MacroEnabledArgumentParser(argparse.ArgumentParser):
    """
    Allows macros to be defined and filled in by user.
    Enables loading a single line of args in a file, split by space
    Also allows the user to specify how help
    """
    def __init__(self,
                 macros: Dict[str, str] = current_macros.macros,
                 macro_prefix: str = '^',
                 allow_abbrev=False,
                 **kwargs):
        self._macros = macros
        self._macro_prefix = macro_prefix
        assert len(self._macro_prefix) > 0 or self._macros is None
        super(MacroEnabledArgumentParser, self).__init__(allow_abbrev=allow_abbrev, **kwargs)

        if self.fromfile_prefix_chars is not None:
            assert self._macro_prefix not in self.fromfile_prefix_chars, [self._macro_prefix,
                                                                          self.fromfile_prefix_chars]

    def _parse_known_args(self, arg_strings, namespace):
        if self._macros is not None:
            arg_strings = self._substitute_macros(arg_strings)
        return super(MacroEnabledArgumentParser, self)._parse_known_args(arg_strings, namespace)

    def convert_arg_line_to_args(self, arg_line):
        """
        OVERRIDE to enable splitting line of args in file by space
        The parent class just returns [arg_line], which doesn't work when multiple args are on one line
        """
        return arg_line.split()

    def _substitute_macros(self, arg_strings):
        # specify macros as ^{MACRO}
        incorrect_structure = r'%s{(?!.*})' % self._macro_prefix
        options = '|'.join(self._macros.keys())
        pattern = f'{self._macro_prefix}{{{options}}}'
        # expand arguments referencing files
        new_arg_strings = []
        for arg_string in arg_strings:
            # missing end bracket
            if re.search(incorrect_structure, arg_string) is not None:
                self.error(f"Unclosed brackets in argument: {arg_string}")

            # regex macro substitution, while matches exist
            j = 0
            while re.search(pattern, arg_string) is not None:
                for macro, value in self._macros.items():
                    # e.g., !{ROOT}
                    arg_string = arg_string.replace(f'{self._macro_prefix}{{{macro}}}', str(value))
                j += 1
                if j == 10:
                    # print error
                    logger.warn(f'Recursing >10 times for macro {arg_string}, searching for {pattern}')
            # for regular arguments, just add them back into the list
            new_arg_strings.append(arg_string)

        # return the modified argument list
        return new_arg_strings


class _CallFnAction(argparse.Action):

    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 call_fn=None,
                 help=None):
        self.call_fn = call_fn
        super(_CallFnAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        self.call_fn(parser, namespace, values, option_string=option_string)


def get_call_fn_action_class(call_fn):
    class CustomCallFnAction(_CallFnAction):
        def __init__(self, **kwargs):
            kwargs.update({'call_fn': call_fn})
            super().__init__(**kwargs)
    return CustomCallFnAction
