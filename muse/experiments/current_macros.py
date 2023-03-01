"""
This is a file that you might update depending on the given experiment.

It allows you to define macros that can shorten command strings.

These macros will be prefixed by '^{MACRO_STR}' on command line.

macros below is the default argument in muse.experiments.grouped_parser.MacroEnabledArgumentParser
"""

cfg_exp = 'configs/exp_hvs'

macros = {
    'CFG': cfg_exp,
    'STAT': f'{cfg_exp}/static',
}