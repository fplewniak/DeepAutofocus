import argparse
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt


def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                print(f'{parser.prog}: error: argument {option_string}: requires value between {nmin} and {nmax}')
                exit(0)
            setattr(args, self.dest, values)

    return RequiredLength


def equal_nargs(arg):
    class EqualNargs(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            dest = arg.lstrip(parser.prefix_chars)
            if len(values) == len(vars(args)[dest]) or (len(values) == 0 and self.required is False):
                setattr(args, self.dest, values)
            else:
                print(f'{parser.prog}: error: arguments {",".join(self.option_strings)} and {arg} require the same number of values '
                      f'(found {len(values)} and {len(vars(args)[dest])})')
                exit(0)
    return EqualNargs

def get_params(argv):
    parser = argparse.ArgumentParser(description='Compare models.')

    parser.add_argument('--data', metavar='STR', help='List of csv files containing prediction vs ground truth',
                        type=str, nargs='+', action=equal_nargs('--stats'))
    parser.add_argument('--stats', metavar='STR', help='List of csv files containing evaluation stats', type=str,
                        required=True, nargs='+', action=required_length(1, 7))
    parser.add_argument('--title', metavar='STR', help='Plot title', type=str, default=None)
    parser.add_argument('--legend', metavar='STR', help='Legend specification', type=str, default=None, nargs='+',
                        action=equal_nargs('--stats'))
    parser.add_argument('--savefig', metavar='FILE', help='Save plot to file', default=None)


    a = parser.parse_args()

    return a.stats, a.title, a.legend, a.data, a.savefig


if __name__ == '__main__':
    stat_files, title, legend, data_files, savefig = get_params(sys.argv[1:])

    fig, axis = plt.subplots(nrows=1, ncols=1)

    if title is not None:
        fig.suptitle(title)

    # axis.set(xlim=(-8, 8), ylim=(-8, 8))

    colours = ('b', 'r', 'g', 'c', 'm', 'y', 'k')

    for i, stat_file in enumerate(stat_files):
        stats_df = pd.read_csv(stat_file, sep=',')
        axis.plot('gt', 'error', f'{colours[i]}-', data=stats_df)

    for i, stat_file in enumerate(stat_files):
        stats_df = pd.read_csv(stat_file, sep=',')
        axis.plot('gt', 'error_quant5', f'{colours[i]}:', data=stats_df, alpha=0.5)
        axis.plot('gt', 'error_quant95', f'{colours[i]}:', data=stats_df, alpha=0.5)
        # axis.fill_between('gt', 'error_quant5', 'error_quant95', alpha=0.2, data=stats_df, color=f'{colours[i]}')
        if data_files is not None:
            pred_df = pd.read_csv(data_files[i], sep=',')
            axis.plot('gt', 'error', f'{colours[i]}D', markersize=2.5, data=pred_df, alpha=0.1)

    axis.grid(visible=True, axis='both', which='major')
    axis.set_aspect('equal', adjustable='datalim', anchor='S')
    if legend is not None:
        axis.legend(legend)
    else:
        axis.legend(stat_files)

    if savefig is not None:
        plt.savefig(savefig)
    else:
        plt.show()
    print("Done.")
