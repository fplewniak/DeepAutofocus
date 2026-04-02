import argparse
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
                print(
                        f'{parser.prog}: error: arguments {",".join(self.option_strings)} and {arg} require the same number of values '
                        f'(found {len(values)} and {len(vars(args)[dest])})')
                exit(0)

    return EqualNargs


def check_equal_nargs(arg1, arg2, parser, a):
    dest1 = arg1.lstrip(parser.prefix_chars)
    dest2 = arg2.lstrip(parser.prefix_chars)
    a_var = vars(a)
    if (a_var[dest1] is not None) and (a_var[dest2] is not None) and not (len(a_var[dest1]) == len(a_var[dest2])):
        print(f'{parser.prog}: error: arguments {arg1} and {arg2} require the same number of values '
              f'(found {len(a_var[dest1])} and {len(a_var[dest2])})')
        exit(0)


def get_params(argv):
    parser = argparse.ArgumentParser(description='Compare models.')

    parser.add_argument('--data', metavar='STR', help='List of csv files containing prediction vs ground truth',
                        type=str, required=True, nargs='+', action=required_length(1, 7))
    parser.add_argument('--title', metavar='STR', help='Plot title', type=str, default=None)
    parser.add_argument('--legend', metavar='STR', help='Legend specification', type=str, default=None, nargs='+')
    parser.add_argument('--show_pred', help='Toggle plot of prediction errors', default=False, action='store_true')
    parser.add_argument('--savefig', metavar='FILE', help='Save plot to file', default=None)

    a = parser.parse_args()

    check_equal_nargs('--data', '--legend', parser, a)

    return a.title, a.legend, a.data, a.show_pred, a.savefig


if __name__ == '__main__':
    title, legend, data_files, show_pred, savefig = get_params(sys.argv[1:])

    fig, axis = plt.subplots(nrows=1, ncols=1)

    if title is not None:
        fig.suptitle(title)

    colours = ('b', 'r', 'g', 'c', 'm', 'y', 'k')
    stats_df_list = []

    for i, data_file in enumerate(data_files):
        data_df = pd.read_csv(data_file, sep=',')
        stats_df = data_df.groupby('gt', as_index=False).quantile(0.5, numeric_only=True)

        first_quant = data_df.groupby('gt', as_index=False).quantile(0.05, numeric_only=True, )
        stats_df = stats_df.merge(first_quant, on='gt', suffixes=('', '_quant5'))

        second_quant = data_df.groupby('gt', as_index=False).quantile(0.95, numeric_only=True, )
        stats_df = stats_df.merge(second_quant, on='gt', suffixes=('', '_quant95'))

        # print(stats_df)
        stats_df_list.append(stats_df)
        axis.plot('gt', 'error', f'{colours[i]}-', data=stats_df)

    for i, stats_df in enumerate(stats_df_list):
        axis.plot('gt', 'error_quant5', f'{colours[i]}:', data=stats_df, alpha=0.5)
        axis.plot('gt', 'error_quant95', f'{colours[i]}:', data=stats_df, alpha=0.5)
        # axis.fill_between('gt', 'error_quant5', 'error_quant95', alpha=0.2, data=stats_df, color=f'{colours[i]}')
        if show_pred:
            pred_df = pd.read_csv(data_files[i], sep=',')
            axis.plot('gt', 'error', f'{colours[i]}D', markersize=2.5, data=pred_df, alpha=0.1)

    axis.grid(visible=True, axis='both', which='major')
    axis.set_aspect('equal', adjustable='datalim')

    if legend is not None:
        axis.legend(legend)
    else:
        axis.legend(data_files)

    if savefig is not None:
        plt.savefig(savefig)
    else:
        plt.show()
    print("Done.")
