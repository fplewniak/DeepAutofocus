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

    parser.add_argument('--data', metavar='FILE', type=str,
                        help='CSV file containing information about data to plot', required=True)
    # parser.add_argument('--data', metavar='STR', help='List of csv files containing prediction vs ground truth',
    #                     type=str, required=True, nargs='+', action=required_length(1, 16))
    parser.add_argument('--title', metavar='STR', help='Plot title', type=str, default=None)
    # parser.add_argument('--legend', metavar='STR', help='Legend specification', type=str, nargs='+',
    #                     action=equal_nargs('--data'))
    parser.add_argument('--savefig', metavar='FILE', help='Save plot to file', default=None)
    parser.add_argument('--violin', help='Plot violin plot instead of boxplot', default=False, action='store_true')

    a = parser.parse_args()

    # check_equal_nargs('--data', '--legend', parser, a)

    # return a.title, a.legend, a.data, a.savefig, a.violin
    return a.title, a.data, a.savefig, a.violin


if __name__ == '__main__':
    # title, legend, data_files, savefig, violin = get_params(sys.argv[1:])
    title, file_list, savefig, violin = get_params(sys.argv[1:])

    files_df = pd.read_csv(file_list)
    # print(files_df)

    fig, axis = plt.subplots(nrows=1, ncols=1)

    if title is not None:
        fig.suptitle(title)

    data_df = None
    legend = []

    # for i, data_file in enumerate(data_files):
    for row in files_df.iterrows():
        data_file = row[1]['filename']
        # legend = row[1]['legend']
        legend.append(row[1]['legend'])
        if data_df is None:
            data_df = pd.read_csv(data_file, sep=',', header=0, names=['pred', 'gt', 'filename', legend[-1]])
            data_df = data_df.drop(['pred', 'gt'], axis=1)
        else:
            df = pd.read_csv(data_file, sep=',', header=0, names=['pred', 'gt', 'filename', legend[-1]])[['filename', legend[-1]]]
            data_df = pd.merge(data_df, df, on="filename")

    if violin:
        axis.violinplot(data_df[legend], showmedians=False, showextrema=False)

    axis.boxplot(data_df[legend], tick_labels=legend, whis=1.5, notch=True, sym='.')

    for label in axis.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    axis.grid(visible=True, axis='y', which='major', linestyle='dotted')
    # axis.set_aspect('equal', adjustable='datalim')

    # axis.set_xticklabels(legend)

    # if legend is not None:
    #     axis.legend(legend)
    # else:
    #     axis.legend(data_files)

    if savefig is not None:
        plt.savefig(savefig)
    else:
        plt.show()
    print("Done.")
