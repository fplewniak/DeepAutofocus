import argparse
import re
import sys

import pandas as pd


def get_params(argv):
    parser = argparse.ArgumentParser(description='Create annotation file.')

    parser.add_argument('--init', metavar='STR', help='Initial annotation file name', type=str, required=True)
    parser.add_argument('--correction', metavar='STR', help='Correction file name', type=str, required=True)
    parser.add_argument('--out', metavar='STR', help='Output file name', type=str, required=True)

    argscope = parser.parse_args()

    return argscope.init, argscope.correction, argscope.out


def create_annotation_file(initial_filename, correction_filename, out_filename):
    initial_annotations = pd.read_csv(initial_filename, sep=',', header=0)
    corrections = pd.read_csv(correction_filename, sep=',', header=0)
    corrected_annotations = initial_annotations.copy(deep=True)

    epsilon = 1e-10

    for row in initial_annotations.itertuples():
        for correction in corrections.itertuples():
            if re.search('/'+correction.FOV+'/', row.filename) is not None:
                new_deltaz = float(f'{row.deltaz + 0.2 * (correction.z_layer - 30):.2}')
                if abs(new_deltaz) < epsilon:
                    new_deltaz = 0.0
                # print(str(row.filename) + ': ' + f'{initial_annotations.iloc[row.Index]["deltaz"]} =>{new_deltaz}')
                corrected_annotations.loc[row.Index] = (row.filename, new_deltaz)

    print(corrected_annotations)
    corrected_annotations.to_csv(out_filename, sep=',', index=False)

if __name__ == '__main__':
    initial_filename, correction_filename, out_filename = get_params(sys.argv[1:])

    create_annotation_file(initial_filename, correction_filename, out_filename)
