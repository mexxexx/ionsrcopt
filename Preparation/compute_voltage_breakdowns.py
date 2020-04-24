import numpy as np
import pandas as pd

import argparse
import sys, os

sys.path.insert(1, os.path.abspath("../ionsrcopt"))
import load_data as ld
from source_features import SourceFeatures
from processing_features import ProcessingFeatures


def main(input_file):
    file_path = "../Data_Preprocessed/" + input_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args.input_file + ".csv")
