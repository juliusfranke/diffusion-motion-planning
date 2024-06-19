import argparse
import sys
import os
from icecream import ic
# from diffusion_model import trainRun, loadRun


def isFile(string: str):
    if os.path.isfile(string):
        return os.path.abspath(string)
    else:
        raise FileNotFoundError(string)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--list", action="store_true", help="list trained models")
    # Plots
    displayParserGroup = parser.add_argument_group("display")
    displayParserGroup.add_argument(
        "-pt", "--plot-training", action="store_true", help="plot training set"
    )
    displayParserGroup.add_argument(
        "-pe", "--plot-error", action="store_true", help="plot sampling error"
    )

    subparsers = parser.add_subparsers()

    # Training new model
    trainModelParser = subparsers.add_parser("train", help="train new model")
    trainModelParser.set_defaults(fn="trainRun")
    trainModelParser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10_000,
        metavar="AMOUNT",
        help="epochs to train",
    )
    trainModelParser.add_argument(
        "-t",
        "--trainingsize",
        type=int,
        default=1000,
        metavar="AMOUNT",
        help="size of training dataset",
    )
    trainModelParser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=1000,
        metavar="AMOUNT",
        help="amount of samples to generate",
    )
    trainModelParser.add_argument(
        "-nh",
        "--nhidden",
        type=int,
        default=4,
        metavar="AMOUNT",
        help="amount of hidden layers",
    )
    trainModelParser.add_argument(
        "-sh",
        "--shidden",
        type=int,
        default=256,
        metavar="AMOUNT",
        help="size of hidden layers",
    )

    datagroup = trainModelParser.add_mutually_exclusive_group()
    datagroup.add_argument("-g", "--generate", action="store_true")
    datagroup.add_argument("-l", "--load", metavar="FILE", type=isFile)

    # Loading model
    loadModelParser = subparsers.add_parser("load", help="load model from file")
    loadModelParser.set_defaults(fn="loadRun")
    loadModelParser.add_argument(
        "file", type=isFile, metavar="FILE", help="file to load"
    )

    args = parser.parse_args()

    if args.list:
        from diffusion_model import listRun

        listRun()
    elif args.fn == "trainRun":
        from diffusion_model import trainRun

        trainRun(args)
    elif "loadRun":
        from diffusion_model import loadRun

        loadRun(args)


if __name__ == "__main__":
    main()
