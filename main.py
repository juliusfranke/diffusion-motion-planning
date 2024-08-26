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
        "model", type=str, metavar="MODEL", help="model name"
    )
    loadModelParser.add_argument("load", type=isFile, metavar="FILE", help="training data file")
    loadModelParser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=1000,
        metavar="AMOUNT",
        help="amount of samples to generate",
    )
    loadModelParser.add_argument(
        "-ph", "--plot-hist", action="store_true", help="plot histogram of actions"
    )

    # Export motion primitives
    exportParser = subparsers.add_parser("export", help="export motion primitives")
    exportParser.set_defaults(fn="export")
    exportParser.add_argument(
        "model", type=str, metavar="MODEL", help="model name"
    )
    exportParser.add_argument(
        "-d",
        "--delta",
        type=float,
        default=0.5,
        metavar="DELTA",
        help="delta_0")
    exportParser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=1000,
        metavar="AMOUNT",
        help="amount of samples to generate",
    )
    exportParser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        metavar="OUT",
        help="file to output to")

    args = parser.parse_args()

    if args.list:
        from model_runner import listRun

        listRun()
    elif args.fn == "trainRun":
        from model_runner import trainRun

        args = vars(args)
        trainRun(args)
    elif args.fn == "loadRun":
        from model_runner import loadRun

        args = vars(args)
        loadRun(args)
    elif args.fn == "export":
        from model_runner import export

        args = vars(args) 
        export(args)

if __name__ == "__main__":
    main()
