import argparse
import os
import sys
import logging

# from diffusion_model import trainRun, loadRun
logger = logging.getLogger(__name__)


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
    # trainModelParser.add_argument(
    #     "dataset", type=str, help="training dataset file (.parquet)"
    # )
    trainModelParser.add_argument(
        "config",
        type=isFile,
        help="yaml config for model layout",
    )
    trainModelParser.add_argument("name", type=str, help="output name")
    trainModelParser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10_000,
        help="epochs to train (default 10_000)",
    )
    trainModelParser.add_argument(
        "-t",
        "--trainingsize",
        type=int,
        default=-1,
        help="amount of data to load (default -1, for all)",
    )
    trainModelParser.add_argument(
        "-v",
        "--valsplit",
        type=float,
        default=0.8,
        help="validation split (default 0.8)",
    )
    # datagroup = trainModelParser.add_mutually_exclusive_group()
    # datagroup.add_argument("-g", "--generate", action="store_true")
    # datagroup.add_argument("-l", "--load", metavar="FILE", type=isFile)

    # Loading model
    loadModelParser = subparsers.add_parser("load", help="load model from file")
    loadModelParser.set_defaults(fn="loadRun")
    loadModelParser.add_argument("model", type=str, metavar="MODEL", help="model name")
    loadModelParser.add_argument(
        "load", type=isFile, metavar="FILE", help="training data file"
    )
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
    exportParser.add_argument("model", type=str, metavar="MODEL", help="model name")
    exportParser.add_argument(
        "-d", "--delta_0", type=float, default=0.5, metavar="DELTA", help="delta_0"
    )
    exportParser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=1000,
        nargs="*",
        metavar="AMOUNT",
        help="amount of samples to generate",
    )
    exportParser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=1,
        help="how often to repeat the process",
    )
    exportParser.add_argument(
        "-o", "--out", type=str, default=None, metavar="OUT", help="file to output to"
    )

    exportParser.add_argument(
        "-i",
        "--instance",
        type=isFile,
        default=None,
        metavar="INSTANCE",
        help="Path to extended instance yaml",
    )
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

        for model_size in args["samples"]:
            for trial in range(args["repeat"]):
                i_args = args.copy()
                i_args["samples"] = model_size
                i_args["out"] = (
                    i_args["out"]
                    .replace("MODEL_SIZE", str(model_size))
                    .replace("TRIAL", str(trial))
                )
                export(i_args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )
    main()
