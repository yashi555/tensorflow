import argparse
import os
import logging
import sys

from darkflow.net.build import TFNet

logger = logging.getLogger("Rotating Log")
logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logger.setLevel(logging.INFO)




def save_in_pbformat(args):
    args.load = -1
    tfnet = TFNet(args)
    tfnet.savepb()


if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(cwd + "/cfg/yolov2-voc.cfg"),
        help="CONFIGURATION TO BE USED FOR TRAINING YOLO MODEL",
    )
    parser.add_argument(
        "--load",
        default=os.path.join(cwd + "/bin/yolov2-voc.weights"),
        help=""
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help=""
    )
    parser.add_argument(
        "--annotation",
        default=os.path.join(cwd + '/new_data/annots'),
        help=""
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(cwd + '/new_data/images'),
        help=""
    )
    parser.add_argument(
        "--pbLoad",
        default='',
        help=""
    )
    parser.add_argument(
        "--metaLoad",
        default='',
        help=""
    )
    parser.add_argument(
        "--binary",
        default='',
        help=""
    )
    parser.add_argument(
        "--config",
        default='',
        help=""
    )
    parser.add_argument(
        "--labels",
        default=os.path.join(cwd + '/labels.txt'),
        help=""
    )
    parser.add_argument(
        "--threshold",
        default=-0.1,
        type=float,
        help=""
    )
    parser.add_argument(
        "--verbalise",
        default=True,
        help=""
    )
    parser.add_argument(
        "--gpu",
        default=0.0,type=float,help=""
    )
    parser.add_argument(
        "--trainer",
        default='rmsprop', help=""
    )
    parser.add_argument(
        "--lr",default=0.0001,
        type=float,help=""
    )
    parser.add_argument(
        "--summary",
        default='', help=""
    )
    parser.add_argument(
        "--keep",
        default=20, help=""
    )
    parser.add_argument(
        "--batch",
        default=16, help=""
    )
    parser.add_argument(
        "--epoch",type=int,default=5,
        help=""
    )
    parser.add_argument(
        "--save",
        default=2000, help=""
    )
    parser.add_argument(
        "--backup",
        default=os.path.join(cwd + '/ckpt/'), help=""
    )
    args = parser.parse_args()
    tfnet = TFNet(args)
    tfnet.train()
    save_in_pbformat(args)


