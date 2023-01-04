import ads3 as ads3
import torch
import torchvision.transforms as transforms

import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def, fn, types

from pathlib import Path
from PIL import Image
from torch.utils import data as D

import argparse

# You should probably import some of DALI here

torch.manual_seed(0)
INPUT_SIZE = 224
root = Path("data")
file_train = root / "train.txt"
folder_images = root / "image"

# based on https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/expressions/expr_conditional_and_masking.html
def mux(condition, true_case, false_case):
    neg_condition = condition ^ True
    return condition * true_case + neg_condition * false_case

@pipeline_def
def train_pipeline():
    inputs, labels = fn.readers.file(
            file_list="data/train_final.txt",
            random_shuffle=True,
            prefetch_queue_depth=2,
            name="Reader",
        )
    
    images = fn.decoders.image(inputs, device = "mixed")

    images = fn.resize(images, resize_shorter=INPUT_SIZE)
    images = fn.crop(images, crop=(INPUT_SIZE, INPUT_SIZE))

    flip_coin = fn.random.coin_flip()
    images = fn.flip(images, horizontal = flip_coin)

    images = dalitorch.fn.torch_python_function(
        images, 
        function=transforms.Compose([
            transforms.RandomPerspective(p=0.5),
            ])
        )

    images_jittered = fn.color_twist(
        images, 
        contrast=fn.random.uniform(range=[0.7, 1.3]), 
        saturation=fn.random.uniform(range=[0.7, 1.3]),
        hue=fn.random.uniform(range=[0.7, 1.3]),
        brightness=fn.random.uniform(range=[0.5, 1.5])
        )

    flip_coin = fn.random.coin_flip()
    # dali does not yet support conditional execution, this is workaround
    images = mux(flip_coin, images, images_jittered)

    images = fn.transpose(images, perm=[2, 0, 1])

    images = fn.normalize(images, dtype=types.FLOAT)

    labels = fn.squeeze(labels, axes=[0])
    
    return images.gpu(), labels.gpu()

@pipeline_def
def valid_pipeline():
    inputs, labels = fn.readers.file(
            file_list="data/valid_final.txt",
            random_shuffle=True,
            prefetch_queue_depth=2,
            name="Reader",
        )
    
    images = fn.decoders.image(inputs, device = "mixed")

    images = fn.resize(images, resize_shorter=INPUT_SIZE)
    images = fn.crop(images, crop=(INPUT_SIZE, INPUT_SIZE))

    images = fn.transpose(images, perm=[2, 0, 1])

    images = fn.normalize(images, dtype=types.FLOAT)

    labels = fn.squeeze(labels, axes=[0])
    
    return images.gpu(), labels.gpu()

class DaliTrain():
    def __init__(self, threads):
        self.dataset = DALIGenericIterator(
            # exec params set to false because we rely on torch for one of the transforms
            train_pipeline(device_id=0, batch_size=20, num_threads=threads, exec_async=False, exec_pipelined=False, seed=0),
            ["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def __len__(self):
        with open("data/train_final.txt","r") as f:
            return len(f.readlines())

class DaliValid():
    def __init__(self, threads):
        self.dataset = DALIGenericIterator(
            valid_pipeline(device_id=0, batch_size=20, num_threads=threads, exec_async=False, exec_pipelined=False, seed=0),
            ["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def __len__(self):
        with open("data/valid_final.txt","r") as f:
            return len(f.readlines())


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j", "--threads", type=int,
        default=1
        )
    parser.add_argument(
        "--logfile", default="out/9py.csv"
    )
    parser.add_argument(
        "--tracefile", default="out/trace9py.json"
    )
    parser.add_argument(
        "-p", "--profile", action="store_true"
    )
    return parser

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    if args.profile:
        import ads3_pt_profile as ads3
    else:
        import ads3 as ads3

    loader_train = DaliTrain(args.threads) 
    loader_valid = DaliValid(args.threads)   

    print("train size: %d, valid size %d" % (len(loader_train), len(loader_valid)))

    log_file = args.logfile
    trace_file = args.tracefile

    ads3.run_experiment(
        loader_train, loader_valid, log_file, trace_file
    )  # For profiling feel free to lower epoch count via epoch=X
