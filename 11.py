import ads3 as ads3
import torch
import torchvision.transforms as transforms

import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def, fn, types

from pathlib import Path
from PIL import Image
from torch.utils import data as D

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
    
    images = fn.decoders.image(inputs, device = "cpu")

    images = fn.resize(images, resize_shorter=INPUT_SIZE)
    images = fn.crop(images, crop=(INPUT_SIZE, INPUT_SIZE))

    flip_coin = fn.random.coin_flip()
    images = fn.flip(images, horizontal = flip_coin, device = "cpu")

    """
    This does not work as the shear does not support Data nodes as parameters
    sxy = fn.random.uniform(range=[0.5, 1.5], dtype=types.FLOAT)
    sxz = fn.random.uniform(range=[0.5, 1.5], dtype=types.FLOAT)
    syx = fn.random.uniform(range=[0.5, 1.5], dtype=types.FLOAT)
    syz = fn.random.uniform(range=[0.5, 1.5], dtype=types.FLOAT)
    szx = fn.random.uniform(range=[0.5, 1.5], dtype=types.FLOAT)
    szy = fn.random.uniform(range=[0.5, 1.5], dtype=types.FLOAT)

    # this is equivalent to perspective
    images_sheared = fn.transforms.shear(
        images,
        shear=(sxy, sxz)
    )

    flip_coin = fn.random.coin_flip()
    # dali does not yet support conditional execution, this is workaround
    images = mux(flip_coin, images, images_sheared)
    """

    images = fn.transpose(images, perm=[2, 0, 1])

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
    
    images = fn.decoders.image(inputs, device = "cpu")

    images = fn.resize(images, resize_shorter=INPUT_SIZE)
    images = fn.crop(images, crop=(INPUT_SIZE, INPUT_SIZE))

    images = fn.transpose(images, perm=[2, 0, 1])

    images = fn.normalize(images, dtype=types.FLOAT)

    labels = fn.squeeze(labels, axes=[0])
    
    return images.gpu(), labels.gpu()

class DaliTrain():
    def __init__(self):
        self.dataset = DALIGenericIterator(
            # exec params set to false because we rely on torch for one of the transforms
            train_pipeline(device_id=0, batch_size=20, num_threads=1, exec_async=False, exec_pipelined=False, seed=0),
            ["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def __len__(self):
        with open("data/train_final.txt","r") as f:
            return len(f.readlines())

class DaliValid():
    def __init__(self):
        self.dataset = DALIGenericIterator(
            valid_pipeline(device_id=0, batch_size=20, num_threads=1, exec_async=False, exec_pipelined=False, seed=0),
            ["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def __len__(self):
        with open("data/valid_final.txt","r") as f:
            return len(f.readlines())


if __name__ == "__main__":
    loader_train = DaliTrain() 
    loader_valid = DaliValid()   

    print("train size: %d, valid size %d" % (len(loader_train), len(loader_valid)))

    log_file = "out/11py.csv"
    trace_file = "out/trace11py.json"

    ads3.run_experiment(
        loader_train, loader_valid, log_file, trace_file
    )  # For profiling feel free to lower epoch count via epoch=X
