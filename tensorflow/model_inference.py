"""
Runs N inferences on a model A (maybe pretrained) on GPU K with inputs X.
This file is turned into an executable and profiling is enabled while running the executable.
"""
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFAutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="resnet50", required=False)
parser.add_argument("-n", type=int, default=1, required=False,
                    help="number of inferences")
parser.add_argument("-gpu", type=int, default=-1, required=False,
                    help="-1 for cpu, else number of gpu")
# parser.add_argument("-input", type=str, default="random",
#                     help="Input type to pass to model. See construct_inputs.py")
# parser.add_argument("-seed", type=int, default=42, help="Random seed for random inputs.")
# parser.add_argument("-pretrained", action='store_true', help="Use a pretrained model")
# parser.add_argument("-load_path", default=None, required=False, help="Provide a path to a model to be used.")
parser.add_argument("-debug", default=False, help="Prints which device completes each operation")

args = parser.parse_args()

MODEL_MAP = {
    "resnet50": tf.keras.applications.ResNet50,
    "resnet101": tf.keras.applications.ResNet101,
    "resnet152": tf.keras.applications.ResNet152,
    "vgg16": tf.keras.applications.VGG16,
    "vgg19": tf.keras.applications.VGG19,
    "densenet121": tf.keras.applications.DenseNet121,
    "densenet169": tf.keras.applications.DenseNet169,
    "densenet201": tf.keras.applications.DenseNet201,
    "mobilenet_v2": tf.keras.applications.MobileNetV2,
    "mobilenet_v3_large": tf.keras.applications.MobileNetV3Large,
    "mobilenet_v3_small": tf.keras.applications.MobileNetV3Small,
    "inception_v3": tf.keras.applications.InceptionV3,
    "inception_resnet_v2": tf.keras.applications.InceptionResNetV2,
    "efficientnet_b0": tf.keras.applications.EfficientNetB0,
    "efficientnet_b1": tf.keras.applications.EfficientNetB1,
    "efficientnet_b2": tf.keras.applications.EfficientNetB2,
    "efficientnet_b3": tf.keras.applications.EfficientNetB3,
    "efficientnet_b4": tf.keras.applications.EfficientNetB4,
    "efficientnet_b5": tf.keras.applications.EfficientNetB5,
    "efficientnet_b6": tf.keras.applications.EfficientNetB6,
    "efficientnet_b7": tf.keras.applications.EfficientNetB7,
    "convnext_tiny": tf.keras.applications.ConvNeXtTiny,
    "convnext_small": tf.keras.applications.ConvNeXtSmall,
    "convnext_base": tf.keras.applications.ConvNeXtBase,
    "convnext_large": tf.keras.applications.ConvNeXtLarge,
    "convnext_xlarge": tf.keras.applications.ConvNeXtXLarge,
    "bert": lambda: hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"),
    "xlm_roberta_base": lambda: TFAutoModel.from_pretrained("jplu/tf-xlm-roberta-base"),
    "llama3": lambda: TFAutoModel.from_pretrained("facebook/llama-3b"),
    "llama": lambda: TFAutoModel.from_pretrained("facebook/llama-7b"),
    "gpt2": lambda: TFAutoModel.from_pretrained("gpt2"),
    "gemma": lambda: TFAutoModel.from_pretrained("gemma"),
    "bert": lambda: TFAutoModel.from_pretrained("bert-base-uncased"),
    "albert": lambda: TFAutoModel.from_pretrained("albert-base-v2"),
    "bart": lambda: TFAutoModel.from_pretrained("facebook/bart-large"),
}

name_to_family = {
    "resnet50": "resnet",
    "resnet101": "resnet",
    "resnet152": "resnet",
    "vgg16": "vgg",
    "vgg19": "vgg",
    "densenet121": "densenet",
    "densenet169": "densenet",
    "densenet201": "densenet",
    "mobilenet_v2": "mobilenet",
    "mobilenet_v3_large": "mobilenet",
    "mobilenet_v3_small": "mobilenet",
    "inception_v3": "inception",
    "inception_resnet_v2": "inception",
    "efficientnet_b0": "efficientnet",
    "efficientnet_b1": "efficientnet",
    "efficientnet_b2": "efficientnet",
    "efficientnet_b3": "efficientnet",
    "efficientnet_b4": "efficientnet",
    "efficientnet_b5": "efficientnet",
    "efficientnet_b6": "efficientnet",
    "efficientnet_b7": "efficientnet",
    "convnext_tiny": "convnext",
    "convnext_small": "convnext",
    "convnext_base": "convnext",
    "convnext_large": "convnext",
    "convnext_xlarge": "convnext",
    "bert": "bert",
    "xlm_roberta_base": "xlm_roberta",
    "llama3": "llama",
    "llama": "llama",
    "gpt2": "gpt2",
    "gemma": "gemma",
    "bert": "bert",
    "albert": "albert",
    "bart": "bart",
}

def getDeviceName(gpu_num: int) -> str:
    if gpu_num < 0:
        return "/device:CPU:0"
    return f"/device:GPU:{gpu_num}"

assert args.gpu <= len(tf.config.list_physical_devices('GPU'))
if args.debug:
    tf.debugging.set_log_device_placement(True)

assert args.model in MODEL_MAP, f"Valid models are {list(MODEL_MAP.keys())}"
model = MODEL_MAP[args.model]()

print(f"Running {args.n} inferences on {args.model} on {getDeviceName(args.gpu)}...")

with tf.device(getDeviceName(args.gpu)):
    for i in range(args.n):
        if name_to_family[args.model] in ["resnet", "vgg", "densenet", "mobilenet", "inception", "efficientnet", "convnext"]:
            input = tf.constant(0.0, dtype=tf.float32, shape=(224, 224, 3))
            input = np.expand_dims(input, axis=0)
            output = model(input)
        elif name_to_family[args.model] in ["bert", "xlm_roberta", "llama", "gpt2", "gemma", "albert", "bart"]:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            input_text = "This is a sample input text for the model."
            inputs = tokenizer(input_text, return_tensors='tf')
            output = model(inputs['input_ids'])
        
        print(output)

print("Completed.")
