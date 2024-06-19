"""
Generates an input to feed to a model.  Seed can be specified for random inputs.
Configuration of input size comes from config.py
"""
import config
import torch
from torchtext.functional import to_tensor

# Random English Paragraph
LLM_INPUT = ["The rain and wind abruptly stopped, but the sky still had the gray swirls of storms in the distance. \
            Dave knew this feeling all too well. The calm before the storm. \
            He only had a limited amount of time before all Hell broke loose, but he stopped to admire the calmness.\
            Maybe it would be different this time, he thought, with the knowledge deep within that it wouldn't."]

VALID_INPUTS = {
    "random": torch.randn,
    "0": torch.zeros,
    "1": torch.ones,
    "text": LLM_INPUT
}

def construct_input(type: str, number: int, seed=None, ftn=None) -> torch.Tensor:
    if type not in VALID_INPUTS:
        raise ValueError(f"Provided input argument {type} but valid options are {list(VALID_INPUTS.keys())}.")
    
    if seed:
        torch.manual_seed(seed)

    if ftn:
        return to_tensor(ftn(LLM_INPUT), padding_value=1)
    return VALID_INPUTS[type](number, config.CHANNELS, config.INPUT_SIZE, config.INPUT_SIZE)