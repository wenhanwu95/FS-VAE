import hashlib
import os
import pdb
import shutil
import urllib
import warnings
from distutils.text_file import TextFile
from turtle import forward

import torch
from torch import nn
from tqdm import tqdm

from .ViT import create_ViT_text

TEXT_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
}

def download(url: str, root: str = './text_extractor'):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    print(f'Downloading {filename}.')
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

class Text_Encoder(nn.Module):
    def __init__(self, text_model_type) -> None:
        super().__init__()
        model_path = download(TEXT_MODELS[text_model_type])
        text_state_dict = torch.jit.load(model_path, map_location=torch.device('cpu')).state_dict()
        text_args = {
            'embed_dim': text_state_dict["text_projection"].shape[1],
            'context_length': text_state_dict["positional_embedding"].shape[0],
            'vocab_size': text_state_dict["token_embedding.weight"].shape[0],
            'transformer_width': text_state_dict["ln_final.weight"].shape[0],
            'transformer_heads': text_state_dict["ln_final.weight"].shape[0] // 64,
            'transformer_layers': len(
                set(k.split(".")[2] for k in text_state_dict if k.startswith(f"transformer.resblocks"))),
            'emb_dropout': 0.
            }
        self.encoder = create_ViT_text(**text_args)
        self.encoder.load_state_dict(text_state_dict, strict=False)
        
    def forward(self, text):
        return self.encoder(text)
