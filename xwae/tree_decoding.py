import os

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from alignlib.metric_search_tree import DynamicSearchMetricTree


def decode_tree(decoding_func,  dtms: DynamicSearchMetricTree, to_folder, device='cpu', max_depth=10):
    """decoding the tree results in the folder trees
    """
    def decode_node(node, to_folder, depth, max_depth):
        node_path = os.path.join(to_folder, 'node_{}'.format(node.idx))
        os.makedirs(node_path, exist_ok=True)
        fp = os.path.join(node_path, 'decoded_latent_node_feature.png')
        # decoded_image = decoding_func(torch.from_numpy(node.vec).to(device))[0]
        # decoded_image = decoded_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # im = Image.fromarray(decoded_image)
        # im.save(fp, format=format)
        decoded_image = decoding_func(torch.from_numpy(node.vec).to(device))
        vutils.save_image(decoded_image, fp)
        for s in node.sub:
            decode_node(s, node_path, depth+1, max_depth)

    decode_node(dtms.root, to_folder, 0, max_depth=max_depth)
