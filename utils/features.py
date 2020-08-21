import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'
vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
vggish.eval()
def get_vggish_embedding(filename):
    return torch.as_tensor(vggish.forward(filename))