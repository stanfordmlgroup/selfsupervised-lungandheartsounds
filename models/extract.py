import torch
import argparse
from models import SSL, ResNetSimCLR
import os
from shutil import copyfile

def scrape_model(log_dir, model_file):
    encoder = ResNetSimCLR(out_dim=256, base_model="resnet18")
    model = SSL(encoder)
    try:
        copyfile(os.path.join(log_dir, 'encoder.pth'),os.path.join(log_dir, 'checkpoints', 'encoder.pth'))
    except:
        print("No original model to backup")
    state_dict = torch.load(os.path.join(log_dir, model_file))
    model.load_state_dict(state_dict)
    encoder = model.encoder

    torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pth'))
    print("Scraped model saved to {}/encoder.pth".format(log_dir))
    with open(os.path.join(log_dir,'extract_log.txt'),'w') as f:
        f.write("Scraped model, {}, saved to {}/encoder.pth".format(model_file, log_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str, default=None)
    args = parser.parse_args()

    scrape_model(args.log_dir, args.model_file)
