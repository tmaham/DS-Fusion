import os 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="input directory",
    )

opt, unknown = parser.parse_known_args()
input_imgs = os.listdir(opt.input_dir)

for file in input_imgs:
    fpath = os.path.join(opt.input_dir, file)
    fpath_out = fpath.split('.')[0] + '_bg.png'
    cmd = f'rembg i {fpath} {fpath_out}'
    print(cmd)
    os.system(cmd)