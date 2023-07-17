#generate the stylized images 

import os 
import pdb 
import random 
import argparse
import sys
from ldm.data.list_fonts import font_list

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-s",
        "--style",
        type=str,
        const=True,
        default="",
        nargs="?",
        required=True,
        help="style word for generation, preferably nouns",
    )

    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="",
        nargs="?",
        required=True,
        help="text to stylize",
    )

    parser.add_argument(
        "--one_font",
        type=str2bool,
        default=True,
        help="determines whether a single font or multiple fonts will be used",
    )

    parser.add_argument(
        "--font_name",
        type=str,
        default="None",
        help="explicitly give a font to stylize, only relevant for one font font-mode",
    )

    parser.add_argument(
        "--custom_font",
        type=str,
        default="None",
        help="user needs to ensure the font name is valid, and available on their system",
    )

    parser.add_argument(
        "--attribute",
        type=str,
        default="None",
        help="additional attribute for style images",
    )

    parser.add_argument(
        "--white_bg",
        type=str2bool,
        default=True,
        help="style images generated with 'white background'",
    )

    parser.add_argument(
        "--cartoon",
        type=bool,
        default=True,
        help="style images generated with cartoon attribute",
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        default="data_style/",
        help="where to store style images",
    )

    parser.add_argument(
        "--data_make",
        type=str2bool,
        default=False,
        help="this will enforce style data to be re-generated, even if it already exists",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default= "0,",
        help="which gpus to use. write as 0,1, etc",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default= "ckpt/model.ckpt",
        help="path to checkpoint of base",
    )

    return parser


if __name__ == "__main__":

    
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    path_data = os.path.join(opt.data_folder)
    

    command = "mkdir "+ path_data
    os.system(command) 

    prompt_cond = ""
    if opt.cartoon:
        prompt_cond = prompt_cond + "cartoon"
    if opt.white_bg:
        prompt_cond = prompt_cond + " with white background"
    
    if opt.attribute != "None":
        prompt = f"'{opt.style} illustration {prompt_cond} in style of {opt.attribute}'"
    else:
        prompt = f"'{opt.style} illustration {prompt_cond}'"

    if opt.custom_font == "None":
        if opt.font_name == "None":
            font_name = random.choice(font_list)
        else:
            font_name = opt.font_name
    else:
        font_name = opt.custom_font
    
    
    log_dir = opt.style +"-"+opt.text
    data_exists = os.path.exists("data_style/"+opt.style)

    if opt.data_make or not data_exists:
        print("Making style images")
        command = "rm -r " + "data_style/"+opt.style
        os.system(command)

        command = "mkdir -p " + "data_style/"+opt.style
        os.system(command)
        
        output_path = f"data_style/{opt.style}"
        li = opt.ckpt_path
        command = "python txt2img.py --ddim_eta 1.0 \
                            --n_samples 10 \
                            --n_iter 2\
                            --ddim_steps 50 \
                            --scale 5.0\
                            --H 256\
                            --W 256\
                            --outdir " + output_path + " --ckpt " +li +" --prompt " + prompt
        os.system(command)
    else:
        print("Style images exist")
    
   
    command = "mkdir out_cur"
    os.system(command)

    config_name = "finetune"

    command = "python main.py --base configs/"+config_name+".yaml \
                -t -n test --gpus 0, --ckpt_resume " + opt.ckpt_path +"\
                --logname " +log_dir+ " --letter "\
                + opt.text + " --style_word " + opt.style + " --font_name " +font_name +\
                (" --one_font True" if opt.one_font else " ") +\
                        " --images " +path_data +" --extra_style_word  "+opt.attribute + " --custom_font " + opt.custom_font
    
    os.system(command)

