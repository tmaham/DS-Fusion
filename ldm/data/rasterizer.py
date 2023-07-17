import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
import random
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from .list_fonts import font_list
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torchvision.transforms as T
import pdb 


color_multi_font =[
    '#a3001b', 
    '#175c7a',
    '#5cd6ce',
    '#d1440c',
    '#1f1775',
    '#9e1e46',
    '#9b100d',
    '#d3760c',
    '#e8bb06',
    '#5135ad',
    '#366993',
    '#470e7c',
    '#070707',
    '#053d22',
    '#7a354c',
    '#7c0b03'
]

alphabets = [
    'A','B','C','D','E','F','G','H','I','J','K','L',
    'M','N','O','P','Q','R','S','T','U','V','W','X', 'Y','Z'
]


class Rasterizer(Dataset):
    def __init__(self, 
                 text = "R", 
                 style_word = "DRAGON", 
                 data_path = "data/DRAGON", 
                 img_size = 256, 
                 num_samples = 1001, 
                 one_font = True, 
                 font_name = None,
                 custom_font = "None"):
        
        self.text = text
        self.use_custom = (custom_font != "None")

        if not self.use_custom:
            self.text = self.text[0].upper()

        self.img_size = img_size
        self.interpolation = PIL.Image.BILINEAR
        self.num_samples = num_samples
        self.style_word = style_word
        self.data_path = data_path
        self.dict = {}
        self.data_img = []
        self.one_font = one_font
        self.classes = []

        if font_name is not None:
            self.fontname = font_name
        else:
            fontname = random.choice(font_list)
            self.fontname = fontname
        if self.use_custom:
            self.fontname = custom_font

        self.load_back()

    def load_back(self):
        style_only = self.style_word.split(" ")[0]
        self.data_path = f"data_style/{style_only}/samples"
        self.data_img = []
        for file in os.listdir(self.data_path):
            self.data_img.append(os.path.join(self.data_path, file))

    def __len__(self):
        return self.num_samples

    def getSize(self, txt, font):
        testImg = Image.new('RGB', (1, 1))
        testDraw = ImageDraw.Draw(testImg)
        return testDraw.textsize(txt, font)

    def __getitem__(self, i):
        output = {}

        fontname = random.choice(font_list)
        if self.one_font or self.use_custom:
            fontname = self.fontname

        if self.use_custom:
            font = ImageFont.truetype(fontname, self.img_size)
            width, height = self.getSize(self.text, font)
            image = Image.new('RGB', (self.img_size, self.img_size), "white")
            d = ImageDraw.Draw(image)
            width_diff = self.img_size-width
            height_diff = self.img_size-height
            colorFont = random.choice(color_multi_font)
            d.text((width_diff/2,height_diff/2-32), self.text, fill=colorFont, font=font)
            img = np.array(image).astype(np.uint8)  
            image = Image.fromarray(img)
            # image = image.resize((self.img_size, self.img_size), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            image_text = (image / 127.5 - 1.0).astype(np.float32)
        else:
            rand_img = str(random.randint(0,15))
            fontname_t = fontname.split(".")[0]
            dir_font = f"data_fonts/{fontname_t}/{self.text}/{rand_img}"+".png"
            image = Image.open(dir_font)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = image.resize((self.img_size, self.img_size), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            image_text = (image / 127.5 - 1.0).astype(np.float32)

        output["image"] = image_text
        output["caption"] = self.text
        
        ##################################################################################
        output2 = {}
        ind = i % (len(self.data_img)-1)
        image = Image.open(self.data_img[ind])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = image.resize((self.img_size, self.img_size), resample=self.interpolation)
        
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
    
        output2["image"] = image
        output2["caption"] = self.style_word

        batch = {}
        batch["base"] = output
        batch["style"] = output2
        batch["font"] = fontname
        batch["number"] = 0

        batch["cond"] = self.style_word + " " + self.text
        
        return batch


