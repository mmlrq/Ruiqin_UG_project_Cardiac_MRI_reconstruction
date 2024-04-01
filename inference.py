import os

################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
result_dir = "./results/ablation_step_100/"
#re
model_path = 'C:/Users/2524923/OneDrive - University of Dundee/Documents/final project/DiffCMR-main/log/t1_04_128/model12000.pt'
# val_pair_file = 'C:/Users/2524923/OneDrive - University of Dundee/Documents/final project/DiffCMR-main/AccFactor10_rMax_512_training_pair.txt'
val_pair_file = r'C:/Users/2524923/OneDrive - University of Dundee/Documents/final project/DiffCMR-main/AccFactor08_rMax_512_training_pair.txt'
val_bs = 24
################################################

from improved_diffusion import dist_util, logger
# from datasets.city import load_data, create_dataset
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
import warnings
warnings.filterwarnings('ignore')

import os
import random
import torchvision.transforms as transforms
from CMRxRecon import CMRxReconDataset
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
current_time = strftime("%m%d_%H_%M", gmtime())
current_day = strftime("%m%d", gmtime())


#inference check
from improved_diffusion.sampling_util import CMR_sampling_major_vote_func

dist_util.setup_dist()
logger.configure(dir=result_dir)
arg_dict = model_and_diffusion_defaults()

arg_dict["image_size"]=128
arg_dict["diffusion_steps"]=100
print(arg_dict)
model, diffusion = create_model_and_diffusion(**arg_dict)
model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
logger.log("creating model and diffusion...")
model.to(dist_util.dev())
model.eval()

tsfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128))
])
dataset = CMRxReconDataset(val_pair_file, transform=tsfm, limit_val=True)

CMR_sampling_major_vote_func(val_bs, diffusion, model, result_dir, dataset, logger, True, vote_num=4)

# CMR_GTINPUT_sampling_major_vote_func(val_bs, diffusion, model, result_dir, dataset, logger, True, vote_num=4)