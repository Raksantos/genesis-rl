import os
import random
import numpy as np
import torch
import genesis as gs
from dotenv import load_dotenv


def set_global_seed():
    load_dotenv()
    seed = int(os.getenv("SEED", "42"))

    gs.init(seed=seed, backend=gs.constants.backend.gpu, logging_level="Warning")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
