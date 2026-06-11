from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C

# breaking Bad geometry assembly dataset configuration

__C.BREAKING_BAD = edict()

# directory containing the Breaking Bad dataset files
__C.BREAKING_BAD.DATA_DIR = "/workspace"

# file naming pattern for train/val/test splits
# will be formatted as "everyday.train.txt", "everyday.val.txt", etc.
__C.BREAKING_BAD.DATA_FN = (
    "everyday.{}.txt"
)

# additional data keys to load with each sample (e.g. part_ids for tracking pieces)
__C.BREAKING_BAD.DATA_KEYS = ("part_ids",)

# dataset subset selection
# empty string means no specific subset filtering
__C.BREAKING_BAD.SUBSET = ""  # must in ['artifact', 'everyday', 'other']

# category filtering - allows training/testing on specific object categories
# empty string means all categories will be used
__C.BREAKING_BAD.CATEGORY = ""

# all available object categories in the everyday subset
# breaking Bad dataset contains 20 everyday object categories
__C.BREAKING_BAD.ALL_CATEGORY = [
    "BeerBottle",
    "Bowl",
    "Cup",
    "DrinkingUtensil",
    "Mug",
    "Plate",
    "Spoon",
    "Teacup",
    "ToyFigure",
    "WineBottle",
    "Bottle",
    "Cookie",
    "DrinkBottle",
    "Mirror",
    "PillBottle",
    "Ring",
    "Statue",
    "Teapot",
    "Vase",
    "WineGlass",
]  # Only used for everyday

# rotation range for data augmentation during training
# -1.0 means random full rotation (SO(3)), otherwise specifies degree range
__C.BREAKING_BAD.ROT_RANGE = -1.0

# number of points to sample per object (not per part)
# total points are distributed among parts based on their surface area
__C.BREAKING_BAD.NUM_PC_POINTS = 5000  # points per part

# minimum number of points guaranteed for each piece when sampling by area
# ensures even small fracture pieces have sufficient points for feature extraction ( > 30)
__C.BREAKING_BAD.MIN_PART_POINT = 30

# range of number of parts per object
# objects with fewer than 2 or more than 20 parts are filtered out
__C.BREAKING_BAD.MIN_NUM_PART = 2
__C.BREAKING_BAD.MAX_NUM_PART = 20

# whether to shuffle the order of parts in each batch
# useful for ensuring the model doesn't rely on part ordering
__C.BREAKING_BAD.SHUFFLE_PARTS = False

# sampling strategy: "area" means points are sampled proportional to surface area
# this is more realistic than uniform point sampling across all pieces
__C.BREAKING_BAD.SAMPLE_BY = "area"

# dataset length controls (-1 means use full dataset)
__C.BREAKING_BAD.LENGTH = -1
__C.BREAKING_BAD.TEST_LENGTH = -1
__C.BREAKING_BAD.OVERFIT = -1

# threshold for determining fracture surface labels (in meters)
# points within this distance to nearest neighbor in another piece are labeled as fracture points
__C.BREAKING_BAD.FRACTURE_LABEL_THRESHOLD = 0.025

# color palette for visualizing different parts in point clouds
__C.BREAKING_BAD.COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]
