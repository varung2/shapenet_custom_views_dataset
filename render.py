import os
import sys
import torch
import pickle
import ntpath
import pytorch3d
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt
from plot_image_grid import image_grid
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas,
)

sys.path.append(os.path.abspath(''))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

shapenet_dict = {
	"04379243" : "table",
	"03593526" : "jar",
	"04225987" : "skateboard",
	"02958343" : "car",
	"02876657" : "bottle",
	"04460130" : "tower",
	"03001627" : "chair",
	"02871439" : "bookshelf",
	"02942699" : "camera",
	"02691156" : "airplane",
	"03642806" : "laptop",
	"02801938" : "basket",
	"04256520" : "sofa",
	"03624134" : "knife",
	"02946921" : "can",
	"04090263" : "rifle",
	"04468005" : "train",
	"03938244" : "pillow",
	"03636649" : "lamp",
	"02747177" : "trash bin",
	"03710193" : "mailbox",
	"04530566" : "watercraft",
	"03790512" : "motorbike",
	"03207941" : "dishwasher",
	"02828884" : "bench",
	"03948459" : "pistol",
	"04099429" : "rocket",
	"03691459" : "loudspeaker",
	"03337140" : "file cabinet",
	"02773838" : "bag",
	"02933112" : "cabinet",
	"02818832" : "bed",
	"02843684" : "birdhouse",
	"03211117" : "display",
	"03928116" : "piano",
	"03261776" : "earphone",
	"04401088" : "telephone",
	"04330267" : "stove",
	"03759954" : "microphone",
	"02924116" : "bus",
	"03797390" : "mug",
	"04074963" : "remote",
	"02808440" : "bathtub",
	"02880940" : "bowl",
	"03085013" : "keyboard",
	"03467517" : "guitar",
	"04554684" : "washer",
	"02834778" : "bicycle",
	"03325088" : "faucet",
	"04004475" : "printer",
	"02954340" : "cap",
	"03991062" : "flowerpot",
	"03513137" : "helmet",
	"03046257" : "clock",
	"02992529" : "radiotelephone",
	"03761084" : "microwave",
}

shapenet_dict_inv = {value : key for (key, value) in shapenet_dict.items()}

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def pickle_camera_matrices(cameras, img_resolution, device):
	fovs = cameras.fov
	R = cameras.R
	T = cameras.T
	principal_offset = img_resolution//2
	Imats = []
	Emats = []
	for i in range(len(R)):
		flen = 1.0/torch.tan((fovs[i]*(np.pi/180.0))/2.0)
		intrinsic_mat = torch.from_numpy(np.array([[flen, 0, principal_offset],
			[0, flen, principal_offset],
			[0, 0, 1]], dtype=np.float32))
		Imats.append(intrinsic_mat)
		extrinsic_mat = torch.cat((R[i], T[i].unsqueeze(1)), 1)
		extrinsic_mat = torch.cat((extrinsic_mat, torch.cat((torch.zeros((3)), torch.ones((1))), 0).unsqueeze(0).to(device)), 0)
		Emats.append(extrinsic_mat)
	emats = torch.stack(Emats).cpu()
	imats = torch.stack(Imats).cpu()
	pickle_dict = {'intrinsic_matrices': imats, 'extrinsic_matrices': emats}
	return pickle_dict

# parameters
NUM_VIEWS = 20
IMAGE_RESOLUTION = 256
NUM_OBJS_PER_CATEGORY = 250
SHAPENET_PATH = "../ShapeNetCore.v2"
CATEGORIES = ["table", "chair", "sofa", "stove", "piano", "mailbox", "file cabinet", "bench", "bed", "bathtub"]
# CATEGORIES = ["chair", "piano", "bed", "stove", "mailbox"]
# CATEGORIES = ["piano", "stove", "bed", "mailbox"]

## DEBUGGING THE TYPES OF CATEGORIES AVAILABLE
# available_categories = [shapenet_dict[path_leaf(f.path)] for f in os.scandir(SHAPENET_PATH) if f.is_dir()]
# print('Available categories: {}'.format(available_categories))
# Available categories: ['pillow',
#  'earphone',
#  'car',
#  'bowl',
#  'train',
#  'cabinet',
#  'watercraft',
#  'table',
#  'trash bin',
#  'cap',
#  'file cabinet',
#  'jar',
#  'keyboard',
#  'telephone',
#  'airplane',
#  'motorbike',
#  'birdhouse',
#  'dishwasher',
#  'rocket',
#  'mug',
#  'sofa',
#  'flowerpot',
#  'knife',
#  'helmet',
#  'bench',
#  'printer',
#  'clock',
#  'radiotelephone',
#  'skateboard',
#  'washer',
#  'basket',
#  'laptop',
#  'guitar',
#  'camera',
#  'bed',
#  'bus',
#  'bottle',
#  'lamp',
#  'pistol',
#  'tower',
#  'bookshelf',
#  'microphone',
#  'mailbox',
#  'rifle',
#  'bathtub',
#  'chair',
#  'display',
#  'faucet',
#  'microwave',
#  'remote',
#  'bag',
#  'loudspeaker',
#  'piano',
#  'stove',
#  'can']

# setting up the dataset directory if it doesn't exist
root = "dataset/"
for cat in CATEGORIES:
	Path(os.path.join(root, cat.replace(' ', '_'))).mkdir(parents=True, exist_ok=True)

# init raw dataset
shapenet_dataset = ShapeNetCore(data_dir=SHAPENET_PATH, version=2)

## Render settings
elev = torch.linspace(1, 1.5, NUM_VIEWS)
azim = torch.linspace(-180, 180, NUM_VIEWS+1)[:NUM_VIEWS]
print('DEBUG: azimuth views angles: {}'.format(azim))
R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
cam_dict = pickle_camera_matrices(cameras, IMAGE_RESOLUTION, device)
pickle.dump(cam_dict, open(os.path.join(root, 'camera_matrices.pkl'), 'wb'))
raster_settings = RasterizationSettings(
    image_size=IMAGE_RESOLUTION, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
lights = PointLights(device=device, location=[[0.0, 1.0, -2.0]])
rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
renderer = MeshRenderer(
    rasterizer=rasterizer,
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)
idxlist = [
	shapenet_dataset._sample_idxs_from_category(sample_num=NUM_OBJS_PER_CATEGORY, category=shapenet_dict_inv[cat])
	for cat in CATEGORIES
	]
idxlist = torch.flatten(torch.stack(idxlist))

obj_iter_count = dict()
for idx in tqdm(idxlist):
	model = None
	try:
		model = shapenet_dataset[idx]
	except Exception as err:
		print('Encountered exception error: "{}"'.format(err))
		continue

	model_verts, model_faces, model_textures = model["verts"], model["faces"], model["textures"]
	model_textures = TexturesAtlas(atlas=model_textures.unsqueeze(0).to(device))
	mesh = Meshes(
	    verts=[model_verts.to(device)],   
	    faces=[model_faces.to(device)],
	    textures=model_textures
	)
	mesh = mesh.extend(NUM_VIEWS)
	images = renderer(mesh)
	depth_img = rasterizer(mesh).zbuf.view(NUM_VIEWS, IMAGE_RESOLUTION, IMAGE_RESOLUTION)
	dmask = ((depth_img > 0.0).type(torch.int)).type(torch.float32)

	# for file naming
	object_count = obj_iter_count.get(model['label'], 0)
	if (object_count == 0): obj_iter_count[model['label']] = 0
	
	# saves the set of images with different views
	for imgidx, (img, dimg) in enumerate(zip(images,dmask)): 
		file_name = '{}_{}.png'.format(object_count, imgidx)
		file_path = os.path.join(root, model['label'].replace(' ', '_'), file_name)
		img[..., 3] = dimg
		plt.imsave(file_path, img.cpu().numpy())

	obj_iter_count[model['label']] += 1