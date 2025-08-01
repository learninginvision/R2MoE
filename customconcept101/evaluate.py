import argparse
import glob
import json
import os
import warnings
from pathlib import Path
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import clip
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from packaging import version
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
import csv
import torch_fidelity
#torch.cuda.set_device(0)

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, append=False, prefix='A photo depicts'):
        self.data = data
        self.prefix = ''
        if append:
            self.prefix = prefix
            if self.prefix[-1] != ' ':
                self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


def Convert(image):
    return image.convert("RGB")


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


class DINOImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(256, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8, append=False):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, append=append),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, datasetclass, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        datasetclass(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            if hasattr(model, 'encode_image'):
                if device == 'cuda':
                    b = b.to(torch.float16)
                all_image_features.append(model.encode_image(b).cpu().numpy())
            else:
                all_image_features.append(model(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.shape[0])+int(target.shape[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2) 
    
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    #高斯核函数的数学表达式
    kernel_val = [torch.exp((-L2_distance / bandwidth_temp).float()) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)
          
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=1):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    source = torch.tensor(source)
    target= torch.tensor(target)
    n = source.shape[0]#
    m = target.shape[0]
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:n, :n]
    YY = kernels[m:, m:]
    XY = kernels[:n, m:]
    YX = kernels[m:, :n]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算

def get_clip_score(model, images, candidates, device, append=False, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device, append=append)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
        candidates = candidates / \
            np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def clipeval(image_dir, candidates_json, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]
    image_ids = [Path(path).stem for path in image_paths]
    with open(candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]

    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, CLIPImageDataset, device, batch_size=64, num_workers=8)

    _, per_instance_image_text, _ = get_clip_score(
        model, image_feats, candidates, device)

    scores = {image_id: {'CLIPScore': float(clipscore)}
              for image_id, clipscore in
              zip(image_ids, per_instance_image_text)}
    print('CLIPScore: {:.4f}'.format(
        np.mean([s['CLIPScore'] for s in scores.values()])))

    return np.mean([s['CLIPScore'] for s in scores.values()]), np.std([s['CLIPScore'] for s in scores.values()])


def clipeval_image(image_dir, image_dir_ref,num_IA, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]#sample_imgs
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)#real_imgs
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]
    image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    #image_paths_50 = [os.path.join(image_dir, path) for path in path_list
    #               if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]#sample_imgs
    image_paths_50 = image_paths[:num_IA]
    
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    image_feats_50 = extract_all_images(
        image_paths_50, model, CLIPImageDataset, device, batch_size=64, num_workers=8)#sample_imgs_feature
    image_feats = extract_all_images(
        image_paths, model, CLIPImageDataset, device, batch_size=64, num_workers=8)#sample_imgs_feature
    image_feats_ref = extract_all_images(
        image_paths_ref, model, CLIPImageDataset, device, batch_size=64, num_workers=8)#real_imgs_feature

    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_50=image_feats_50 / \
        np.sqrt(np.sum(image_feats_50 ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    #loss=mmd_rbf(image_feats_ref,image_feats) 
    loss=0
    res = image_feats @ image_feats_ref.T
    res_50 =  100*(image_feats_50 @ image_feats_ref.T)
    return np.mean(res),np.mean(res_50),loss


def dinoeval_image(image_dir, image_dir_ref, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats_ref = extract_all_images(
        image_paths_ref, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res)

def iseval_image(image_dir, num_IA):
    #image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    path_list = os.listdir(image_dir)
    path_list.sort(key=lambda x:int(x[:-4]))
    image_files = [os.path.join(image_dir, path) for path in path_list
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]#sample_imgs
    # 取每个prompt的50张图像
    image_batches = [image_files[i:i+num_IA] for i in range(0, len(image_files), num_IA)]
    results = []
    for batch in image_batches:
        temp_dir = f"temp_selected_images_{len(results)}"
        os.makedirs(temp_dir, exist_ok=True)
        for idx, image in enumerate(batch):
            os.system(f'cp {image} {temp_dir}/image_{idx}.png')

        res = torch_fidelity.calculate_metrics(
            input1=temp_dir,
            cuda=True,
            isc=True,
            verbose=False,
        )
        results.append(res['inception_score_mean'])
        os.system(f'rm -r {temp_dir}')

    # 计算20个prompts的平均值
    average_result = sum(results) / len(results)    
    return average_result

def calmetrics(sample_root, target_paths, numgen,json_path, outpkl):
    device = 'cuda'

    if os.path.exists(outpkl):
        df = pd.read_pickle(outpkl)
    else:
        df = pd.DataFrame()
    full = {}
  
    assert sample_root.is_dir()
    image_path = sample_root / 'samples'
    #json_path = sample_root / 'prompts.json'
    

    #assert len(glob.glob(str(image_path / '*.png'))) == numgen, "Sample folder does not contain required number of images"

    textalignment, _ = \
        clipeval(str(image_path), str(json_path), device)

    sd = {}
    sd['CLIP Text alignment'] = textalignment

    for i, target_path in enumerate(target_paths.split('+')):
        imagealignment,imagealignment_50,MMDimagealignment = \
            clipeval_image(str(image_path), target_path,int(numgen/20), device)

        dinoimagealignment = \
            dinoeval_image(str(image_path), target_path, device)
        inception_score_mean=0
        #inception_score_mean = iseval_image(str(image_path), num_IA)
        if i > 0:
            sd[f'CLIP Image alignment{i}'] = imagealignment
            sd[f'CLIP Image alignment_50{i}'] = imagealignment_50
            sd[f'DINO Image alignment{i}'] = dinoimagealignment
            sd[f'Inception Score{i}'] = inception_score_mean
        else:
            sd['CLIP Image alignment'] = imagealignment
            sd[f'CLIP Image alignment_50'] = imagealignment_50
            sd['DINO Image alignment'] = dinoimagealignment
            #sd['MMD Image alignment'] = MMDimagealignment
            sd['Inception Score'] = inception_score_mean

    expname = sample_root
    if expname not in full:
        full[expname] = sd
    else:
        full[expname] = {**sd, **full[expname]}
    print(sd)

    print("Metrics:", full)
    '''
    for expname, sd in full.items():
        if expname not in df.index:
            df1 = pd.DataFrame(sd, index=[expname])
            df = pd.concat([df, df1])
        else:
            df.loc[df.index == expname, sd.keys()] = sd.values()

    df.to_pickle(outpkl)
    '''
    return sd


def parse_args():
    parser = argparse.ArgumentParser("metric", add_help=False)
    parser.add_argument("--sample_root", type=str,
                        help="the root folder to generated images")
    parser.add_argument("--numgen", type=int, default=100,
                        help="total number of images.")
    parser.add_argument("--target_paths", type=str,
                        help="+ separated paths to real target images")
    parser.add_argument("--prompt_root", type=str,help="the path of prompts for evaluate")
    parser.add_argument("--outpkl", type=str, default="evaluation.pkl",
                        help="the path to save result pkl file")
    parser.add_argument("--concepts", type=str, default="dog,duck_toy,cat,backpack,teddybear,sofa,purse,dog1,wooden_pot,barn",
                        help="concepts")
    parser.add_argument("--name", type=str, default="evluate_all",
                        help="csv name")
    return parser.parse_args()


def main(args):
    #strArray=( "dog",)
    with open(args.sample_root+args.name+".csv",'w',encoding='UTF8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["tasks","CLIP Image alignment","CLIP Image alignment_50","CLIP text alignment","DINO Image alignment","Inception Score"])
    for file_path in args.concepts.split(','):
            sample_path=os.path.join(args.sample_root,file_path)
            print(sample_path)
            prompt_path=os.path.join(args.prompt_root,file_path)+".json"
            target_path=os.path.join(args.target_paths,file_path)
            scores=calmetrics(Path(sample_path),target_path,
               args.numgen, prompt_path,args.outpkl)
            with open(args.sample_root+args.name+".csv",mode='a',newline = '',encoding='utf-8_sig') as f:
                csv_writer = csv.writer(f)        
                csv_writer.writerow([file_path,scores['CLIP Image alignment'],scores['CLIP Image alignment_50'],scores['CLIP Text alignment'],scores['DINO Image alignment'],scores['Inception Score']])
            
if __name__ == "__main__":
    # distributed setting
    args = parse_args()
    main(args)
