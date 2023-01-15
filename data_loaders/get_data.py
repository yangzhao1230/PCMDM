from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from teach.data.tools.collate import collate_pairs_and_text, collate_datastruct_and_text
from tqdm import tqdm 

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "babel":
        from data_loaders.multi_motion.data.dataset import BABEL
        return BABEL
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset

def get_dataset_loader(args, name, batch_size, num_frames, split='train', hml_mode='train'):
    if name != 'babel':
        dataset = get_dataset(name, num_frames, split, hml_mode)
        collate = get_collate_fn(name, hml_mode)
    else:
        from data_loaders.multi_motion.data.dataset import BABEL
        datapath = '../teach/data/babel/babel-smplh-30fps-male'
        framerate = 30
        dtype = 'separate_pairs'
        from teach.transforms.smpl import SMPLTransform
        from teach.transforms.joints2jfeats import Rifke
        from teach.transforms.rots2joints import SMPLH
        from teach.transforms.rots2rfeats import Globalvelandy
        rifke = Rifke(jointstype='mmm', forward_filter=False,
            path='../teach/deps/transforms/joints2jfeats/rifke/babel-amass',
            normalization=True
            )
        smplh = SMPLH(path='../teach/data/smpl_models/smplh',
            jointstype='mmm', input_pose_rep='matrix', batch_size=16, gender='male')
        globalvelandy = Globalvelandy(canonicalize=True, pose_rep='rot6d', offset=True,
            path='../teach/deps/transforms/rots2rfeats/globalvelandy/rot6d/babel-amass',
            # path='../teach/deps/transforms/rots2rfeats/globalvelandy/rot6d/babel-amass/separate_pairs',
            normalization=True)
        transforms = SMPLTransform(rots2rfeats=globalvelandy, rots2joints=smplh,
            joints2jfeats=rifke)
        dataset = BABEL(datapath=datapath, framerate=framerate, dtype=dtype, transforms=transforms, tiny=args.tiny)
        datatype = 'separate_pairs'
        if datatype == 'separate_pairs':
            collate = collate_pairs_and_text
        else:
            collate = collate_datastruct_and_text

    # collate = get_collate_fn(name, hml_mode)

    # batch_size = 1
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )
    # print(len(loader))
    # for batch in tqdm(loader):
    #     print(batch)
    return loader