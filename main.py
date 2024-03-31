import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import dnnlib
import wandb

sys.path.append('./triplanenet')
sys.path.append('./facemesh')
from vive3D.visualizer import *
from vive3D.eg3d_generator import *
from vive3D.landmark_detector import *
from vive3D.video_tool import *
from vive3D.segmenter import *
from vive3D.inset_pipeline import *
from vive3D.aligner import *
from vive3D.interfaceGAN_editor import *
from vive3D.config import *
from talk3d_helper import TrainOptions
from trainer import Trainer


def _main(rank, world_size):
    opts = TrainOptions().parse()
    setup(rank, world_size,opts)
    main(opts, rank, world_size)

def setup(rank, world_size,opts):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opts.master_port
    distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def spawn_mp(running_fn, world_size):
    mp.spawn(running_fn,args=(world_size,),nprocs=world_size,join=True)

def main(opts, rank, world_size):
    focal_length = opts.focal_length
    camera_position = opts.camera_position
    use_tuned_G = opts.use_tuned_G
    wandb_save_path = f'{opts.saveroot_path}/wandb/{opts.personal_id}'
    os.makedirs(f'{wandb_save_path}/{opts.short_configs}', exist_ok=True)
    device = rank
    
    if opts.use_wandb:
        wandb.init(project='opts.wandb_project_name')
        wandb.run.name = f'{opts.short_configs}'
        wandb.config.update(opts.__dict__)
        wandb.run.save(wandb_save_path)
    
    
    print(f'*******************************************************************************')
    print(f'Loading Generator....')
    generator_path = os.path.join(opts.data_root_dir, opts.personal_id, opts.generator_dir)
    
    assert os.path.exists(generator_path), f'Generator is not available at {generator_path}, please check savepoint path'
    # construct_G toggle -> copying params from .pkl file
    # if you want to modify eg3d generator, this is necessary.
    generator = EG3D_Generator(generator_path, device, load_tuned=use_tuned_G, construct_G=True)
    generator.set_camera_parameters(focal_length=focal_length, cam_pivot=camera_position)
    generator.active_G.eval()
    
    print(f'*******************************************************************************')
    print(f'Setup Trainer...')

    trainer = Trainer(opts, generator, device, world_size)
        
    if opts.only_do_inference:
        print(f'*******************************************************************************')
        print("Start Inference")
        
        trainer.model2eval(trainer.tune_switch)
        with torch.no_grad():
            # trainer.inference_step(inferencetype='train', lip_smoothing=trainer.opts.lip_smoothing)
            if trainer.opts.do_inference_novel:
                trainer.inference_step(inferencetype='novel', cameratype=trainer.opts.inf_camera_type)
            if trainer.opts.do_inference_OOD:
                trainer.inference_step(inferencetype='OOD', cameratype=trainer.opts.inf_camera_type)
        
    else:
        print(f'*******************************************************************************')
        print('Start training')
        
        while True:
            trainer.train_epoch()
            if not trainer.keep_training:
                break
        
        print(f'*******************************************************************************')
        print('Training Finished')

        
if __name__ == '__main__':
    opts = TrainOptions().parse()
    n_gpus = int(opts.num_gpus)
    print('@@@@@@@@@@@@@@@@@@@@@')
    print('@  Training Talk3D  @')
    print('@@@@@@@@@@@@@@@@@@@@@')
    print(f'N_gpus: {n_gpus}')
    world_size = n_gpus
    torch.multiprocessing.set_start_method('spawn')
    spawn_mp(_main, world_size)  
