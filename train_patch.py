"""
Training code for Adversarial patch training

patch read/generate后为浮点小数形式, 原图Image.open后ToTensor,最终也为浮点小数形式
"""
import os
import sys
import PIL
import time
import torch
import wandb
import warnings
from tqdm import tqdm

# import load_data
# from load_data import *
import ld
from ld import *
import patch_config
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

warnings.filterwarnings("ignore")

'''
cd autodl-tmp/AP_plane
cd AP_plane
python train_patch.py yolov5s
'''

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.model = self.config.model.eval().cuda()
        # print(self.model)
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = self.config.prob_extractor.cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = 640
        batch_size = self.config.batch_size  # 1
        n_epochs = 500
        max_lab = self.config.max_det

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        # adv_patch_cpu = self.read_image("saved_patches/patchnew2.jpg")

        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)  # 仅优化adv_patch_cpu，不优化网络参数
        scheduler = self.config.scheduler_factory(optimizer)

        wandb.init(project="Adversarial-attack-plane")
        wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
        wandb.watch(self.model, log="all")
        
        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.set_detect_anomaly(True):  # 运行前向时开启异常检测功能，则在反向时会打印引起反向失败的前向操作堆栈，反向计算出现“nan”时引发异常
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))  # 上/下采样到所设图片大小

                    # img = p_img_batch[1, :, :,]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()

                    # output = self.darknet_model(p_img_batch)
                    output = self.model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.5
                    det_loss = torch.mean(max_prob)
                    # det_loss = torch.where(torch.isnan(det_loss), torch.full_like(det_loss, 0.1, requires_grad=True), det_loss).cuda()
                    # print(det_loss)
                    if(torch.isnan(det_loss).any()):
                        continue
                    
                    # det_loss = max_prob.clone()
                    max_tv_loss = torch.max(tv_loss, torch.tensor(0.1).cuda())  # 最小为0.1
                    loss = det_loss + nps_loss + max_tv_loss
                    # loss = nps_loss + max_tv_loss
                    
                    ep_det_loss += det_loss.detach().cpu().numpy()
                    # ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += max_tv_loss.detach().cpu().numpy()
                    ep_loss += loss
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    bt1 = time.time()
                    if i_batch % 50 == 0:  # 每50个batch记录一次
                        iteration = self.epoch_length * epoch + i_batch

                        wandb.log({
                            "Patches": wandb.Image(adv_patch_cpu, caption="patch{}".format(iteration)),
                            "tv_loss": tv_loss,
                            "nps_loss": nps_loss,
                            "det_loss": det_loss,
                            "total_loss": loss,
                        })
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(train_loader):
                        img = p_img_batch[0, :, :,]
                        img = transforms.ToPILImage()(img.detach().cpu())
                        img.save('pics2/epoch_{}.jpg'.format(epoch + 1))
                    del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_nps_loss = ep_nps_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss.item())
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                #plt.imshow(im)
                #plt.show()
                #im.save("saved_patches/patchnew1.jpg")
                # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()