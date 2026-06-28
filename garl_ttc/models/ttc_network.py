import torch
import torch.nn as nn
import os

from .resnet import ResNetBackbone, resnet_spec
try:
    from colorama import Fore, Style
except ImportError:
    class _NoColor:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = RESET_ALL = ''
    Fore = Style = _NoColor()

WEIGHT_DICT = {
    'visible_height': 1, 
    'TTC': 1.0,
    'height_ratio': 10.0,
    'MiD': 1.0,
    'mask_t1_focal': 500.0,
    'mask_t2_focal': 500.0,
    }

from .focal_loss import focal_loss_forward

class TTCNetwork(nn.Module):

    def __init__(
        self,
        cfg, 
        is_train=True            
        ):
        super(TTCNetwork, self).__init__()
        num_layers = cfg['model']['num_layers']
        block_class, layers = resnet_spec[num_layers]
        
        self.fusion_style = cfg['model'].get('fusion_style', 'early_fusion')
        
        if self.fusion_style == 'early_fusion':
            self.backbone = ResNetBackbone(
                block_class, 
                layers, 
                cfg,            
                )
            self.middle_layer = nn.Linear(
                in_features=2048,
                out_features=cfg['model']['fc_features']
                )   
            self.with_decoder = self.backbone.with_decoder
        elif self.fusion_style == 'late_fusion':
            self.backbone_rgb = ResNetBackbone(
                block_class, 
                layers, 
                cfg,
                input_feat_num=cfg['model']['input_feat_num_rgb'],
                )
            self.with_decoder = self.backbone_rgb.with_decoder
            self.backbone_event = ResNetBackbone(
                block_class, 
                layers, 
                cfg,
                input_feat_num=cfg['model']['input_feat_num_event'],
                )       
            self.middle_layer = nn.Linear(
                in_features=2048 * 2,
                out_features=cfg['model']['fc_features']
                )
        else:
            raise NotImplementedError
            
        self.pool_layer = nn.AvgPool2d(kernel_size=4, stride=1)
        
        self.pred_mode = cfg['model']['mode']
        if cfg['model']['mode'] in ['baseline', 'height_ratio_direct']:
            self.final_layer = nn.Linear(
                in_features=cfg['model']['fc_features'],
                out_features=1
            )
        elif cfg['model']['mode'] == 'height_ratio':
            self.final_layer = nn.Linear(
                in_features=cfg['model']['fc_features'],
                out_features=cfg['model']['frame_num']
            ) 
            
        self.dT = cfg['dataset']['window_interval'] * 0.1
        print(Fore.GREEN + f'Model dT set to: {self.dT} s' + Style.RESET_ALL)
        

        if is_train:
            self.init_weights()
        
        self.load_pretrained_ckpt(cfg)
        
    def load_pretrained_ckpt(self, cfg):
        if self.fusion_style == 'late_fusion' and \
            'pretrained_ckpt_rgb' in cfg['model'] and \
            os.path.exists(cfg['model']['pretrained_ckpt_rgb']):
                ckpt = torch.load(cfg['model']['pretrained_ckpt_rgb'], map_location='cpu')
                weight_dict = {
                    key.replace('backbone.', ''): ckpt[key] 
                    for key in ckpt if key.startswith('backbone')
                    }
                self.backbone_rgb.load_state_dict(weight_dict, strict=False)
                print('Loaded ', cfg['model']['pretrained_ckpt_rgb'])
        if self.fusion_style == 'late_fusion' and \
            'pretrained_ckpt_event' in cfg['model'] and \
            os.path.exists(cfg['model']['pretrained_ckpt_event']):
                ckpt = torch.load(cfg['model']['pretrained_ckpt_event'], map_location='cpu')                
                weight_dict = {
                    key.replace('backbone.', ''): ckpt[key] 
                    for key in ckpt if key.startswith('backbone')
                    }
                self.backbone_event.load_state_dict(weight_dict, strict=False)  
                print('Loaded ', cfg['model']['pretrained_ckpt_event'])
        return
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)                
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    
    def forward(self, x):
        if self.fusion_style == 'early_fusion':
            x, mask_pred = self.backbone(x)
            x = self.pool_layer(x)
            x = x.reshape(len(x), -1)            
        elif self.fusion_style == 'late_fusion':
            x_rgb, _ = self.backbone_rgb(x[:,:6,:,:])
            x_rgb = self.pool_layer(x_rgb)
            x_rgb = x_rgb.reshape(len(x_rgb), -1)
            
            x_event, mask_pred_event = self.backbone_event(x[:,6:,:,:])
            x_event = self.pool_layer(x_event)
            x_event = x_event.reshape(len(x_event), -1)
            
            x = torch.cat([x_rgb, x_event], dim=1)
            mask_pred = mask_pred_event
        else:
            raise NotImplementedError            
                    
        x = self.middle_layer(x)
        out = self.final_layer(x)
        return out, mask_pred
    
    def forward_train(
            self, 
            data, 
            target,
            visible_height_target=None,
            mask_target=None,
            use_mask_supervison=None,
            epoch_idx=None
            ):
        regression_output, mask_pred = self(data)
        dT = self.dT
        
        loss_dict = {}
        print_dict = {}
        has_mask_supervision = use_mask_supervison is not None and sum(use_mask_supervison) > 0
        if self.with_decoder and mask_target is not None and has_mask_supervision:            
            valid_mask_pred_t1 = mask_pred[use_mask_supervison, :2, ...]
            valid_mask_gt_t1 = mask_target[use_mask_supervison, :1, ...]
            
            valid_mask_pred_t2 = mask_pred[use_mask_supervison, 2:, ...]
            valid_mask_gt_t2 = mask_target[use_mask_supervison, 1:, ...]

            loss_dict['mask_t1_focal'] = focal_loss_forward(
                valid_mask_pred_t1, 
                valid_mask_gt_t1[:,0,:,:]
                ) * WEIGHT_DICT['mask_t1_focal']
            
            loss_dict['mask_t2_focal'] = focal_loss_forward(
                valid_mask_pred_t2,
                valid_mask_gt_t2[:,0,:,:]
                ) * WEIGHT_DICT['mask_t2_focal']
            
            
            mask_shape = mask_target.shape
            if mask_shape[-1] == 128:
                print_dict['mask_super_resolution'] = torch.tensor(0)
            else:
                print_dict['mask_super_resolution'] = torch.tensor(1)
            
        if self.pred_mode == 'baseline':
            ttc_pred = regression_output[:,0]
            loss_dict['TTC'] = nn.functional.smooth_l1_loss(
                ttc_pred, 
                target
                ) * WEIGHT_DICT['TTC']

            print_dict['RTE'] = (
                torch.abs(ttc_pred - target) / torch.abs(target)).mean().detach() * 100

        elif self.pred_mode == 'height_ratio':
            height_ratio = regression_output[:,0] / regression_output[:,1]
            visible_height_ratio_target_from_gtttc = 1 - (dT / target)
            
            loss_dict['visible_height'] = nn.functional.smooth_l1_loss(
                regression_output, 
                visible_height_target
                ) * WEIGHT_DICT['visible_height']
            
            if epoch_idx is not None and epoch_idx > 5:
                loss_dict['MiD'] = torch.abs(
                    torch.log(height_ratio) - torch.log(visible_height_ratio_target_from_gtttc)
                    ).mean() * 1e4 * WEIGHT_DICT['MiD']
            
            print_dict['error_ratio'] = torch.abs(
                height_ratio - visible_height_ratio_target_from_gtttc).mean().detach()

            print_dict['MiD'] = torch.abs(
                torch.log(height_ratio) - torch.log(visible_height_ratio_target_from_gtttc)
                ).mean().detach() * 1e4

        elif self.pred_mode == 'height_ratio_direct':
            height_ratio = regression_output[:,0]
            visible_height_ratio_target_from_gtttc = 1 - (dT / target)
            
            loss_dict['height_ratio_direct'] = nn.functional.smooth_l1_loss(
                height_ratio, 
                visible_height_ratio_target_from_gtttc
                ) * WEIGHT_DICT['height_ratio']
            
            print_dict['error_ratio'] = torch.abs(
                height_ratio - visible_height_ratio_target_from_gtttc).mean()
            
        else:
            raise NotImplementedError
                
        return regression_output, mask_pred, loss_dict, print_dict
    
    def forward_test(self, data):
        return self.forward(data)
