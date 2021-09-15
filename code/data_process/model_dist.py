# -- coding: utf-8
import torch

check_point = torch.load('../user_data/model_data/epoch_12.pth')
check_point_dist = check_point['state_dict']
torch.save(check_point_dist, '../user_data/model_data/epoch_12_dist.pth')

