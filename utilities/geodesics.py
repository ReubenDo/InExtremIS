import torch
import numpy as np
import dijkstra3d

def normalize(data):
    return (data-data.min())/(data.max()-data.min())

def generate_geodesics(extreme, img_gradient, prob, inside_bb_value=12, with_prob=True, with_euclidean=True):
    # 
    extreme = extreme.squeeze()
    img_gradient = img_gradient.squeeze()
    prob = prob.squeeze()
    
    if (extreme==inside_bb_value).sum()>0: # If the image patch is not only background
        # Corners of the bounding boxes
        bb_x_min = torch.where(extreme==inside_bb_value)[0].min().item()
        bb_x_max = torch.where(extreme==inside_bb_value)[0].max().item() + 1
        bb_y_min = torch.where(extreme==inside_bb_value)[1].min().item()
        bb_y_max = torch.where(extreme==inside_bb_value)[1].max().item() + 1
        bb_z_min = torch.where(extreme==inside_bb_value)[2].min().item()
        bb_z_max = torch.where(extreme==inside_bb_value)[2].max().item() + 1
        
        # Only paths within the non-relaxed bounding box are considered
        img_gradient_crop = img_gradient[bb_x_min:bb_x_max,bb_y_min:bb_y_max,bb_z_min:bb_z_max].cpu().numpy()
        prob_crop = prob[:,bb_x_min:bb_x_max,bb_y_min:bb_y_max,bb_z_min:bb_z_max].detach()
        prob_crop = torch.nn.Softmax(0)(prob_crop)[0,...].cpu().numpy() # Probability of the background
        
        # Extreme points
        ex_x_min = torch.where(extreme==1)
        ex_x_max = torch.where(extreme==2)
        ex_y_min = torch.where(extreme==3)
        ex_y_max = torch.where(extreme==4)
        ex_z_min = torch.where(extreme==5)
        ex_z_max = torch.where(extreme==6)
        
        # Identifying the pairs of extreme points to join (Extreme points may miss --> patch based approach)
        couples = []
        if ex_x_min[0].shape[0]>0 and ex_x_max[0].shape[0]>0: # Extreme points in the x dimension
            couples.append([[k[0].item() for k in ex_x_min], [k[0].item() for k in ex_x_max]])
        
        if ex_y_min[0].shape[0]>0 and ex_y_max[0].shape[0]>0: # Extreme points in the y dimension
            couples.append([[k[0].item() for k in ex_y_min], [k[0].item() for k in ex_y_max]])
        
        if ex_z_min[0].shape[0]>0 and ex_z_max[0].shape[0]>0: # Extreme points in the z dimension
            couples.append([[k[0].item() for k in ex_z_min], [k[0].item() for k in ex_z_max]])
        
        couples_crop = [[[k[0]-bb_x_min, k[1]-bb_y_min,k[2]-bb_z_min] for k in couple] for couple in couples]
        
        # Calculating the geodesics using the dijkstra3d
        output_crop = inside_bb_value + np.zeros(img_gradient_crop.shape)
        for source, target in couples_crop:
            weights = img_gradient_crop.copy() # Image gradient term
            
            if with_prob:
                weights+=prob_crop # Deep background probability term
            
            if with_euclidean: # Normalized distance map to the target
                x, y, z = np.ogrid[0:img_gradient_crop.shape[0], 0:img_gradient_crop.shape[1], 0:img_gradient_crop.shape[2]]
                distances = np.sqrt((x-target[0])**2+(y-target[1])**2+(z-target[2])**2)
                distances = normalize(distances)
                weights+=distances
            
            path = dijkstra3d.dijkstra(weights, source, target, connectivity=26)
            for k in path:
                x,y,z = k
                output_crop[x,y,z] = 1
            
        
        output = torch.zeros(extreme.shape)
        output[bb_x_min:bb_x_max,bb_y_min:bb_y_max,bb_z_min:bb_z_max] = torch.from_numpy(output_crop.astype(int))
        return output[None,None,...] #Adding batch and channel
    else:
        # No geodesics
        for k in range(1,7):
            extreme[extreme==k] = 1
        return extreme[None,None,...] #Adding batch and channel
