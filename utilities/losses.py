import torch
from torch import nn
from utilities.focal import FocalLoss


class PartialLoss(nn.Module):
    def __init__(self, criterion):
        super(PartialLoss, self).__init__()

        self.criterion = criterion
        self.nb_classes = self.criterion.nb_classes

    def forward(self, outputs, partial_target, phase='training'):
        nb_target = outputs.shape[0]
        loss_target = 0.0
        total = 0

        for i in range(nb_target):
            partial_i = partial_target[i,...].reshape(-1)
            outputs_i = outputs[i,...].reshape(self.nb_classes, -1).unsqueeze(0)
            outputs_i = outputs_i[:,:,partial_i<self.nb_classes]

            nb_annotated = outputs_i.shape[-1]
            if nb_annotated>0:
                outputs_i= outputs_i.reshape(1,self.nb_classes,1,1,nb_annotated) # Reshape to a 5D tensor
                partial_i = partial_i[partial_i<self.nb_classes].reshape(1,1,1,1,nb_annotated) # Reshape to a 5D tensor
                loss_target += self.criterion(outputs_i, partial_i.type(torch.cuda.IntTensor), phase)
                total+=1

        if total>0:
            return loss_target/total
        else:
            return 0.0      
          
            
class DC(nn.Module):
    def __init__(self,nb_classes):
        super(DC, self).__init__()
        
        self.softmax = nn.Softmax(1)
        self.nb_classes = nb_classes

    @staticmethod 
    def onehot(gt,shape):
        shp_y = gt.shape
        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, gt, 1)
        return y_onehot

    def reshape(self,output, target):
        batch_size = output.shape[0]

        if not all([i == j for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)

        target = target.permute(0,2,3,4,1)
        output = output.permute(0,2,3,4,1)
        print(target.shape,output.shape)
        return output, target


    def dice(self, output, target):
        output = self.softmax(output)
        if not all([i == j for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)

        sum_axis = list(range(2,len(target.shape)))

        s = (10e-20)
        intersect = torch.sum(output * target,sum_axis)
        dice = (2 * intersect) / (torch.sum(output,sum_axis) + torch.sum(target,sum_axis) + s)
        #dice shape is (batch_size, nb_classes)
        return 1.0 - dice.mean()  


    def forward(self, output, target):
        result = self.dice(output, target)
        return result


class DC_CE_Focal(DC):
    def __init__(self,nb_classes):
        super(DC_CE_Focal, self).__init__(nb_classes)

        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.fl = FocalLoss(reduction="none")

    def focal(self, pred, grnd, phase="training"):
        score = self.fl(pred, grnd).reshape(-1)

        if phase=="training": # class-balanced focal loss
            output = 0.0
            nb_classes = 0
            for cl in range(self.nb_classes):
                if (grnd==cl).sum().item()>0:
                    output+=score[grnd.reshape(-1)==cl].mean()
                    nb_classes+=1

            if nb_classes>0:
                return output/nb_classes
            else:
                return 0.0

        else:  # class-balanced focal loss
            return score.mean()

    def forward(self, output, target, phase="training"):
        # Dice term
        dc_loss = self.dice(output, target)

        # Focal term
        focal_loss = self.focal(output, target, phase)

        # Cross entropy
        output = output.permute(0,2,3,4,1).contiguous().view(-1,self.nb_classes)
        target = target.view(-1,).long().cuda()
        ce_loss = self.ce(output, target)

        result = ce_loss + dc_loss + focal_loss
        return result


