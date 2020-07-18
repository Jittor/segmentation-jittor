import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from models.danet import DANet # danet
from models.deeplab import DeepLab # deeplab v3 +
from models.pspnet import PSPNet # pspnet
from models.ann import ANNNet
from models.ocnet import OCNet 
from models.ocrnet import OCRNet
from data.voc import TrainDataset, ValDataset
import time
import numpy as np
from utils.utils import Evaluator
import settings
from tensorboardX import SummaryWriter

jt.flags.use_cuda = 1

def poly_lr_scheduler(opt, init_lr, iter, epoch, max_iter, max_epoch):
    new_lr = init_lr * (1 - float(epoch * max_iter + iter) / (max_epoch * max_iter)) ** 0.9
    l = len(opt.param_groups)
    opt.param_groups[0]['lr'] = new_lr 
    for i in range(1, l):
        opt.param_groups[i]['lr'] = new_lr * 10

def get_model():
    if settings.MODEL_NAME == 'deeplab':
        model = DeepLab(output_stride=settings.STRIDE, num_classes=settings.NCLASS)
    elif settings.MODEL_NAME == 'pspnet':
        model = PSPNet(output_stride=settings.STRIDE, num_classes=settings.NCLASS)
    elif settings.MODEL_NAME == 'ann':
        model = ANNNet (output_stride=settings.STRIDE, num_classes=settings.NCLASS)
    elif settings.MODEL_NAME == 'ocnet':
        model = OCNet (output_stride=settings.STRIDE, num_classes=settings.NCLASS)
    elif settings.MODEL_NAME == 'danet':
        model = DANet (output_stride=settings.STRIDE, num_classes=settings.NCLASS)
    elif settings.MODEL_NAME == 'ocrnet':
        model = OCRNet(output_stride=settings.STRIDE, num_classes=settings.NCLASS)
    return model

def train(model, train_loader, optimizer, epoch, init_lr, writer):
    model.train()
    max_iter = len(train_loader)

    for idx, (image, target) in enumerate(train_loader):
        poly_lr_scheduler(optimizer, init_lr, idx, epoch, max_iter, settings.EPOCHS)
        image = image.float32()
        jt.sync_all()
        start_time = time.time()
        context, pred = model(image) 
        
        loss = model.get_loss(target, pred, context, settings.IGNORE_INDEX)
        optimizer.step (loss)
        jt.sync_all()
        end_time = time.time()
        print ('total time =', end_time - start_time)
        writer.add_scalar('train/total_loss_iter', loss.data, idx + max_iter * epoch)
        print ('Training in epoch {} iteration {} loss = {}'.format(epoch, idx, loss.data[0]))


best_miou = 0.0
def val (model, val_loader, epoch, evaluator, writer):
    model.eval()
    evaluator.reset()
    avg_time = 0.0 
    total_time = 0.0
    for idx, (image, target) in enumerate(val_loader):
        image = image.float32()
        #print (image.shape)
        start_time = time.time()
        output = model(image)
        end_time =  time.time()
        total_time = total_time + end_time - start_time 
        #print ('val time =', end_time - start_time)
        target = target.data
        pred = output.data
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
        #print ('Eval at epoch {} iteration {}'.format(epoch, idx))
        #print (jt.display_memory_info())
    avg_time = total_time / idx 
    print ('eval avg_time =', avg_time)
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    writer.add_scalar('val/mIoU', mIoU, epoch)
    writer.add_scalar('val/Acc', Acc, epoch)
    writer.add_scalar('val/Acc_class', Acc_class, epoch)
    writer.add_scalar('val/fwIoU', FWIoU, epoch)
    global best_miou
    if (mIoU > best_miou):
        best_miou = mIoU
    if mIoU > 75.0:
        save_path = settings.SAVE_MODEL_PATH + "_" + (str)(mIoU) + '.pkl'
        print ('save checkpoint at ', save_path)
        model.save(save_path)
    print ('Testing result of epoch {} miou = {} Acc = {} Acc_class = {} \
                FWIoU = {} Best Miou = {}'.format(epoch, mIoU, Acc, Acc_class, FWIoU, best_miou))


def main():
    jt.seed(settings.SEED)
    np.random.seed(settings.SEED)
    model = get_model()
    train_loader = TrainDataset(data_root=settings.DATA_ROOT, split='train', batch_size=settings.BATCH_SIZE, shuffle=True)
    val_loader = ValDataset(data_root=settings.DATA_ROOT, split='val', batch_size=1, shuffle=False)
    writer = SummaryWriter(settings.WRITER_PATH)
    learning_rate = settings.LEARNING_RATE
    momentum = settings.MOMENTUM
    weight_decay = settings.WEIGHT_DECAY
    
    model_backbone = []
    model_backbone.append(model.get_backbone())
    model_head = model.get_head()
    params_list = []
    for module in model_backbone:
        params_list.append(dict(params=module.parameters(), lr=learning_rate))
    for module in model_head:
        for m in module.modules():
            print (type(m).__name__, type(m))
        params_list.append(dict(params=module.parameters(), lr=learning_rate * 10))


    optimizer = nn.SGD(params_list, learning_rate, momentum, weight_decay)
    epochs = settings.EPOCHS
    evaluator = Evaluator(settings.NCLASS)
    for epoch in range (epochs):
        #train(model, train_loader, optimizer, epoch, learning_rate, writer)
        val(model, val_loader, epoch, evaluator, writer)

if __name__ == '__main__' :
    main ()
    jt.flags.use_cuda = 0

