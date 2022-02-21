from model import CTPN
from torch import nn, optim
from data import make_loader
import os
import torch as t
from loss import LossFunc
CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device_ids = list(range(len(CUDA_VISIBLE_DEVICES.split(","))))
img_size = (640, 640)
backbone_type = "resnet34"  # "resnet18" or "resnet34" or "resnet50"
epoch = 800
anchor_batch_size = 128  # anchor_batch_size / 2 positive anchor and anchor_batch_size / 2 negative anchor selected from one image
img_batch_size = 4  # image batch size
lr = 0.001
lr_de_rate = 0.1
patience = 50
weight_decay = 0.00001
lamda_1 = 1
lamda_2 = 2
print_step = 1
anchor_count = 10
negative_anchor_iou_thresh = 0.5
side_ref_dist_thresh = 16
num_workers = 2
train_side_ref = True
data_aug_level = "l"
train_img_dir = r"/home/yuyang/data/id_card/train_image"
train_label_dir = r"/home/yuyang/data/id_card/train_label"
valid_img_dir = r"/home/yuyang/data/id_card/train_image"
valid_label_dir = r"/home/yuyang/data/id_card/train_label"
best_valid_loss = float("inf")


def train_epoch(current_epoch, model, train_loader, criterion, optimizer):
    model.train()
    step = len(train_loader)
    current_step = 1
    for imgs, cls_labels, reg_labels, side_ref_labels in train_loader:
        imgs_cuda = imgs.cuda(0)
        rpn_cls, rpn_reg, side_ref = model(imgs_cuda)
        train_total_loss, train_cls_loss, train_reg_loss, train_side_ref_loss = criterion(rpn_cls, rpn_reg, side_ref, cls_labels, reg_labels, side_ref_labels)
        optimizer.zero_grad()
        train_total_loss.backward()
        optimizer.step()
        if current_step % print_step == 0:
            if train_side_ref:
                print("epoch:%d/%d, step:%d/%d, train_total_loss:%.5f, train_cls_loss:%.5f, train_reg_loss:%.5f, train_side_ref_loss:%.5f" % (current_epoch, epoch, current_step, step, train_total_loss.item(), train_cls_loss.item(), train_reg_loss.item(), train_side_ref_loss.item()))
            else:
                print("epoch:%d/%d, step:%d/%d, train_total_loss:%.5f, train_cls_loss:%.5f, train_reg_loss:%.5f" % (current_epoch, epoch, current_step, step, train_total_loss.item(), train_cls_loss.item(), train_reg_loss.item()))
        current_step += 1
    return model


def valid_epoch(current_epoch, model, criterion, valid_loader):
    global best_valid_loss
    model.eval()
    accum_total_loss = 0
    step = len(valid_loader)
    accum_cls_loss = 0
    accum_reg_loss = 0
    accum_side_ref_loss = 0
    for imgs, cls_labels, reg_labels, side_ref_labels in valid_loader:
        imgs_cuda = imgs.cuda(0)
        with t.no_grad():
            rpn_cls, rpn_reg, side_ref = model(imgs_cuda)
            valid_total_loss, valid_cls_loss, valid_reg_loss, valid_side_ref_loss = criterion(rpn_cls, rpn_reg, side_ref, cls_labels, reg_labels, side_ref_labels)
            accum_total_loss += valid_total_loss.item()
            accum_cls_loss += valid_cls_loss.item()
            accum_reg_loss += valid_reg_loss.item()
            accum_side_ref_loss += valid_side_ref_loss.item()
    print("saving epoch model......")
    t.save(model.state_dict(), "epoch.pth")
    avg_valid_total_loss = accum_total_loss / step
    avg_valid_cls_loss = accum_cls_loss / step
    avg_valid_reg_loss = accum_reg_loss / step
    avg_valid_side_ref_loss = accum_side_ref_loss / step
    if avg_valid_total_loss < best_valid_loss:
        print("saving best model......")
        best_valid_loss = avg_valid_total_loss
        t.save(model.state_dict(), "best.pth")
    print("###############valid epoch:%d################" % (current_epoch,))
    if train_side_ref:
        print("valid_total_loss:%.5f, valid_cls_loss:%.5f, valid_reg_loss:%.5f, valid_side_ref_loss:%.5f" % (avg_valid_total_loss, avg_valid_cls_loss, avg_valid_reg_loss, avg_valid_side_ref_loss))
    else:
        print("valid_total_loss:%.5f, valid_cls_loss:%.5f, valid_reg_loss:%.5f" % (avg_valid_total_loss, avg_valid_cls_loss, avg_valid_reg_loss))
    return model, avg_valid_total_loss


def main():
    model = CTPN(anchor_count, backbone_type)
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(0)
    criterion = LossFunc(lamda_1, lamda_2, train_side_ref).cuda(0)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_de_rate, patience=patience, verbose=True)
    for e in range(epoch):
        current_epoch = e + 1
        train_loader = make_loader(img_batch_size, anchor_batch_size, train_img_dir, train_label_dir, img_size, anchor_count, negative_anchor_iou_thresh, side_ref_dist_thresh, num_workers, True, data_aug_level)
        valid_loader = make_loader(img_batch_size, anchor_batch_size, valid_img_dir, valid_label_dir, img_size, anchor_count, negative_anchor_iou_thresh, side_ref_dist_thresh, num_workers, False, data_aug_level)
        model = train_epoch(current_epoch, model, train_loader, criterion, optimizer)
        model, val_loss = valid_epoch(current_epoch, model, criterion, valid_loader)
        lr_sch.step(val_loss)


if __name__ == "__main__":
    main()