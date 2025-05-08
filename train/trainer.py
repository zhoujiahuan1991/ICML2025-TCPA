# Desc 

import torch
from models import get_model
import time
from train.utils import cosine_lr, get_mixup_fn, device
import torch.optim as optim
import torch.nn as nn
import logging
import os
from torch.amp import autocast, GradScaler

from timm.loss import SoftTargetCrossEntropy
from data_utils import loader as data_loader




def train(args):
    """
    The training process
    """
    # 配置日志记录器
    # 名字为年月日时分秒+info
    logger_path = os.path.join(args.output_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.log')  
    logging.basicConfig(filename=logger_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    model = get_model(args)
    model.mode = 'train'
    
    # print("1")
    train_loader = data_loader.construct_train_loader(args)
    # print("2")
    
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.learnable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else: raise NotImplementedError()
    
    if args.scheduler == 'cosine':
        scheduler = cosine_lr(optimizer, args.lr, len(train_loader)*args.n_epochs//5, len(train_loader)*args.n_epochs)
    else: raise NotImplementedError()
    
    scaler = GradScaler()   

    if args.mixup == 'mixup':
        mixup_fn = get_mixup_fn(args)
        criterion = SoftTargetCrossEntropy()
    else: criterion = nn.CrossEntropyLoss()

    accs = []
    best_acc = 0.0
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        model.train()
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            if args.scheduler == 'cosine':
                global_step = len(train_loader) * epoch + i
                scheduler(global_step)
            
            # images, labels = data
            images, labels = data['image'], data['label']
            images = images.to(device)
            labels = labels.to(device)
            
            if args.mixup == 'mixup':
                # 如果images的数量为奇数,则删除最后一个
                if images.size(0) % 2 == 1:
                    images = images[:-1]
                    labels = labels[:-1]
                images, labels = mixup_fn(images, labels)  
            
            with autocast(device_type='cuda'):
                if epoch < args.random_epoch: random=True
                else: random=False
                if args.TDVP or args.RDVP: 
                    outputs, dist = model(images, random=random)  
                else:
                    dist = None
                    outputs = model(images)  


            loss = criterion(outputs, labels)
            # print(loss)
            # print(dist)
            # input()
            if dist is not None:
                loss += args.pool_loss_w * dist
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if args.mixup == 'mixup':
                ...
            else:
                correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            # print(f"Epoch {epoch+1}, Iteration {i}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Iteration {i}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
            logging.info(f"Epoch {epoch+1}, Iteration {i}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
        print(f"Epoch {epoch+1} Finished, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
        logging.critical(f"Epoch {epoch+1} Finished, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
        
        acc = eval(model, args, epoch=epoch)
        accs.append(acc)
          
        if acc > best_acc:
            best_acc = acc
            # best_model_path = os.path.join(args.output_path, 'best_model.pth')
            # torch.save(model.state_dict(), best_model_path)
            
        print(f"Accuracy: {best_acc:.4f}")
        logging.critical(f"Accuracy: {best_acc:.4f}")
        # print(f"Best accuracy: {best_acc:.4f}")
        # logging.critical(f"Best accuracy: {best_acc:.4f}")
        # 打印accs，并转化为4位小数
        accs_tmp = [round(acc, 4) for acc in accs]
        print(f"All accuracy: {accs_tmp}")
        logging.critical(f"All accuracy: {accs_tmp}")  
        # 将最好的准确率和所有准确率保存为txt文件, 并转化为4位小数
        best_acc_tmp = round(best_acc, 4)
        with open(os.path.join(args.output_path, 'accuracy.txt'), 'w') as f:
            f.write(f"Accuracy:\n {best_acc_tmp}\n")
            # f.write(f"Best accuracy:\n {best_acc_tmp}\n")
            # f.write(f"All accuracy:\n {accs_tmp}\n")
            
    best_acc = round(best_acc, 4)
    accs = [round(acc, 4) for acc in accs]
        
    print("Finished Training")
    logging.critical("Finished Training")
    # # 输出最好的模型的准确率和所有模型的准确率
    # print(f"Best accuracy: {best_acc}")
    # logging.critical(f"Best accuracy: {best_acc}")
    # print(f"All accuracy: {accs}")
    # logging.critical(f"All accuracy: {accs}")
    # # 将最好的准确率和所有准确率保存为txt文件
    # with open(os.path.join(args.output_path, 'accuracy.txt'), 'w') as f:
    #     f.write(f"Best accuracy:\n {best_acc}\n")
    #     f.write(f"All accuracy:\n {accs}\n")


    
def eval(model, args, epoch=0):
    """
    The evaluation process
    """
    model.mode = 'test'
    model.eval()
    test_loader = data_loader.construct_test_loader(args)
    
    # 测试模型在测试集上的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # images, labels = data
            images, labels = data['image'], data['label']
            images = images.to(device)
            labels = labels.to(device)
            if args.TDVP or args.RDVP:
                outputs, dist = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
            
    print(f"Accuracy of the network on the {total} test images:\n {acc:.4f}")
    logging.critical(f"Accuracy of the network on the {total} test images:\n {acc:.4f}")
    model.mode = 'train'
    return acc
            
    
    
