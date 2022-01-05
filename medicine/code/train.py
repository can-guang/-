from load_data_gcn import *
from resnet18_gcn import RestNet18
import torch
import math
from tensorboardX import *
from ml_metrics import *
import os
import config
from common_loss import My_logit_ML_loss
from common_loss import My_KL_loss
np.set_printoptions(threshold=np.inf)
arg = config.Config.config()
fold = arg['epoch']


def train():
    dtype = arg['type']
    fold = arg['epoch']
    device = arg['device']
    model_save_dir = arg['model_save_dir']
    model_save_epoch = arg['model_save_epoch']
    data_loader = load_data(arg['is_train'])
    writer = SummaryWriter(logdir=arg['log_dir'])
    model = RestNet18()
    model = model.to(device)
    optimizer_orig = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    for epoch in range(fold):
        model.train()
        running_loss = 0.0
        running_loss_gcn = 0.0
        print(epoch)
        for data, label in data_loader:
            data = data.type(dtype).to(device)
            label = label.type(dtype).to(device)
            outputs, outputs_gcn = model(data)
            # print(outputs)
            running_loss = My_logit_ML_loss(outputs, label)
            running_loss_gcn = My_logit_ML_loss(outputs_gcn, label)
            out_final = 0.5 * outputs + 0.5 * outputs_gcn
            loss = 0.5 * running_loss + 0.5 * running_loss_gcn
            # print(loss)
            optimizer_orig.zero_grad()
            loss.backward()
            optimizer_orig.step()
            out_final = torch.sigmoid(out_final)
            outputs_new = out_final.cpu()
            outputs_new = outputs_new.detach().numpy()
            label_new = label.cpu()
            pre_labels = np.array(outputs_new > 0.52, dtype=np.int32)
            true_labels = np.array(label_new, dtype=np.int32)
            metrics_result = all_metrics(outputs_new, pre_labels, true_labels)
            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
            print(metrics_result)
            # min_ham = metrics_result[0][1]
            # max_avg = metrics_result[1][1]
            # min_err = metrics_result[2][1]
            # min_ran = metrics_result[3][1]
            # min_cov = metrics_result[4][1]
            # max_mac = metrics_result[5][1]
            # max_mic = metrics_result[6][1]

        print(epoch, loss)
        if (epoch + 1) % model_save_epoch == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir,
                                    'fold' + str(fold) + '_' + 'epoch' + str(epoch + 1) + '.pth'))
        writer.add_scalar("train_loss_paral", loss, epoch)
    print("Finished Training")
