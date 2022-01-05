import torch
from load_data_gcn import *
from resnet18_gcn import *
from ml_metrics import *
from config import *
arg = config.Config.config()
model_save_dir = arg['model_save_dir']
device = arg['device']
fold = arg['epoch']
data_loader = load_data(is_train=False)
model = RestNet18()
model = model.to(device)

min_ham = []
min_ham = 1
max_avg = []
max_avg = 0
min_err = []
min_err = 1
min_ran = []
min_ran = 1
min_cov = []
min_cov = 1
max_mac = []
max_mac = 0
max_mic = []
max_mic = 0
result = np.zeros((7, 30), dtype=float)
index = 0
for i in range(fold-30, fold):
    epoch_to_predict = i + 1
    all_list_new = []
    ham = []
    avg = []
    err = []
    ran = []
    cov = []
    mac = []
    mic = []
    for data, label in data_loader:
        data = data.to(device)
        label = label.to(device)
        model_state_path = os.path.join(model_save_dir,
                                        'fold' + str(fold) + '_' + 'epoch' + str(epoch_to_predict) + '.pth')
        # print(model_state_path)
        model.load_state_dict(torch.load(model_state_path))
        model.eval()
        outputs, outputs_gcn = model(data)
        output_final = 0.5 * outputs + 0.5 * outputs_gcn
        output_final = torch.sigmoid(output_final)
        outputs_new = output_final.cpu()
        outputs_new = outputs_new.detach().numpy()
        label_new = label.cpu()
        pre_labels = np.array(outputs_new > 0.52, dtype=np.int32)
        true_labels = np.array(label_new, dtype=np.int32)
        # print(true_labels)
        # print(pre_labels)
        metrics_result = all_metrics(outputs_new, pre_labels, true_labels)
        # print(metrics_result)

        ham.append(metrics_result[0][1])
        avg.append(metrics_result[1][1])
        err.append(metrics_result[2][1])
        ran.append(metrics_result[3][1])
        cov.append(metrics_result[4][1])
        mac.append(metrics_result[5][1])
        mic.append(metrics_result[6][1])
    ham_mean = np.mean(ham)
    avg_mean = np.mean(avg)
    err_mean = np.mean(err)
    ran_mean = np.mean(ran)
    cov_mean = np.mean(cov)
    mac_mean = np.mean(mac)
    mic_mean = np.mean(mic)
    result[0][index] = ham_mean
    result[1][index] = avg_mean
    result[2][index] = err_mean
    result[3][index] = ran_mean
    result[4][index] = cov_mean
    result[5][index] = mac_mean
    result[6][index] = mic_mean
    index = index + 1
    print('ham_mean', ham_mean, 'avg_mean', avg_mean, 'err_mean', err_mean, 'ran_mean', ran_mean,
          'cov_mean', cov_mean, 'mac_mean', mac_mean, 'mic_mean', mic_mean)
    if ham_mean < min_ham:
        min_ham = ham_mean

    if avg_mean > max_avg:
        max_avg = avg_mean

    if err_mean < min_err:
        min_err = err_mean

    if ran_mean < min_ran:
        min_ran = ran_mean

    if cov_mean < min_cov:
        min_cov = cov_mean

    if mac_mean > max_mac:
        max_mac = mac_mean

    if mic_mean > max_mic:
        max_mic = mic_mean

print('min_ham', min_ham, 'max_avg', max_avg, 'err_mean', err_mean, 'min_ran', min_ran,
      'min_cov', min_cov, 'max_mac', max_mac, 'max_mic', max_mic)
all_results = np.array(result)
np.save('resnet18_result_2', all_results)
print(all_results)
all_mean = np.mean(all_results, axis=1)
print("mean value")
np.save('resnet18_result_mean_2', all_mean)
print(all_mean)
all_std = np.std(all_results, axis=0, ddof=1)
np.save('resnet18_result_std_2', all_std)
print("std value")
print(all_std)
