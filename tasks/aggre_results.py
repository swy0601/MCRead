from os.path import join
import sys

sys.path.append('/')

from utils.file import load_json
from utils.task_argparse import read_args

args = read_args()

apca_run_log_base_dir = f'models/'
result_idx = 0

cv_accuracys = []
cv_precisions = []
cv_recalls = []
cv_f1s = []
cv_aucs = []
cv_mccs = []
test_models = []

cv_ver = './'
cv_results = load_json(join(apca_run_log_base_dir.format(subset=cv_ver), 'eval_results.json'))
cv_result_obj = cv_results[result_idx]
cv_accuracys.append(cv_result_obj['Accuracy'])
cv_precisions.append(cv_result_obj['Precision'])
cv_recalls.append(cv_result_obj['Recall'])
cv_f1s.append(cv_result_obj['F1-Score'])
cv_aucs.append(cv_result_obj['AUC'])
cv_mccs.append(cv_result_obj['MCC'])
test_models.append(cv_result_obj['test_model_name'])

print('\n' + '*'*50)
print(f'Cross-validation results(model={args.model}, dataset={args.dataset})')
print(f'Avg Accuracy: {sum(cv_accuracys) / len(cv_accuracys)}')
print(f'Avg Precision: {sum(cv_precisions) / len(cv_precisions)}')
print(f'Avg Recall: {sum(cv_recalls) / len(cv_recalls)}')
print(f'Avg F1: {sum(cv_f1s) / len(cv_f1s)}')
print(f'Avg AUC: {sum(cv_aucs) / len(cv_aucs)}')
print(f'Avg MCC: {sum(cv_mccs) / len(cv_mccs)}')
print('*'*50)
