import numpy as np
import pandas as pd

# Step 1: Load EVA stacked predictions
test_df = pd.read_csv('data/Test.csv')
fold_preds = []
for fold in range(5):
    preds = np.load(f'output/test_preds_fold{fold}.npy')
    fold_preds.append(preds)
files = np.mean(fold_preds, axis=0)
if files.ndim > 1:
    probs = files[:, 1]
else:
    probs = files
ids = test_df['ID'].values
ensemble_sub = pd.DataFrame({'ID': ids, 'Prob_Class_1': probs})
ensemble_sub.to_csv('output/ensemble_sub_probs.csv', index=False)

# Step 2: Load other model outputs
sub1 = pd.read_csv('output/yolo_submission_probs_labels.csv')
sub2 = pd.read_csv('output/efficientnet_submission_probs.csv')
sub3 = pd.read_csv('output/lgbm_submission_probs.csv')
sub1 = sub1[['Image_ID','Prob_Class_1']]
common_ids = sub1['Image_ID'].isin(sub2['ID'])
sub1_common = sub1[common_ids]
sub1_common = sub1_common.rename(columns={'Image_ID': 'ID', 'Prob_Class_1': 'Prob1'})
sub2 = sub2.rename(columns={'probability': 'Prob2'})

# Step 3: First blend
sub = pd.merge(sub1_common, sub2, on='ID', how='inner')
sub['Prob_Class_1'] = 0.57 * sub['Prob1'] + 0.43 * sub['Prob2']
probs_sub = sub[['ID', 'Prob_Class_1']]

# Step 4: Second blend with LGBM
sub = probs_sub.rename(columns={'Prob_Class_1': 'Prob1'})
sub3 = sub3.rename(columns={'Probs': 'Prob3'})
final_sub = pd.merge(sub, sub3, on='ID', how='inner')
final_sub['Prob_Class_1'] = (0.6 * final_sub['Prob1'] + 0.4 * final_sub['Prob3'])
final_sub = final_sub[['ID', 'Prob_Class_1']]

# Step 5: Third blend with EVA
sub = ensemble_sub.rename(columns={'Prob_Class_1': 'Prob1'})
sub3 = sub3.rename(columns={'Probs': 'Prob3'})
final_sub2 = pd.merge(sub, sub3, on='ID', how='inner')
final_sub2['Prob_Class_1'] = (0.6 * final_sub2['Prob1'] + 0.4 * final_sub2['Prob3'])
final_sub2 = final_sub2[['ID', 'Prob_Class_1']]

# Step 6: Final merge & weighted average
sub = pd.merge(final_sub, final_sub2, on='ID', how='inner')
w = 0.45
sub['final_probs'] = (sub['Prob_Class_1_x'] * w) + (sub['Prob_Class_1_y'] * (1 - w))
sub['Target'] = (sub['final_probs'] > 0.52).astype(int)
sub = sub[['ID', 'Target']]

print(sub.Target.value_counts())
sub.to_csv('output/sub_last_ensemble.csv', index=False)
print("✅ Final ensemble saved as sub_last_ensemble.csv")
