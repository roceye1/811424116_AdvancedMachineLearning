# ============================================================
# FINAL MODEL-SPECIFIC EXACT IMAGE DUPLICATE AUDIT
# Paste into the END of the executed Kaggle notebook.
# Requires its earlier dataset cells (q1/q2/q3/train_12k/etc.)
# Generates Kaggle CSVs; DOES NOT modify images or scores.
# ============================================================

from pathlib import Path
import pandas as pd

hashes_now = pd.read_csv(ROOT / 'assignment2_image_hash_audit.csv')
assert {'filename', 'sha256', 'split', 'label'} <= set(hashes_now.columns)
assert hashes_now['filename'].is_unique
assert len(hashes_now) == 25000, 'Hash manifest does not cover the source dataset.'
lookup_hash = hashes_now.set_index('filename')['sha256']

def hashes_for(frame):
    names = frame['filename'].astype(str).tolist()
    missing = set(names) - set(lookup_hash.index)
    assert not missing, f'Missing SHA records: {sorted(missing)[:5]}'
    return lookup_hash.loc[names].tolist()

evaluation_frames = {
    'Original Validation': val_frame,
    'Original Test': test_frame,
    'Extended Validation': extra_val,
    'Holdout 1': holdout1,
    'Holdout 2': holdout2,
}

train_experiments = [
    ('Scratch 1K', q1, ['Original Validation','Original Test']),
    ('Scratch 1.5K', q2, ['Original Validation','Original Test']),
    ('Scratch 2K', q3, ['Original Validation','Original Test','Holdout 1']),
    ('Scratch 12K', train_12k, ['Original Validation','Original Test','Extended Validation','Holdout 1']),
    ('Scratch 18K', train_18k, ['Original Validation','Original Test','Extended Validation','Holdout 1']),
    ('Pretrained 1K', q1, ['Original Validation','Original Test']),
    ('Pretrained 1.5K', q2, ['Original Validation','Original Test']),
    ('Pretrained 2K', q3, ['Original Validation','Original Test','Holdout 1']),
    ('Dropout 0.0', q3, ['Original Validation','Holdout 2']),
    ('Dropout 0.25', q3, ['Original Validation','Holdout 2']),
    ('Dropout 0.5', q3, ['Original Validation','Holdout 2']),
]

rows = []
for model_name, train_selection, eval_sets in train_experiments:
    training_content = hashes_for(train_selection)
    train_unique = set(training_content)
    repeated = len(training_content) - len(train_unique)
    for eval_set_name in eval_sets:
        eval_content = set(hashes_for(evaluation_frames[eval_set_name]))
        overlaps = len(train_unique.intersection(eval_content))
        rows.append({
            'Model': model_name,
            'Evaluation Set': eval_set_name,
            'Training Images': len(train_selection),
            'Repeated Training Content': repeated,
            'Train Evaluation Hash Overlap': overlaps,
            'Status': 'PASS' if overlaps == 0 else 'REVIEW REQUIRED',
        })

model_audit = pd.DataFrame(rows)
assert len(model_audit) == 28

duplicate_groups = hashes_now[
    hashes_now.duplicated('sha256', keep=False)
].sort_values(['sha256','filename'])

eval_internal = []
for name, frame in evaluation_frames.items():
    content = hashes_for(frame)
    eval_internal.append({
        'Evaluation Set': name, 'Files': len(content),
        'Redundant Byte-Identical Files': len(content)-len(set(content))
    })

AUDIT_OUT = ROOT / 'final_integrity_audit'
AUDIT_OUT.mkdir(parents=True, exist_ok=True)
model_audit.to_csv(AUDIT_OUT / 'model_specific_hash_audit.csv', index=False)
duplicate_groups.to_csv(AUDIT_OUT / 'exact_duplicate_groups.csv', index=False)
pd.DataFrame(eval_internal).to_csv(AUDIT_OUT / 'evaluation_set_internal_duplicates.csv', index=False)

print('MODEL-SPECIFIC EXACT CONTENT AUDIT')
print(model_audit.to_string(index=False))
print('\nDUPLICATE GROUPS:', duplicate_groups.sha256.nunique())
print('FILENAMES IN DUPLICATE GROUPS:', len(duplicate_groups))
print('\nEVALUATION SET INTERNAL DUPLICATES')
print(pd.DataFrame(eval_internal).to_string(index=False))
print('\nFINAL AUDIT STATUS:')
if (model_audit['Train Evaluation Hash Overlap'] > 0).any():
    print('REVIEW REQUIRED: at least one evaluated model had byte-identical training/evaluation content.')
else:
    print('PASS: zero byte-identical training/evaluation overlap across all 28 tested pairs.')
print('Saved audit outputs to:', AUDIT_OUT)
