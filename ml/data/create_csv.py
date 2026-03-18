import os, csv, pathlib

DATA_ROOT = 'ml/data/raw'
OUTPUT_DIR = 'ml/data/processed'

def create_csv(split):
    rows = []
    for label, cls in enumerate(['NORMAL', 'PNEUMONIA']):
        folder = pathlib.Path(f'{DATA_ROOT}/{split}/{cls}')
        for img_path in folder.glob('*.jpeg'):
            rows.append({'filepath': str(img_path).replace('\\', '/'), 'label': label, 'class': cls})
    out_path = f'{OUTPUT_DIR}/{split}.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath','label','class'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Created {out_path} with {len(rows)} rows')

for s in ['train', 'val', 'test']:
    create_csv(s)
