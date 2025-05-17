import os
import shutil

# Paths
train_txt_path = 'train.txt'  # Your train.txt file
source_folder = '/shared/chamika2/wacodataset/csr_train2'           # Where all .txt matrix data files live
target_folder = '/home/chamika2/waco-extend/dataset'             # Where you want to copy the relevant files
output_list_path = './filtered_train.txt'

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Read matrix names from train.txt
with open(train_txt_path, 'r') as f:
    matrix_names = [line.strip() for line in f]

# Copy matching files
copied_names = []
copied = 0
for name in matrix_names:
    src_file = os.path.join(source_folder, f'{name}.csr')
    dst_file = os.path.join(target_folder, f'{name}.csr')
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)
        copied_names.append(name)
        copied += 1
    else:
        print(f'[Warning] File not found: {src_file}')

print(f'Done. Copied {copied} files to {target_folder}.')

# Write copied matrix names to filtered_train.txt
with open(output_list_path, 'w') as f:
    for name in copied_names:
        f.write(name + '\n')

print(f'Done. Copied {len(copied_names)} files to {target_folder} and saved names to {output_list_path}.')