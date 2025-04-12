import torch

# Replace with the path to your .pth file
pth_path = 'checkpoints/CelebA128_cd/teacher_resume60-80/teacher_60_target.pth'

# Load the checkpoint dictionary from the file
checkpoint = torch.load(pth_path, map_location='cpu')

# Print all keys stored in the checkpoint
print("Keys in .pth file:")
for key in checkpoint.keys():
    print(f"- {key}")
    if key == 'epoch':
        print(f"epoch:{checkpoint[key]}")