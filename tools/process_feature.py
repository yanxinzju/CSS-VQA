import os
from tqdm import tqdm
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root='data/2018-04-27_bottom-up-attention_fixed_36'
new_root='data/rcnn_feature'
os.mkdir(new_root)
for file in tqdm(os.listdir(root)):
    old_name=os.path.join(root,file)
    file=int(file.split('_')[2].split('.')[0])
    new_name=os.path.join(new_root,str(file)+'.pth')
    shutil.copyfile(old_name,new_name)