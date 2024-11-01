from tutorial_dataset import MasksDataset

train_images_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/images"
train_masks_dir = r"/Users/egorprokopov/Documents/Work/ITMO_ML/data/bubbles_split/test/masks"

dataset = MasksDataset(train_images_dir, train_masks_dir)
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
