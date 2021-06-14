import os.path as osp
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
from utils.augmentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


def make_datapath_list(rootpath):
    # Get original template
    original_image_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
    # Get annotation template
    annotation_image_template = osp.join(rootpath, "SegmentationClass", "%s.png")

    # Train, val path
    train_ids = osp.join(rootpath, "ImageSets/Segmentation/train.txt")
    val_ids = osp.join(rootpath, "ImageSets/Segmentation/val.txt")

    # Create train image and annotation list
    train_img_list = list()
    train_annotation_list = list()
    """ 
    Create the lists to store URL link to image file and annotation file 
    """
    for line in open(train_ids):
        # IMPORTANT: dùng strip() để xóa bỏ khoảng trắng (whitespace) và các phần xuống dòng trong line
        # nếu không khi ghép các img_id vào sẽ bị lỗi
        img_id = line.strip()
        # Inject img_id to image_template (syntax C)
        img_path = original_image_template % img_id
        annotation_path = annotation_image_template % img_id

        # Set image and annotation path to list
        train_img_list.append(img_path)
        train_annotation_list.append(annotation_path)

    # Create val image and annotation list
    val_img_list = list()
    val_annotation_list = list()

    # Get each item in val path and put to list
    for line in open(val_ids):
        # IMPORTANT: dùng strip() để xóa bỏ khoảng trắng (whitespace) và các phần xuống dòng trong line
        # nếu không khi ghép các img_id vào sẽ bị lỗi
        img_id = line.strip()
        # Inject img_id to image_template (syntax C)
        img_path = original_image_template % img_id
        annotation_path = annotation_image_template % img_id

        # Set image and annotation path to list
        val_img_list.append(img_path)
        val_annotation_list.append(annotation_path)

    return train_img_list, train_annotation_list, val_img_list, val_annotation_list


class DataTransform:
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose(
                [
                    Scale(scale=[0.5, 1.5]),  # Scale image smaller or bigger
                    RandomRotation(angle=[-10, 10]),  # Random rotate image from -10 to 10
                    RandomMirror(),  # Random mirror or not
                    Resize(input_size),  # Reize image by image_size
                    Normalize_Tensor(color_mean, color_std),  # Normalize Tensor by color_mean and color_std
                ]
            ),
            "val": Compose(
                [
                    Resize(input_size),  # Reize image by image_size
                    Normalize_Tensor(color_mean, color_std),  # Normalize Tensor by color_mean and color_std
                ]
            ),
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)


"""
Handle dataset
"""


class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        ### Handle image ###
        # Original image
        img_file_path = self.img_list[index]
        img = Image.open(img_file_path)

        # Annotation image
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)  # PIL -> (height, width, channel(RGB))

        # Transform
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


""" MAIN """
if __name__ == "__main__":
    rootpath = "./data/VOCdevkit/VOC2012/"

    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(rootpath)

    # print(len(train_img_list))
    # print(len(val_img_list))

    # print(train_img_list[0])
    # print(train_annotation_list[0])

    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    # train dataset
    train_dataset = MyDataset(
        train_img_list,
        train_annotation_list,
        phase="train",
        transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std),
    )

    val_dataset = MyDataset(
        val_img_list,
        val_annotation_list,
        phase="val",
        transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std),
    )

    # print("val_dataset_img: {}".format(val_dataset.__getitem__(0)[0].shape))
    # print("val_dataset_anno_class_img: {}".format(val_dataset.__getitem__(0)[1].shape))
    # print("val_dataset: {}".format(val_dataset.__getitem__(0)))

    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

    batch_interator = iter(dataloader_dict["train"])

    images, anno_class_images = next(batch_interator)
    # print(images.size())
    # print(anno_class_images.size())

    ### TEST IMAGE AND ANNOTATION IMAGE ###
    # Permute the axes of an element the images list
    img = (
        images[0].numpy().transpose(1, 2, 0)
    )  # (channel(RGB), height, width) => (height, width, channel(RGB))
    plt.imshow(img)
    plt.show()

    anno_class_image = anno_class_images[0].numpy()
    plt.imshow(anno_class_image)
    plt.show()
