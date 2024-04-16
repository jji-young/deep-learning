import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir): #데이터셋 초기화, 필요한 변환 반환

        self.data_dir = data_dir
        self.image_file_lst = os.listdir(data_dir)
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  
        ])
        

    def __len__(self): #데이터셋의 총 이미지 수 반환

        return len(self.image_file_lst)

    def __getitem__(self, idx): #주어진 인덱스에 해당하는 image, lable load + 변환적용
        img_path = os.path.join(self.data_dir, self.image_file_lst[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        label = int(self.image_files[idx].split('_')[1].split('.')[0])

        return img, label


if __name__ == '__main__':


    train_data = MNIST(data_dir='../data/train')
    test_data = MNIST(data_dir='../data/test')
    
    print(f"Number of training images: {len(train_data)}")
    print(f"Number of test images: {len(test_data)}")