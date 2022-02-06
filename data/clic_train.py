from .vision import VisionDataset
from PIL import Image
from glob import glob


class CLICTrain(VisionDataset):
    def __init__(self, root, transform):
        super(CLICTrain, self).__init__(root, None, transform, None)

        assert root[-1] == '/', "root to CLIC-train dataset should end with \'/\', not {}.".format(root)

        self.image_paths = sorted(glob(root + "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)
