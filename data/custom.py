from glob import glob

from PIL import Image

from .vision import VisionDataset


class CustomData(VisionDataset):
    def __init__(self, root, transform, img_ext="*.png"):
        super(CustomData, self).__init__(root, None, transform, None)

        assert root[-1] == '/', "root to test dataset should end with \'/\', not {}.".format(root)

        self.image_paths = sorted(glob(root + img_ext))
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

        return img, img_path

    def __len__(self):
        return len(self.image_paths)
