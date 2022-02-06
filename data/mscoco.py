from .vision import VisionDataset
from PIL import Image
from glob import glob


class MSCOCO(VisionDataset):
    """`MS Coco <http://mscoco.org/dataset>`_ Dataset.
        Args:
            root (string): Root directory where images are downloaded to.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``

        Example:
            .. code:: python
                import torchvision.datasets as dset
                import torchvision.transforms as transforms
                cap = dset.CocoCaptions(root = 'dir where images are',
                                        transform=transforms.ToTensor())
                print('Number of samples: ', len(cap))
                img, target = cap[3] # load 4th sample
                print("Image Size: ", img.size())
                print(target)
            Output: ::
                Number of samples: 82783
                Image Size: (3L, 427L, 640L)
                [u'A plane emitting smoke stream flying over a mountain.',
                u'A plane darts across a bright blue sky behind a mountain covered in snow',
                u'A plane leaves a contrail above the snowy mountain top.',
                u'A mountain that has a plane flying overheard in the distance.',
                u'A mountain view with a plume of smoke in the background']
    """

    def __init__(self, root, transform):
        super(MSCOCO, self).__init__(root, None, transform, None)

        assert root[-1] == '/', "root to COCO dataset should end with \'/\', not {}.".format(root)

        self.image_paths = sorted(glob(root + "*.jpg"))
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
