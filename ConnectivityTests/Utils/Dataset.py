# Various imports.
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset

# The batch size for the training and validation datasets.
BATCH_SIZE = 32


# This method is used to load the training, validation and test datasets.
# The code is lifted entirely from the flwr_datasets library - not my own.
def load_datasets():
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 1})

    # Normalises a batch of images.
    def apply_transforms(batch):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    # Create train/vals for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(1):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))
    testset = fds.load_full("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders[0], valloaders[0], testloader
