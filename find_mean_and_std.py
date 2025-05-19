"""
find_mean_and_std loads the CIFAR-10 training dataset and calculates the mean and standard deviation of each colour channel.
"""

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)

pixels_per_channel = len(training_data) * 32*32
sum_values_per_channel = [0, 0, 0]
sum_squared_values_per_channel = [0, 0, 0]
for batch in train_dataloader:
    for image in batch[0]:
        sum_values_per_channel[0] += image[0].sum()
        sum_values_per_channel[1] += image[1].sum()
        sum_values_per_channel[2] += image[2].sum()
        sum_squared_values_per_channel[0] += image[0].pow(2).sum()
        sum_squared_values_per_channel[1] += image[1].pow(2).sum()
        sum_squared_values_per_channel[2] += image[2].pow(2).sum()
means = [float(x) / pixels_per_channel for x in sum_values_per_channel]
stdvs = [0, 0, 0]
for i in range(3):
    xsquared = sum_squared_values_per_channel[i]
    mean_xsquared = float(xsquared) / pixels_per_channel
    squared_meanx = means[i]**2
    stdvs[i] = (mean_xsquared - squared_meanx)**0.5

print(f"Means: {means}\nStdvs: {stdvs}")
