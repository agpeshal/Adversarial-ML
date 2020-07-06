import numpy as np
import torch
import argparse
import eagerpy as ep
import foolbox.attacks as fa
from foolbox import PyTorchModel
import matplotlib.pyplot as plt
from time import time
from mnist_cnn import Net
from torchvision import transforms, datasets

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_norm', type=float, default=0.01,
                        help='Minimum perturbation norm')
    parser.add_argument('--max_norm', type=float, default=15,
                        help='Maximum perturbation norm')
    parser.add_argument('--num', type=int, default=12,
                        help='Number of norms to evaluate on')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For Colab
    # class args:
    #     # repeats = 1000
    #     min_norm = 0.01
    #     max_norm = 15
    #     num = 12

    # List of max query count
    queries = [10, 100, 1000, 5000]

    # Load the pretrained model
    model = Net()
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    model.eval()

    # preprocess the model inputs and set pixel bound in [0, 1]
    preprocessing = dict(mean=0.1307, std=0.3081)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    fmodel = fmodel.transform_bounds((0, 1))
    assert fmodel.bounds == (0, 1)

    # Set the perturbation norm space
    epsilons = np.linspace(args.min_norm, args.max_norm, args.num)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=10000, shuffle=True)

    total = 0                                          # total input count
    successful = torch.zeros(args.num, device=device)  # success for each norm
    legends = []
    start = time()
    for query in queries:
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            ep.astensor_(images)
            ep.astensor_(labels)

            # Additive Gaussian noise attack with L2 norm
            attack = fa.L2RepeatedAdditiveGaussianNoiseAttack(repeats=query)

            raw, clipped, success = attack(fmodel, images, labels,
                                           epsilons=epsilons)

            # Add the total number of successful attacks for each norm value
            successful += success.sum(axis=1)
            total += len(labels)

        robust_accuracy = (1 - 1.0 * successful / total).cpu()
        plt.plot(epsilons, robust_accuracy.numpy())
        legends.append("{} Queries".format(query))

    plt.xlabel("Perturbation Norm (L2)")
    plt.ylabel("Robust Accuracy")
    plt.title("Gaussian Noise")
    plt.legend(legends, loc='upper right')
    plt.ylim([0, 1])
    plt.savefig('mnist_RA_robust_acc.jpg')
    plt.show()

    end = time()

    print("Time taken: {:.1f} minutes".format((end - start) / 60))

if __name__ == '__main__':
    main()