import numpy as np
import torch
import torchvision
import argparse
from torchvision.utils import save_image
from torchvision import transforms, datasets
import os
from foolbox import PyTorchModel, samples
import foolbox as fb
import matplotlib.pyplot as plt

from mnist_cnn import Net

np.random.seed(1)
torch.cuda.manual_seed(1)
torch.manual_seed(1)


def get_perturbed_samples(base_image, size, std, device):
    """
    :param device: GPU or CPU
    :param base_image: Input image
    :param size: Number of perturbed samples
    :param std: Control norm of the perturbation
    :return: a batch of images with batch size = size
    """

    c, h, w = base_image.shape

    dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([std]))
    perturbations = dist.rsample((size, c, h, w))
    perturbations.squeeze_(-1)
    base_image = base_image.unsqueeze(0)
    perturbations = perturbations.to(device)
    perturbed_samples = base_image.reshape((1, c, h, w)).add(perturbations)
    # perturbed_samples = torch.clamp(base_image.reshape((1, c, h, w)).add(perturbations), 0, 1)
    #
    # normalize = transforms.Compose(
    #     [transforms.ToPILImage(), transforms.ToTensor(),
    #      transforms.Normalize((0.1307,), (0.3081,))])
    # perturbed_samples = base_image.reshape((1, c, h, w)).add(perturbations)
    # perturbed_samples = [normalize(x) for x in perturbed_samples]
    # perturbed_samples = torch.clamp(perturbed_samples, 0, 1)

    return perturbed_samples.float()


def load_ImageNet():
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('../data/ImageNet',
                             transform=transforms.Compose({
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             })),
        batch_size=100, shuffle=True)
    return test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mnist",
                        help='Dataset Name')
    parser.add_argument('--std', type=float, default=0.25,
                        help='To control the norm of perturbation')
    parser.add_argument('--steps', type=int, default=1e5,
                        help='The number of calls made to the model')
    parser.add_argument('--save_count', type=int, default=10,
                        help='Number of adversarial images to be saved')

    args = parser.parse_args()

    path = os.path.join("./Results", args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    ## Download imagent data and set the correct Path

    if args.dataset == "imagenet":
        model = torchvision.models.resnet18(pretrained=True)
        test_loader = load_ImageNet()

    elif args.dataset == "mnist":

        # Load pretrained CNN on MNIST

        model = Net()
        model.load_state_dict(torch.load('mnist_cnn.pt', map_location=device))
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=100, shuffle=True)
        model = model.to(device)

    else:
        raise ValueError(
            f"Dataset {args.dataset} not available"
        )
    model = model.to(device)
    model = model.eval()

    # Loading Test data

    successful = 0
    total = 0
    steps = args.steps

    while True:

        # Need data in proper format to use PyTorch loader
        # instead using foolbox!
        if args.dataset == "imagenet":
            preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
            bounds = (0, 1)
            fmodel = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
            fmodel = fmodel.transform_bounds((0, 1))
            assert fmodel.bounds == (0, 1)
            images, labels = samples(fmodel, dataset='imagenet', batchsize=20)
            batch = 500  # number of random perturbations in each iteration
        else:
            examples = iter(test_loader)
            images, labels = examples.next()
            batch = 10000  # number of random perturbations in each iteration

        iterations = int(np.ceil(steps / batch)) + 1

        images = images.to(device)
        labels = labels.to(device)

        # no more test images
        if not labels.size:
            break

        total += len(labels)

        for image, label in zip(images, labels):
            output = model(image.unsqueeze(0))
            if output.argmax() == label:
                base_image = torch.clamp(image, 0, 1)
                base_label = label

                for iteration in range(1, iterations):

                    perturbed_samples = get_perturbed_samples(base_image, batch, args.std, device)

                    prediction = model(perturbed_samples).argmax(dim=1)
                    success = (False == prediction.eq(base_label)).nonzero()  # Indexes of all incorrect predictions

                    if success.nelement():
                        successful += 1
                        print("Success rate so far :{}/{}".format(successful, total))

                        if args.save_count:
                            index = success[0].item()
                            print("Norm of image", torch.norm(base_image))
                            print("Norm of added noise", torch.norm(perturbed_samples[index] - base_image))

                            adversarial_image = perturbed_samples[index].to("cpu")
                            if adversarial_image.shape[0] == 1:
                                plt.imshow(adversarial_image[0], cmap='gray')
                                plt.show()
                            else:
                                plt.imshow(adversarial_image.permute(1, 2, 0))
                                plt.show()

                            # rescale image before saving
                            resize = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(size=200),
                                transforms.ToTensor()
                            ])
                            adversarial_image = resize(adversarial_image)
                            save_image(adversarial_image, os.path.join(path, str(args.save_count) + ".png"), padding=0)
                            args.save_count -= 1

                        break

    print("Accuracy on perturbed samples", 100.0 * successful / total)


if __name__ == "__main__":
    main()
