import numpy as np
import torch
import argparse
import eagerpy as ep
import foolbox.attacks as fa
from foolbox import PyTorchModel, accuracy, samples
import foolbox as fb
import matplotlib.pyplot as plt

from mnist_cnn import Net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000,
                        help='Iteration of BA')
    parser.add_argument('--targeted', action='store-true', default=False,
                        help='For targeted attack')

    args = parser.parse_args()

    model = Net()
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    model.eval()

    preprocessing = dict(mean=0.1307, std=0.3081)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    fmodel = fmodel.transform_bounds((0, 1))
    assert fmodel.bounds == (0, 1)

    images, labels = ep.astensors(*samples(fmodel, dataset="mnist", batchsize=10))

    print('Model accuracy on clean examples: {}'.format(accuracy(fmodel, images, labels)))

    if args.targeted:
        target_class = (labels + 7) % 10
        criterion = fb.criteria.TargetedMisclassification(target_class)
    else:
        criterion = fb.criteria.Misclassification(labels)

    attack = fa.BoundaryAttack(steps=args.steps)
    epsilons = np.linspace(0.01, 10, 20)
    raw, clipped, success = attack(fmodel, images, labels, criterion=criterion,
                                   epsilons=epsilons)

    robust_accuracy = 1 - success.float32().mean(axis=-1)

    plt.plot(epsilons, robust_accuracy.numpy())
    plt.xlabel("Epsilons")
    plt.ylabel("Robust Accuracy")
    plt.savefig('mnist_BA_robust_acc.jpg')

    mean_distance = []
    for i in range(len(clipped)):
        dist = np.mean(fb.distances.l2(clipped[i], images).numpy())
        mean_distance.append(dist)

    plt.plot(epsilons, mean_distance)
    plt.xlabel('Epsilons')
    plt.ylabel('Mean L2 distance')
    plt.savefig("mnist_BA_mean_L2distance.jpg")


if __name__ == '__main__':
    main()
