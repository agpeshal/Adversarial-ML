import numpy as np
import torch
import argparse
import eagerpy as ep
import foolbox.attacks as fa
from foolbox import PyTorchModel, accuracy, samples
import matplotlib.pyplot as plt

from mnist_cnn import Net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000,
                        help='Iteration of BA')
    parser.add_argument('--targeted', action='store', default=False,
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
    epsilons = np.linspace(0.01, 10, 20)

    boundary_attack = fa.BoundaryAttack(steps=args.steps, tensorboard=None)
    _, _, ba_success = boundary_attack(fmodel, images, labels,
                                   epsilons=epsilons)

    ba_robust_accuracy = 1 - ba_success.float32().mean(axis=-1)

    random_attack = fa.L2RepeatedAdditiveGaussianNoiseAttack(repeats=args.steps)
    _, _, ra_success = random_attack(fmodel, images, labels,
                              epsilons=epsilons)
    ra_robust_accuracy = 1 - ra_success.float32().mean(axis=-1)

    legends = ["Boundary Attack", "Random Attack"]
    plt.plot(epsilons, ba_robust_accuracy.numpy())
    plt.plot(epsilons, ra_robust_accuracy.numpy())
    plt.legend(legends, loc='upper right')
    plt.xlabel("Perturbation Norm (L2)")
    plt.ylabel("Robust Accuracy")
    plt.title("{} Queries".format(args.steps))
    plt.savefig('mnist_robust_acc.jpg')
    plt.show()

if __name__ == '__main__':
    main()
