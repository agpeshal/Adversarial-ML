import numpy as np
import torch
import argparse
import eagerpy as ep
import foolbox.attacks as fa
from foolbox import PyTorchModel, accuracy, samples
import foolbox as fb

from mnist_cnn import Net


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000,
                        help='Maximum number of steps to perform')
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

    if args.targeted:
        target_class = (labels + 7) % 10
        criterion = fb.criteria.TargetedMisclassification(target_class)
    else:
        criterion = fb.criteria.Misclassification(labels)

    attack = fa.L2DeepFoolAttack(steps=args.steps)
    epsilons = None
    raw, clipped, success = attack(fmodel, images, labels, epsilons=epsilons)

    robust_accuracy = 1 - success.float32().mean()
    print("Robust Accuracy", robust_accuracy.item())

    dist = np.mean(fb.distances.l2(clipped, images).numpy())
    print("Average perturbation norm", dist)

if __name__ == "__main__":
    main()