import numpy as np
import argparse
import pandas as pd
import matplotlib.image as mpimg
from torch.autograd import Variable
import os

# import h5py
import os
# from PIL import Image
# import PIL

import matplotlib.pyplot as plt
# import csv
from DatasetGenerator import MultiDirectoryDataSequence
import time
import sys
sys.path.append("../models")
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v1, DAVE2v2, DAVE2v3, Epoch

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='parent directory of training dataset')
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--robustification", type=bool, default=False)
    parser.add_argument("--noisevar", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()
    return args


def characterize_steering_distribution(y_steering, generator):
    turning = []; straight = []
    for i in y_steering:
        if abs(i) < 0.1:
            straight.append(abs(i))
        else:
            turning.append(abs(i))
    # turning = [i for i in y_steering if i > 0.1]
    # straight = [i for i in y_steering if i <= 0.1]
    try:
        print("Moments of abs. val'd turning steering distribution:", generator.get_distribution_moments(turning))
        print("Moments of abs. val'd straight steering distribution:", generator.get_distribution_moments(straight))
    except Exception as e:
        print(e)
        print("len(turning)", len(turning))
        print("len(straight)", len(straight))


def main():
    start_time = time.time()
    input_shape = (640, 360)
    model = DAVE2v3(input_shape=input_shape)
    args = parse_arguments()
    print(args)
    dataset = MultiDirectoryDataSequence(args.dataset, image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]),\
                                         robustification=args.robustification, noise_level=args.noisevar) #, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())
    print("Total samples:", dataset.get_total_samples())
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, worker_init_fn=worker_init_fn)

    print("time to load dataset: {}".format(time.time() - start_time))

    iteration = f'{model._get_name()}-{input_shape[0]}x{input_shape[1]}-lr{args.lr}-{args.epochs}epoch-{args.batch}batch-lossMSE-{int(dataset.get_total_samples()/1000)}Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"{device=}")
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    logfreq = 20
    for epoch in range(args.epochs):
        running_loss = 0.0
        predictions_train = []
        true_values_train = []

        for i, hashmap in enumerate(trainloader, 0):
            x = hashmap['image'].float().to(device)
            y = hashmap['angular_speed_z'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)

            optimizer.zero_grad()
            outputs = model(x)
            loss = F.mse_loss(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('Actual Values:', y)
            print(f'Predicted Values:', {outputs})

            predictions_train.extend(outputs.cpu().detach().numpy())
            true_values_train.extend(y.cpu().detach().numpy())

            if i % logfreq == logfreq-1:
                print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / logfreq))
                
                if (running_loss / logfreq) < lowest_loss:
                    print(f"New best train model! MSE loss: {running_loss / logfreq}")
                    model_name = f"./model-{iteration}-train-best.pt"
                    print(f"Saving model to {model_name}")
                    torch.save(model.state_dict(), model_name)
                    lowest_loss = running_loss / logfreq
                running_loss = 0.0

        # Calculate and print training metrics
        mae_train = mean_absolute_error(true_values_train, predictions_train)
        print(f'Train MAE: {mae_train}')
        mse_train = mean_squared_error(true_values_train, predictions_train)
        print(f'Train MSE: {mse_train}')
        # Evaluation on the test dataset
        test_loss = 0.0
        predictions_test = []
        true_values_test = []

        for i, hashmap in enumerate(testloader, 0):
            x = hashmap['image'].float().to(device)
            y = hashmap['angular_speed_z']

            with torch.no_grad():  # No need to compute gradients during evaluation
                outputs = model(x)
                loss = F.mse_loss(outputs, y)
                test_loss += loss.item()

                predictions_test.extend(outputs.cpu().detach().numpy())
                true_values_test.extend(y.cpu().detach().numpy())

            if i % logfreq == logfreq-1:
                print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, test_loss / logfreq))
                
                if (test_loss / logfreq) < lowest_loss:
                    print(f"New best test model! MSE loss: {test_loss / logfreq}")
                    model_name = f"./model-{iteration}-test-best.pt"
                    print(f"Saving model to {model_name}")
                    torch.save(model.state_dict(), model_name)
                    lowest_loss = test_loss / logfreq
                test_loss = 0.0

        # Calculate and print test metrics
        mae_test = mean_absolute_error(true_values_test, predictions_test)
        print(f'Test MAE: {mae_test}')
        mse_test = mean_squared_error(true_values_test, predictions_test)
        print(f'Test MSE: {mse_test}')
        model_name = f"/mnt/c/Users/ishit/Documents/ROSbot_data_collection/model-{iteration}-epoch{epoch}.pt"
        print(f"Saving model to {model_name}")
        torch.save(model.state_dict(), model_name)
        # if loss < 0.002:
        #     print(f"Loss at {loss}; quitting training...")
        #     break
        print(f"Finished {epoch=}")
    print('Finished Training')

    # save model
    # torch.save(model.state_dict(), f'H:/GitHub/DAVE2-Keras/test{iteration}-weights.pt')
    model_name = f'/mnt/c/Users/ishit/Documents/ROSbot_data_collection/model-{iteration}.pt'
    torch.save(model.state_dict(), model_name)

    # delete models from previous epochs
    # print("Deleting models from previous epochs...")
    # for epoch in range(args.epochs):
    #     os.remove(f"/mnt/c/Users/ishit/Documents/ROSbot_data_collection/model-{iteration}-epoch{epoch}.pt")
    print(f"Saving model to {model_name}")
    print("All done :)")
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train))
    # save metainformation about training
    with open(f'/mnt/c/Users/ishit/Documents/ROSbot_data_collection/model-{iteration}-metainfo.txt', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={dataset.get_total_samples()}\n"
                f"{args.epochs=}\n"
                f"{args.lr=}\n"
                f"{args.batch=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{device=}\n"
                f"{args.robustification=}\n"
                f"{args.noisevar=}\n"
                f"dataset_moments={dataset.get_outputs_distribution()}\n"
                f"{time_to_train=}\n"
                f"dirs={dataset.get_directories()}")


if __name__ == '__main__':
    main()
