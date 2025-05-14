import assessment_utils as au
import matplotlib.pyplot as plt
import json
from datetime import datetime
import torch

if __name__ == "__main__":
    # Printing the shape of the dataset allows you to know the number of input features
    # for X, y in au.train_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break

    # Instantiate the model
    model = au.Cifar_NN().to(au.device)
    # print(model)

    # THIS WAS GIVING ME SO MANY HEADACHES
    # ADAM BLOWS SGD OUT OF THE WATER!
    # WOOOOO
    optimiser = torch.optim.Adam(
        model.parameters(), lr=au.learning_rate, weight_decay=0.0005)

    # Perform the training
    accuracies = [0]
    start_time = datetime.now()
    for t in range(au.epochs):
        print(
            f"\nEpoch {t+1:>2d} / {au.epochs:>2d} [{"#" * round(86*((t+1)/au.epochs))}{"-" * round(86*(1-((t+1)/au.epochs)))}]")

        epoch_start_time = datetime.now()

        au.train(au.train_dataloader, model, au.loss_fn, optimiser)
        correct = au.test(au.test_dataloader, model, au.loss_fn)
        accuracies.append(100*correct)

        epoch_end_time = datetime.now()

        print(
            f"Epoch completed in {(epoch_end_time - epoch_start_time)}. Elapsed: {(epoch_end_time - start_time)}")
        improvement = (accuracies[-1] - accuracies[-2])
        print(
            f"Improvement of {"\033[92m" if improvement > 0 else "\033[91m\a"}{improvement:0.1f}\033[00m%")

    end_time = datetime.now()
    print(f"Done in {end_time - start_time}!")

    au.save_model_weights(model, au.model_save_path)

    with open("history.txt", "a") as history_file:
        history_file.write(f"{accuracies}\n")

    # Plot results
    plt.scatter(range(1, au.epochs+1), accuracies[1:])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy / %")
    plt.ylim(0, 100)
    plt.show()
