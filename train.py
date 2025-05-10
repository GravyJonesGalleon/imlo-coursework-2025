import assessment_utils as au
import torch

if __name__ == "__main__":
    # Printing the shape of the dataset allows you to know the number of input features
    for X, y in au.train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Instantiate the model
    model = au.Cifar_NN(au.input_features).to(au.device)
    print(model)

    optimiser = torch.optim.SGD(model.parameters(), lr=au.lr)

    # Perform the training
    for t in range(au.epochs):
        print(f"Epoch {t+1}\n----------------------------")
        au.train(au.train_dataloader, model, au.loss_fn, optimiser)
        au.test(au.test_dataloader, model, au.loss_fn)
    print("Done!")

    au.save_model_weights(model, au.model_save_path)
