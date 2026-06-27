from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def one_hot(*, actions: np.ndarray):
    """
    One hot encode the actions
    Parameters:
    ----------
    actions: np.ndarray
        The actions to one hot encode
    Returns:
    -------
    one_hot_actions: np.ndarray
        The one hot encoded actions
    """
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)
    
    one_hot_actions = nn.functional.one_hot(actions, num_classes=2)
    # convert to numpy array
    one_hot_actions = one_hot_actions.numpy()
    return one_hot_actions

def normalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Normalize the array
    Parameters:
    ----------
    array: np.ndarray
        The array to normalize
    mean: np.ndarray
        The mean of the array
    std: np.ndarray
        The std of the array
    Returns:
    -------
    normalized_array: np.ndarray
        The normalized array
    """
    return (array - mean) / std

def calculate_mean_std(*,states: np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and std of the states
    Parameters:
    ----------
    states: np.ndarray
        The states to calculate the mean and std of
    Returns:
    -------
    mean: np.ndarray
        The mean of the states
    std: np.ndarray
        The std of the states
    """
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0)
    return mean, std

def convert_to_dataloader(*,X:np.ndarray, y:np.ndarray, batch_size:int):
    tensor_X = torch.from_numpy(X).float()
    tensor_Y = torch.from_numpy(y).float()
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def trainer(*,
    model:nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    epochs:int,
    batch_size:int,
    learning_rate:float):

    """
    Train the model
    Parameters:
    ----------
    model: nn.Module
        The model to train
    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader
    epochs: int
    batch_size: int
    learning_rate: float
    Returns:
    -------
    model: nn.Module
        The trained model
    validation_loss_per_dim: torch.Tensor
        The validation loss per dimension
    """

    predicted_val = []
    true_val = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        validation_loss_per_dim = torch.zeros(4)
        with torch.no_grad():
            for i, (batch_X, batch_y) in enumerate(val_dataloader):
                outputs = model(batch_X)
                squared_error = (outputs - batch_y) ** 2
                validation_loss_per_dim += squared_error.sum(dim=0)
        validation_loss_per_dim /= len(val_dataloader)
        print(f"Validation Loss per dimension: {validation_loss_per_dim}")
        print(f"Train Loss: {train_loss}")

    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(val_dataloader):
            outputs = model(batch_X)
            predicted_val.append(outputs)
            true_val.append(batch_y)
    return model, predicted_val, true_val

class SimpleTransitionMLP(nn.Module):
    """
    Simple transition MLP model
    Parameters:
    ----------
    input_size: int
        The input size
    output_size: int
        The output size
    Returns:
    -------
    self.mlp: nn.Sequential
        The MLP model
    """
    def __init__(self, input_size:int, output_size:int):
        super(SimpleTransitionMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.mlp = nn.Sequential(nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, output_size))
    
    def forward(self, x: torch.Tensor):
        return self.mlp(x)

if __name__ == "__main__":
    data = np.load("C:\\AB_Personal\\world_model_pilot_project\\Phase_1\\data\\transitions.npz")
    actions = data["actions"]
    print("Actions shape: ", actions.shape)
    # one hot encode the actions
    one_hot_actions = one_hot(actions=actions)
    print("One hot actions shape: ", one_hot_actions.shape)

    states = data["states"]
    next_states = data["next_states"]  

    train_states, val_states, train_next_states, val_next_states, train_one_hot_actions, val_one_hot_actions = train_test_split(states, 
    next_states, one_hot_actions, test_size=0.2, random_state=42)

    print("Train states shape: ", train_states.shape)
    print("Val states shape: ", val_states.shape)
    print("Train next states shape: ", train_next_states.shape)
    print("Val next states shape: ", val_next_states.shape)
    print("Train one hot actions shape: ", train_one_hot_actions.shape)
    print("Val one hot actions shape: ", val_one_hot_actions.shape)

    # calculate the mean and std of the train states
    train_mean, train_std = calculate_mean_std(states=train_states)
    print("Train mean: ", train_mean)
    print("Train std: ", train_std)
    # save the mean and std to a file
    np.savez("data/mean_std.npz", mean=train_mean, std=train_std)

    # normalize the train states
    normalized_train_states = normalize(train_states, train_mean, train_std)
    normalized_val_states = normalize(val_states, train_mean, train_std)
    normalized_train_next_states = normalize(train_next_states, train_mean, train_std)
    normalized_val_next_states = normalize(val_next_states, train_mean, train_std)

    train_X = np.concatenate([normalized_train_states, train_one_hot_actions], axis=1)
    val_X = np.concatenate([normalized_val_states, val_one_hot_actions], axis=1)
    train_y = normalized_train_next_states
    val_y = normalized_val_next_states
    
    print("Train X shape: ", train_X.shape)
    print("Val X shape: ", val_X.shape)
    print("Train y shape: ", train_y.shape)
    print("Val y shape: ", val_y.shape)

    train_dataloader = convert_to_dataloader(X=train_X, y=train_y, batch_size=32)
    val_dataloader = convert_to_dataloader(X=val_X, y=val_y, batch_size=32)

    model = SimpleTransitionMLP(input_size=train_X.shape[1], output_size=train_y.shape[1])
    trained_model, predicted_val, true_val = trainer(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=10, batch_size=32, learning_rate=0.001)

    # save the trained model uisng torch.save
    if (not os.path.exists("../deep_learning_models")):
        os.makedirs("../deep_learning_models")
    torch.save(trained_model, "../deep_learning_models/transition_mlp.pth")
    print("Trained model saved to ../deep_learning_models/transition_mlp.pth")
        
    # convert torch to numpy array
    predicted_value = torch.concatenate(predicted_val, axis=0)
    true_value = torch.concatenate(true_val, axis=0)
    predicted_val_array = predicted_value.detach().numpy()
    true_val_array = true_value.detach().numpy()
    print("Predicted val array shape: ", predicted_val_array.shape)
    print("True val array shape: ", true_val_array.shape)
    names = ["car_pos", "car_vel", "pole_angle", "pole_ang_vel"]
    # plot the predicted vs true values for the other dimensions
    for i in range(1, 4):
        plt.scatter(range(len(true_val_array)), true_val_array[:, i], label="True", color="blue")
        plt.scatter(range(len(predicted_val_array)), predicted_val_array[:, i], label="Predicted", color="red")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"Predicted vs True Values for {names[i]}")
        plt.legend()
        plt.savefig(f"images/predicted_vs_true_values_{names[i]}.png")
        plt.close()

    # do diagonal plot for the predicted vs true values
    for i in range(4):
        plt.scatter(true_val_array[:, i], predicted_val_array[:, i], label="Predicted", color="red")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Predicted vs True Values for {names[i]}")
        plt.legend()
        plt.savefig(f"images/predicted_vs_true_values_diagonal_{names[i]}.png")
        plt.close()
