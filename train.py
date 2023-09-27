import random
import json

import torch
import torch.nn as nn

from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('dic.json', 'r') as json_data:
    intents = json.load(json_data)

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to draw the bar graph
def draw_bar_graph():
    # Data for the bar graph (you can replace this with your own data)
    categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
    values = [10, 24, 15, 30]

    # Create a bar graph
    fig, ax = plt.subplots()
    ax.bar(categories, values)
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Bar Graph')

    # with open('data.csv', 'r') as file:
    #     csv_reader = csv.reader(file)
    #     next(csv_reader)  # Skip the header row if it exists
    #
    #     categories = []
    #     values = []
    #
    #     for row in csv_reader:
    #         categories.append(row[0])  # Assuming the first column contains category names
    #         values.append(int(row[1]))  # Assuming the second column contains numerical values
    #
    #     # Create a bar graph
    # fig, ax = plt.subplots()
    # ax.bar(categories, values)
    # ax.set_xlabel('Categories')
    # ax.set_ylabel('Values')
    # ax.set_title('Bar Graph')

    # Embed the Matplotlib plot into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack()

# Create the main window


# start the event







class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out



FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Zoobeasts"
while True:
    # sentence = "how are you?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    elif sentence == "transfer matrix method":
        window = tk.Tk()
        window.title("Bar Graph App")

        # Create a button to trigger the bar graph drawing
        draw_button = ttk.Button(window, text="Draw Bar Graph", command=draw_bar_graph)
        draw_button.pack()

        # Run the Tkinter main loop
        window.mainloop()
    else:
        pass

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Sorry, I am still learning, please come back later")
