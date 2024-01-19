import numpy as np

# Dane treningowe
# Każda para liczb odpowiada jednemu zestawowi danych wejsciowych
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 

outputs = np.array([[0], [1], [1], [0]])

# Inicjalizacja wag i obciążeń
input_size = 2
hidden_units = 2
output_size = 1

# Tworzymy wagi wypelnione losowymi wartosciami
weights_input_hidden = np.random.rand(input_size, hidden_units)
bias_hidden = np.zeros((1, hidden_units))

# Tworzymy obciazenia wypelnione losowymi wartosciami
weights_hidden_output = np.random.rand(hidden_units, output_size)
bias_output = np.zeros((1, output_size))

# Funkcja aktywacji (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pętla treningowa
# Rozpoczynamy w tym miejscu trening ktory bedzie trwal okreslona liczbe
# epok (iteracji). Przy kazdej epoce przechodzimy raz przez caly zestaw 
# danych treningowych.
epochs = 1000000

# Parametr ten okresla jak duzy wplyw ma wyliczony blad na 
learning_rate = 0.01

for epoch in range(epochs):
    # Feedforward - przekazywanie w przod
    # Tutaj obliczana jest wartosc na wyjsciu sieci na podstawie danych wejsciowych
    # Obliczamy tu wartosc do warstwy ukrytej (hidden_layer_output) oraz 
    # wyjscie warstwy wyjsciowej (predicted_output) za pomoca funkcji aktywacji 
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Obliczenie błędu - jest to roznica miedzy przewidywanym wyjsciem, a wyjsciem rzeczywistym (outputs)
    error = outputs - predicted_output

    # Backpropagation - wsteczne propagowanie bledu
    # Tutaj obliczamy blad dla kazdej warstwy, nastepnie aktualizujemy wagi i obciazenia,
    # aby zminimializowac ten blad.
    output_error = error * predicted_output * (1 - predicted_output)
    hidden_layer_error = np.dot(output_error, weights_hidden_output.T) * hidden_layer_output * (1 - hidden_layer_output)

    # Aktualizacja wag i obciążeń - tutaj aktualizujemy wagi oraz bociazenia za pomoca 
    # gradientu i wspolczynnika uczenia (learning_rate).
    weights_hidden_output += np.dot(hidden_layer_output.T, output_error) * learning_rate
    bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += np.dot(inputs.T, hidden_layer_error) * learning_rate
    bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

# Testowanie na przykładzie danych - po zakonczeniu treningu sieci testujemy jej dzialanie podajac
# na jej wejscie konkretny przypadek do rozwiazania.
def xor_test(test_data):
    hidden_layer_input_test = np.dot(test_data, weights_input_hidden) + bias_hidden
    hidden_layer_output_test = sigmoid(hidden_layer_input_test)

    output_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output
    predicted_output_test = sigmoid(output_layer_input_test)

    print("Wynik dla danych testowych", data_1, ":", predicted_output_test)

data_1 = np.array([[0, 0]])
xor_test(data_1)
data_1 = np.array([[1, 0]])
xor_test(data_1)
data_1 = np.array([[0, 1]])
xor_test(data_1)
data_1 = np.array([[1, 1]])
xor_test(data_1)
