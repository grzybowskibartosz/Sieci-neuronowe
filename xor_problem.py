import numpy as np
import matplotlib.pyplot as plt

# Dane treningowe
# Każda para liczb odpowiada jednemu zestawowi danych wejsciowych
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 

outputs = np.array([[0], [1], [1], [0]])

mse_values = []
weights_hidden_output_history = []
weights_input_hidden_history = []
classification_error_values = []

end_of_training = True
requested_error = 0.01


# Inicjalizacja wag i obciążeń
input_size = 2
hidden_units = 2
output_size = 1
#epoch = 0

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
epochs = 20000

# Parametr ten okresla jak duzy wplyw ma wyliczony blad na 
learning_rate = 0.05

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

    classification_error = np.mean(np.abs(np.round(predicted_output)-outputs))

    if end_of_training and classification_error <= requested_error:
        print(f"\nZakończono uczenie po epoce {epoch + 1} z powodu osiągnięcia zadanego błędu poniżej {requested_error}")
        break

    classification_error_values.append(classification_error)

    mse_values.append(np.mean((outputs - predicted_output)**2))

    # Backpropagation - wsteczne propagowanie bledu
    # Tutaj obliczamy blad dla kazdej warstwy, nastepnie aktualizujemy wagi i obciazenia,
    # aby zminimializowac ten blad.
    output_error = error * predicted_output * (1 - predicted_output)
    hidden_layer_error = np.dot(output_error, weights_hidden_output.T) * hidden_layer_output * (1 - hidden_layer_output)

    # Aktualizacja wag i obciążeń - tutaj aktualizujemy wagi oraz bociazenia za pomoca 
    # gradientu i wspolczynnika uczenia (learning_rate). W tym miejscu tworzym historię wag dodając przed zaktualizowaniem wag ich wartosc do tabeli. Potem aktualizujemy wagi.

    weights_hidden_output_history.append(np.copy(weights_hidden_output))
    weights_hidden_output += np.dot(hidden_layer_output.T, output_error) * learning_rate
    bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate

    weights_input_hidden_history.append(np.copy(weights_input_hidden))
    weights_input_hidden += np.dot(inputs.T, hidden_layer_error) * learning_rate
    bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

weights_hidden_output_history = np.array(weights_hidden_output_history)
weights_input_hidden_history = np.array(weights_input_hidden_history)

plt.plot(range(epoch), classification_error_values)
plt.xlabel("Epoki")
plt.ylabel("Błąd klasyfikacji")
plt.title("Zmiana błędu klasyfikacji w czasie")
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(epoch), weights_input_hidden_history[:, 0, 0], label='Waga 1,1')
plt.plot(range(epoch), weights_input_hidden_history[:, 0, 1], label='Waga 1,2')
plt.xlabel('Epoki')
plt.ylabel('Wartość wagi')
plt.title('Zmiana wag dla warstwy input-hidden')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epoch), weights_hidden_output_history[:, 0, 0], label='Waga 1,1')
plt.plot(range(epoch), weights_hidden_output_history[:, 1, 0], label='Waga 2,1')
plt.xlabel('Epoki')
plt.ylabel('Wartość wagi')
plt.title('Zmiana wag dla warstwy hidden-output')
plt.legend()

plt.tight_layout()
plt.show()

plt.plot(range(epoch), mse_values)
plt.xlabel("Epoki")
plt.ylabel("MSE")
plt.show()
# Testowanie na przykładzie danych - po zakonczeniu treningu sieci testujemy jej dzialanie podajac
# na jej wejscie konkretny przypadek do rozwiazania.
def xor_test(test_data):
    hidden_layer_input_test = np.dot(test_data, weights_input_hidden) + bias_hidden
    hidden_layer_output_test = sigmoid(hidden_layer_input_test)

    output_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output
    predicted_output_test = sigmoid(output_layer_input_test)

    return predicted_output_test

# data_1 = np.array([[0, 0]])
# xor_test(data_1)
# print("Wynik dla danych testowych", data_1, ":", xor_test(data_1))

data_1 = np.array([[1, 0]])
xor_test(data_1)
print("Wynik dla danych testowych", data_1, ":", xor_test(data_1), "\n")

# data_1 = np.array([[0, 1]])
# xor_test(data_1)
# print("Wynik dla danych testowych", data_1, ":", xor_test(data_1))

# data_1 = np.array([[1, 1]])
# xor_test(data_1)
# print("Wynik dla danych testowych", data_1, ":", xor_test(data_1))
