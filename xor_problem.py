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

end_of_training = False
requested_error = 0.01


# Inicjalizacja wag i obciążeń
input_size = 2
hidden_units = 2
output_size = 1

# Tworzymy wagi wypelnione losowymi wartosciami
weights_input_hidden = np.random.rand(input_size, hidden_units)
weights_hidden_output = np.random.rand(hidden_units, output_size)


# Tworzymy obciazenia wypelnione losowymi wartosciami
bias_hidden = np.zeros((1, hidden_units))
bias_output = np.zeros((1, output_size))

# Tworzymy zmienne do przechowywania poprzednich aktualizacji wag
prev_delta_weights_input_hidden = np.zeros_like(weights_input_hidden)
prev_delta_weights_hidden_output = np.zeros_like(weights_hidden_output)

rho = 0.9

# Funkcja aktywacji (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pętla treningowa
# Rozpoczynamy w tym miejscu trening ktory bedzie trwal okreslona liczbe
# epok (iteracji). Przy kazdej epoce przechodzimy raz przez caly zestaw 
# danych treningowych.
epochs = 200000

# Parametr ten okresla jak duzy wplyw ma wyliczony blad na aktualne wagi
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
        epoch += 1
        print(f"\nZakończono uczenie po epoce {epoch} z powodu osiągnięcia zadanego błędu poniżej {requested_error}")
        break

    classification_error_values.append(classification_error)

    mse_values.append(np.mean((outputs - predicted_output)**2))

    # Backpropagation - wsteczne propagowanie bledu
    # Tutaj obliczamy blad dla kazdej warstwy, nastepnie aktualizujemy wagi i obciazenia,
    # aby zminimializowac ten blad.
    output_error = error * predicted_output * (1 - predicted_output)
    hidden_layer_error = np.dot(output_error, weights_hidden_output.T) * hidden_layer_output * (1 - hidden_layer_output)

    # Aktualizacja wag i obciążeń - tutaj aktualizujemy wagi oraz bociazenia za pomoca 
    # gradientu, momentum oraz wspolczynnika uczenia (learning_rate). W tym miejscu tworzymy 
    # historię wag dodając przed zaktualizowaniem wag ich wartosc do tabeli. Potem aktualizujemy wagi.

    # Oblicz gradienty
    gradient_hidden_output = hidden_layer_output.T.dot(output_error)
    gradient_input_hidden = inputs.T.dot(hidden_layer_error)

     # Oblicz aktualizacje wag z uwzględnieniem momentum
    delta_weights_hidden_output = rho * prev_delta_weights_hidden_output - learning_rate * gradient_hidden_output
    delta_weights_input_hidden = rho * prev_delta_weights_input_hidden - learning_rate * gradient_input_hidden

    weights_hidden_output_history.append(np.copy(weights_hidden_output))
    weights_hidden_output += delta_weights_hidden_output
    bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate

    weights_input_hidden_history.append(np.copy(weights_input_hidden))
    weights_input_hidden += delta_weights_input_hidden
    bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

    # Zaktualizuj poprzednie aktualizacje wag
    prev_delta_weights_hidden_output = delta_weights_hidden_output
    prev_delta_weights_input_hidden = delta_weights_input_hidden

weights_hidden_output_history = np.array(weights_hidden_output_history)
weights_input_hidden_history = np.array(weights_input_hidden_history)
epoch += 1

plt.figure(figsize=(12,6))

# Wykres błędu klasyfikacji
plt.subplot(2, 2, 1)
plt.plot(range(epoch), classification_error_values)
plt.xlabel("Epoki")
plt.ylabel("Błąd klasyfikacji")
plt.title("Zmiana błędu klasyfikacji w czasie")

# Wykres zmian wag dla warstwy input-hidden
plt.subplot(2, 2, 2)
plt.plot(range(epoch), weights_input_hidden_history[:, 0, 0], label='Waga 1,1')
plt.plot(range(epoch), weights_input_hidden_history[:, 0, 1], label='Waga 1,2')
plt.xlabel('Epoki')
plt.ylabel('Wartość wagi')
plt.title('Zmiana wag dla warstwy input-hidden')
plt.legend()

# Wykres zmian wag dla warstwy hidden-output
plt.subplot(2, 2, 3)
plt.plot(range(epoch), weights_hidden_output_history[:, 0, 0], label='Waga 1,1')
plt.plot(range(epoch), weights_hidden_output_history[:, 1, 0], label='Waga 2,1')
plt.xlabel('Epoki')
plt.ylabel('Wartość wagi')
plt.title('Zmiana wag dla warstwy hidden-output')
plt.legend()

# Wykres MSE
plt.subplot(2, 2, 4)
plt.plot(range(epoch), mse_values)
plt.xlabel("Epoki")
plt.ylabel("MSE")
plt.title("Zmiana MSE w czasie")

# Pokaż wszystkie wykresy
plt.tight_layout()
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

data_1 = np.array([[1, 1]])
xor_test(data_1)
print("Wynik dla danych testowych", data_1, ":", xor_test(data_1))

# data_1 = np.array([[0, 1]])
# data_1 = np.array([[1, 1]])
xor_test(data_1)
print("Wynik dla danych testowych", data_1, ":", xor_test(data_1), "\n")


