import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import math as mt

from torchvision import datasets,transforms
tr=transforms.ToTensor()
train_data=datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=tr
)
test_data=datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=tr

)

x_tr=train_data.data
y_tr=train_data.targets
x_trr=np.array(x_tr)
y_trr=np.array(y_tr)
x_te=test_data.data
y_te=test_data.targets
x_tee=np.array(x_te)
y_tee=np.array(y_te)




x_train=x_trr.reshape(x_trr.shape[0],-1)/x_trr.max()
x_test=x_tee.reshape(x_tee.shape[0],-1)/x_tee.max()
x_test=x_test.reshape(x_test.shape[0],-1)
x_train=x_train.T
x_train=x_train[:,0:5000]
print(x_train.shape)
y_train=y_trr.reshape(y_trr.shape[0],1)
y_train=y_train[0:5000,0]

x_test=x_test.T

y_train=y_train.reshape(y_train.shape[0],1)
y_test=y_tee.reshape(y_tee.shape[0],1)



def initialisation(dimensions):
    parametres = {}
    C = len(dimensions)
    np.random.seed(1)
    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)
    M={}
    V={}
    for c in range(1, C):
        M['mw' + str(c)] = np.zeros((dimensions[c], dimensions[c - 1]))
        M['mb' + str(c)] = np.zeros((dimensions[c], 1))
        V['vw' + str(c)] = np.zeros((dimensions[c], dimensions[c - 1]))
        V['vb' + str(c)] = np.zeros((dimensions[c], 1))

    return parametres,M,V

def AdamAlgorithm(gradient,parametres,m,v,i):
    C = len(parametres) // 2
    beta=0.9
    beta2= 0.999
    Mg={}
    Vg={}
    for c in reversed(range(1, C + 1)):
        m['mw' + str(c)] = beta * m['mw' + str(c)] + (1 - beta) * gradient['dW' + str(c)]
        m['mb' + str(c)] = beta * m['mb' + str(c)] + (1 - beta) * gradient['db' + str(c)]
        v['vw' + str(c)] = beta2 * v['vw' + str(c)] + (1 - beta2) * np.power(gradient['dW' + str(c)], 2)
        v['vb' + str(c)] = beta2 * v['vb' + str(c)] + (1 - beta2) * np.power(gradient['db' + str(c)], 2)
        Mg['mw' + str(c)] = m['mw' + str(c)]/ (1 - pow(beta, i))
        Mg['mb' + str(c)] = m['mb' + str(c)]/ (1 - pow(beta, i))
        Vg['vw' + str(c)] = v['vw' + str(c)]/ (1 - pow(beta2, i))
        Vg['vb' + str(c)]=  v['vb' + str(c)]/ (1 - pow(beta2, i))

    return Mg,Vg,m,v
def update1(M,V, parametres, learning_rate):
    C = len(parametres) // 2
    e = mt.pow(10, -8)
    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W'+str(c)] - (learning_rate*M['mw'+ str(c)]/np.sqrt(V['vw'+ str(c)])+e)
        parametres['b' + str(c)] = parametres['b'+str(c)] - (learning_rate*M['mb'+ str(c)]/np.sqrt(V['vb'+ str(c)])+e)

    return parametres


def forward_propagation(X, parametres):
    activations = {'A0': X}
    C = len(parametres) // 2
    for c in range(1, C + 1):
        Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    return activations

def back_propagation(y, parametres, activations):
    m = y.shape[1]
    C = len(parametres) // 2
    dZ = activations['A' + str(C)] - y
    gradients = {}
    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (
                        1 - activations['A' + str(c - 1)])
    return gradients


def predict(X,parameters):
    AL=forward_propagation(X,parameters)
    C = len(parameters) // 2
    AL = AL['A' + str(C)]
    #print(AL)
    prediction=np.argmax(AL,axis=0)
    #print(prediction)
    prediction=np.array(prediction)
    prediction=prediction.reshape(prediction.shape[0],1)
    return prediction

def fonction_cout(y,A):
    epsilon = 1e-15
    m = len(y)
    return (-1 / m) * (np.sum(y * np.log(A + epsilon)) + np.sum((1 - y) * np.log(1 - A + epsilon)))

def deep_neural_network(X, y, hidden_layers, learning_rate, n_iter):
    # initialisation parametres
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])

    np.random.seed(1)
    parametres,mg1,vg1 = initialisation(dimensions)
    training_history = np.zeros((int(n_iter), 2))
    C = len(parametres) // 2
    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parametres)
        gradients = back_propagation(one_hot_encode(y, num_classes=10).T, parametres, activations)
        mg,vg,mg1,vg1 = AdamAlgorithm(gradients, parametres, mg1,vg1,i)

        parametres = update1(mg,vg, parametres, learning_rate)
        Af = activations['A' + str(C)]

        training_history[i, 0] = (fonction_cout(one_hot_encode(y, num_classes=10).T.flatten(), Af.flatten()))
        y_pred = predict(X, parametres)
        training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))

    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()

    return parametres,training_history


def one_hot_encode(targets, num_classes):

    n = targets.shape[0]
    one_hot_targets = np.zeros((n, num_classes))
    for i, target in enumerate(targets):
        one_hot_targets[i, target] = 1

    return one_hot_targets





X = x_train
y=y_train
parameter,accuracy=deep_neural_network(X, y, hidden_layers = (128,10), learning_rate = 0.002, n_iter = 5)

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

X_test=x_test[:,0:20]
y_pred=predict(X_test, parameter)


ac1=accuracy_score(y_test[0:20,0].flatten(),y_pred.flatten())
print(f"output:{y_test[0:20,0].flatten()}")
print(f"prediction:{y_pred.flatten()}")
print(f"accuracy:{accuracy[len(accuracy)-1,1]}")
print(f"accuracy:{ac1}")

x=x_tee[0:20]
for i in range(len(x)):
    plt.subplot(5,4,i+1)
    plt.imshow(x[i], cmap='gray')
    plt.title(f"prediction:{y_pred.flatten()[i]}".format(i))
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show()
