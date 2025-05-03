import matplotlib.pyplot as plt
import pickle

def main():
    file_name = './save/objects/cifar_cnn_3_C[0.5]_iid[1]_E[3]_B[32].pkl'
    with open(file_name, 'rb') as f:
        _, train_accuracy, test_acc = pickle.load(f)

        print(train_accuracy)
        print(test_acc)
        print(len(train_accuracy))

if __name__ == '__main__':
    main()