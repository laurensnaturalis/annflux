import pandas
import matplotlib.pyplot as plt

def m():
    t = pandas.read_csv("test_al_strategies.csv")

    plt.plot(t.accuracy, label="Accuracy")
    plt.plot(t.recall, label="Recall")

    plt.legend()
    plt.show()



if __name__ == '__main__':
    m()