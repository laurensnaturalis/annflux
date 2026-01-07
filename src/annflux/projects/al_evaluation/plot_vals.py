import pandas
import matplotlib.pyplot as plt

def m():
    t = pandas.read_csv("test_al_strategies_step_50_linear_mod_1000.csv")
    t2 = pandas.read_csv("test_al_strategies_step_50_linear_at_round_10.csv")
    t3 = pandas.read_csv("test_al_strategies_step_50_linear_at_round_10_fre_strat.csv")
    t4 = pandas.read_csv("test_al_strategies.csv")
    plt.subplot(221)
    plt.plot(t.accuracy, label="step 50")
    plt.plot(t2.accuracy, label="step - linear at 10")
    plt.plot(t3.accuracy, label="step - linear at 10 - frestrat")
    plt.plot(t4.accuracy, label="frestrat - linear")
    plt.legend()

    plt.subplot(222)
    plt.plot(t.recall, label="Recall")
    plt.plot(t2.recall, label="Recall")
    plt.plot(t3.recall, label="Recall")
    plt.plot(t4.recall, label="Recall")

    plt.legend()
    plt.show()



if __name__ == '__main__':
    m()