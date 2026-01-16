import pandas


def m():
    t = pandas.read_csv("/home/lhogeweg/annflux/data/papbig/annflux/annflux.csv")

    t = t[t.in_test==1]

    print(list(zip(t.uid, t.label_predicted, t.label_true)))



if __name__ == '__main__':
    m()