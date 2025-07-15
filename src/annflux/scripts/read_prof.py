import glob
import pstats


def m():
    for fn in glob.glob("/home/lhogeweg/Documents/annflux_ln/src/annflux/ui/basic/prof/*"):
        for fn2 in pstats.Stats(fn).get_print_list([])[1]:
            if "quick.py" in fn2[0]:
                print(fn, fn2)


if __name__ == '__main__':
    m()