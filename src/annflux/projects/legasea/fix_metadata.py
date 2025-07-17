import pandas


def m():
    t_meta = pandas.read_csv("/mnt/big/indeed/legasea_big/annflux/stream_process_tile.csv")
    t = pandas.read_csv("/mnt/big/indeed/legasea_big/annflux/annflux.csv")

    t_meta = t_meta[['image_id', 'patch_x', 'patch_y']]
    print(t_meta.columns)
    print(t.columns)

    t = pandas.merge(t, t_meta, on='image_id')
    t.to_csv("/mnt/big/indeed/legasea_big/annflux/annflux.csv", index=False)


if __name__ == '__main__':
    m()