import h5py

with h5py.File('data/Pollen.h5', "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)

    d4_group = f["var"]
    for key in d4_group.keys():
        print(key, d4_group[key], d4_group[key].name, d4_group[key].value);