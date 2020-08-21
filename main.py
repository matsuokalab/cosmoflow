import tfrecord

path = "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small/train/univ_ics_2019-03_a10000668_000.tfrecord"

reader = tfrecord.reader.tfrecord_loader(
    data_path=path,
    index_path=None)

for data in reader:
    print(data)
    x = data["x"].reshape(128, 128, 128, 8) / 255 - 0.5
    y = data["y"]
    print(x.shape)
    print(x.max())