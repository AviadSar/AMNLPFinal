import data_loader

data = data_loader.read_data_from_csv("/home/aviad/Documents/AMNLPFinal/datasets/missing_middle_5_sentences_out_of_11/classification_target")
data = data[0][:][:10]
print(data)
a = 1
