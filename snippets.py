import data_loader
import numpy as np

data = data_loader.read_data_from_csv("C:\\my_documents\\AMNLPFinal\\datasets\\missing_middle_5_sentences_out_of_11\\text_target")
dev = data[1]
np.random.seed(40)
chosen_indices = np.random.choice(range(len(dev)), 100, replace=False)
chosen_samples = dev.iloc[chosen_indices]
chosen_samples.to_csv("C:\\my_documents\\AMNLPFinal\\datasets\\missing_middle_5_sentences_out_of_11\\text_target_human_eval.tsv", sep='\t')
