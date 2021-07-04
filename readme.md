data_loader.py parameters example:
--n_data_samples 900000 --wiki_dir C:\\my_documents\\datasets\\AMNLPFinal --filtered_data_dir C:\\my_documents\\datasets\\AMNLPFinal\\wiki_gt_6_sentences_100k --final_data_dir C:\\my_documents\\datasets\\AMNLPFinal\\miss_last_sentence_5_sentences_100k --manipulation_func miss_last_sentence_5_sentences --clean_and_filter_funcs longest_sequence longer_then_6_sentences clip_to_100k

trainer.py parameters example:
--end_epoch 20 --data_dir C:\\my_documents\\datasets\\AMNLPFinal\\miss_last_sentence_5_sentences_100k --model_dir C:\\my_documents\\models\\roberta_miss_last_sentence_5_sentences_100k