import torch
import numpy as np
import pandas as pd
from models import PretrainedModel, Model, obtain_glove_embeddings, obtain_fasttext_embeddings
from data import get_ASR_datasets, get_SLU_datasets, read_config
from training import Trainer
import argparse
import os

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='run ASR pre-training')
parser.add_argument('--train', action='store_true', help='run SLU training')
parser.add_argument('--pipeline_train', action='store_true', help='run SLU training in pipeline manner')
parser.add_argument('--get_words', action='store_true', help='get words from SLU pipeline')
parser.add_argument('--save_words_path', default="/tmp/word_transcriptions.csv", help='path to save audio transcription CSV file')
parser.add_argument('--postprocess_words', action='store_true', help='postprocess words obtained from SLU pipeline')
parser.add_argument('--use_semantic_embeddings', action='store_true', help='use Glove embeddings')
parser.add_argument('--use_FastText_embeddings', action='store_true', help='use FastText embeddings')
parser.add_argument('--semantic_embeddings_path', type=str, help='path for semantic embeddings')
parser.add_argument('--finetune_embedding', action='store_true', help='tune SLU embeddings')
parser.add_argument('--finetune_semantics_embedding', action='store_true', help='tune semantics embeddings')
parser.add_argument('--resplit_style', required=True, choices=['original','random', 'utterance_closed', "speaker_or_utterance_closed", "mutually_closed","unseen","challenge"], help='Path to root of fluent_speech_commands_dataset directory')
parser.add_argument('--utility', action='store_true', help='Use utility driven splits')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
parser.add_argument('--pipeline_gold_train', action='store_true', help='run SLU training in pipeline manner with gold set utterances')
parser.add_argument('--seperate_RNN', action='store_true', help='run seperate RNNs over semantic embeddings and over SLU output')
parser.add_argument('--save_best_model', action='store_true', help='save the model with best performance on validation set')
parser.add_argument('--smooth_semantic', action='store_true', help='sum semantic embedding of top k words')
parser.add_argument('--smooth_semantic_parameter', type=str, default="5",help='value of k in smooth_smantic')
parser.add_argument('--complete', action='store_true', help='get over complete dataset')
parser.add_argument('--single_label', action='store_true',help='Whether our dataset contains a single intent label (or a full triple). Only applied for the FSC dataset.')
parser.add_argument('--nlu_setup', action='store_true', help='use Gold utterances to run an NLU test pipeline')
parser.add_argument('--perfect', action='store_true', help='compute results on perfect split')
parser.add_argument('--noBLEU', action='store_true', help='compute results on split not optimised on BLEU score')
parser.add_argument('--replace', action='store_true', help='compute results on split optimised using substitution')
parser.add_argument('--wer', action='store_true', help='compute results on split optimised using WER')
parser.add_argument('--dele', action='store_true', help='compute results on split optimised using delete')
parser.add_argument('--aggressive', action='store_true', help='compute results on split optimised aggressively')
parser.add_argument('--nonagg', action='store_true', help='compute results on split optimised using delete')
parser.add_argument('--seed', default=None, help='run on diff variants of same dataset')

args = parser.parse_args()
pretrain = args.pretrain

train = args.train
pipeline_train = args.pipeline_train
pipeline_gold_train = args.pipeline_gold_train
get_words = args.get_words
postprocess_words = args.postprocess_words
restart = args.restart
config_path = args.config_path
use_semantic_embeddings = args.use_semantic_embeddings
use_FastText_embeddings = args.use_FastText_embeddings
semantic_embeddings_path = args.semantic_embeddings_path
finetune_embedding = args.finetune_embedding
finetune_semantics_embedding = args.finetune_semantics_embedding
save_best_model = args.save_best_model
seperate_RNN = args.seperate_RNN
smooth_semantic = args.smooth_semantic
smooth_semantic_parameter = int(args.smooth_semantic_parameter)
resplit_style = args.resplit_style
utility = args.utility
complete = args.complete
perfect=args.perfect
noBLEU=args.noBLEU
replace=args.replace
wer=args.wer
dele=args.dele
aggressive=args.aggressive
nonagg = args.nonagg
seed=args.seed

data_str=f"{resplit_style}_splits"

if utility:
	data_str=data_str+"_utility"
if perfect:
	data_str=data_str+"_perfect"
if noBLEU:
	data_str=data_str+"_noBLEU"
if replace:
	print("yes")
	data_str=data_str+"_replace"
if wer:
	print("yes1")
	data_str=data_str+"_WER"
if dele:
	print("yes2")
	data_str=data_str+"_del"
if aggressive:
	print("yes3")
	data_str=data_str+"_aggressive"
if nonagg:
	print("yes3")
	data_str=data_str+"_nonagg"
if seed is not None:
	data_str=data_str+"_"+str(seed)

single_label = args.single_label
nlu_setup = args.nlu_setup 


# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

if pretrain:
	# Generate datasets
	train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)

	# Initialize base model
	pretrained_model = PretrainedModel(config=config)

	# Train the base model
	trainer = Trainer(model=pretrained_model, config=config)
	if restart: trainer.load_checkpoint()

	for epoch in range(config.pretraining_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
		train_phone_acc, train_phone_loss, train_word_acc, train_word_loss = trainer.train(train_dataset)
		valid_phone_acc, valid_phone_loss, valid_word_acc, valid_word_loss = trainer.test(valid_dataset)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
		print("*phonemes*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_phone_acc, train_phone_loss, valid_phone_acc, valid_phone_loss) )
		print("*words*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_word_acc, train_word_loss, valid_word_acc, valid_word_loss) )

		trainer.save_checkpoint()

if train:

	# Create corresponding model path based on the implementation
	log_file="log"
	model_path="model_state"

	if postprocess_words:
		log_file=log_file+"_postprocess"
		model_path=model_path + "_postprocess"

	log_file=log_file+"_"+f"{resplit_style}"
	model_path=model_path+"_"+f"{resplit_style}"

	if utility:
		log_file=log_file+"_utility"
		model_path=model_path + "_utility"

	if noBLEU:
		log_file=log_file+"_noBLEU"
		model_path=model_path + "_noBLEU"

	if replace:
		log_file=log_file+"_replace"
		model_path=model_path + "_replace"

	if wer:
		log_file=log_file+"_WER"
		model_path=model_path + "_WER"

	if dele:
		log_file=log_file+"_del"
		model_path=model_path + "_del"

	if aggressive:
		log_file=log_file+"_aggressive"
		model_path=model_path + "_aggressive"
	if nonagg:
		log_file=log_file+"_nonagg"
		model_path=model_path + "_nonagg"

	if seed is not None:
		log_file=log_file+"_"+str(seed)
		model_path=model_path + "_"+str(seed)

	if use_semantic_embeddings:
		log_file=log_file+"_glove"
		model_path=model_path + "_glove"
	elif use_FastText_embeddings:
		log_file=log_file+"_FastText"
		model_path=model_path + "_FastText"

	if smooth_semantic:
		log_file=log_file+"_smooth_"+str(smooth_semantic_parameter)
		model_path=model_path + "_smooth_"+str(smooth_semantic_parameter)

	if seperate_RNN:
		log_file=log_file+"_seperate"
		model_path=model_path + "_seperate"

	if finetune_semantics_embedding:
		log_file=log_file+"_finetune_semantic"
		model_path=model_path + "_finetune_semantic"

	if save_best_model:
		best_model_path=model_path + "_best.pth"
		best_valid_acc=0.0

	model_path=model_path + ".pth"

	# Generate datasets
	use_gold_utterances = False
	use_all_gold=False
	if nlu_setup:
		# make sure to load up word transcripts for train, val, AND test sets
		use_gold_utterances = True
		use_all_gold=True
	if (resplit_style=="unseen" or resplit_style=="challenge"):
		train_dataset, valid_dataset, test_closed_utterance_dataset, test_closed_speaker_dataset = get_SLU_datasets(config,data_str=data_str,split_style=resplit_style, single_label=single_label,\
	 	use_all_gold = use_all_gold, use_gold_utterances = use_gold_utterances)
	else:
		train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config,data_str=data_str,split_style=resplit_style, single_label=single_label,\
	 	use_all_gold = use_all_gold, use_gold_utterances = use_gold_utterances)
	# Initialize final model

	if use_semantic_embeddings: # Load Glove embedding
		Sy_word = []
		with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
			for line in f.readlines():
				Sy_word.append(line.rstrip("\n"))
		glove_embeddings=obtain_glove_embeddings(semantic_embeddings_path, Sy_word )
		model = Model(config=config,pipeline=False, use_semantic_embeddings = use_semantic_embeddings, glove_embeddings=glove_embeddings, finetune_semantic_embeddings= finetune_semantics_embedding, seperate_RNN=seperate_RNN, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
	elif use_FastText_embeddings: # Load FastText embedding
		Sy_word = []
		with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
			for line in f.readlines():
				Sy_word.append(line.rstrip("\n"))
		FastText_embeddings=obtain_fasttext_embeddings(semantic_embeddings_path, Sy_word)
		model = Model(config=config,pipeline=False, use_semantic_embeddings = use_FastText_embeddings, glove_embeddings=FastText_embeddings,glove_emb_dim=300, finetune_semantic_embeddings= finetune_semantics_embedding, seperate_RNN=seperate_RNN, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
	else:
		model = Model(config=config)

	# Train the final model
	trainer = Trainer(model=model, config=config)
	if restart: 
		trainer.load_checkpoint(model_path)
		config.training_num_epochs=0
		valid_intent_acc=0
		valid_intent_loss=0
		log_file=log_file+"_restart"
	log_file=log_file+".csv"
	for epoch in range(config.training_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		train_intent_acc, train_intent_loss = trainer.train(train_dataset,log_file=log_file)
		valid_intent_acc, valid_intent_loss = trainer.test(valid_dataset,log_file=log_file)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )

		trainer.save_checkpoint(model_path=model_path)
		if save_best_model: # Save best model observed till now
			if (valid_intent_acc>best_valid_acc):
				best_valid_acc=valid_intent_acc
				best_valid_loss=valid_intent_loss
				trainer.save_checkpoint(model_path=best_model_path)		

	if (resplit_style=="unseen" or resplit_style=="challenge"):
		test_utterance_intent_acc, test_utterance_intent_loss = trainer.test(test_closed_utterance_dataset,log_file=log_file)
		test_speaker_intent_acc, test_speaker_intent_loss = trainer.test(test_closed_speaker_dataset,log_file=log_file)
		print("========= Test results =========")
		print("*intents*| test speaker accuracy: %.2f| test speaker loss: %.2f| test utterance accuracy: %.2f| test utterance loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_speaker_intent_acc, test_speaker_intent_loss,test_utterance_intent_acc, test_utterance_intent_loss, valid_intent_acc, valid_intent_loss) )
	else:
		test_intent_acc, test_intent_loss = trainer.test(test_dataset,log_file=log_file)
		print("========= Test results =========")
		print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, valid_intent_acc, valid_intent_loss) )
	if save_best_model:
		if restart: 
			exit()
		trainer.load_checkpoint(model_path=best_model_path) # Compute performance of best model on test set
		if (resplit_style=="unseen" or resplit_style=="challenge"):
			test_utterance_intent_acc, test_utterance_intent_loss = trainer.test(test_closed_utterance_dataset,log_file=log_file)
			test_speaker_intent_acc, test_speaker_intent_loss = trainer.test(test_closed_speaker_dataset,log_file=log_file)
			print("========= Test results =========")
			print("*intents*| test speaker accuracy: %.2f| test speaker loss: %.2f| test utterance accuracy: %.2f| test utterance loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_speaker_intent_acc, test_speaker_intent_loss,test_utterance_intent_acc, test_utterance_intent_loss, valid_intent_acc, valid_intent_loss) )
		else:
			test_intent_acc, test_intent_loss = trainer.test(test_dataset,log_file=log_file)
			print("========= Test results =========")
			print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, best_valid_acc, best_valid_loss) )

if get_words: # Generate predict utterances by ASR module
	# Generate datasets
	Sy_word = []
	with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
		for line in f.readlines():
			Sy_word.append(line.rstrip("\n"))
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config,data_str=data_str,split_style=resplit_style)

	# Initialize final model
	if use_FastText_embeddings: # Load FastText embeddings
		FastText_embeddings=obtain_fasttext_embeddings(semantic_embeddings_path, Sy_word)
		model = Model(config=config,pipeline=False, use_semantic_embeddings = use_FastText_embeddings, glove_embeddings=FastText_embeddings,glove_emb_dim=300)

	else:
		model = Model(config=config)

	# Load pretrained model
	trainer = Trainer(model=model, config=config)
	if restart:
		if use_FastText_embeddings and smooth_semantic:
			trainer.load_checkpoint("model_state_disjoint_FastText_smooth_10_finetune_semantic_best.pth")
		elif use_FastText_embeddings and resplit_style=="utterance_closed":
			trainer.load_checkpoint("model_state_disjoint_FastText_finetune_semantic_best.pth")
		elif resplit_style=="utterance_closed":
			trainer.load_checkpoint("model_state_disjoint_best.pth")
		elif use_FastText_embeddings:
			trainer.load_checkpoint("model_state_FastText.pth")

	# get words from pretrained model
	if complete:
		train_predicted_words, train_audio_paths = trainer.get_word_SLU(train_dataset,Sy_word, postprocess_words, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
		valid_predicted_words, valid_audio_paths = trainer.get_word_SLU(valid_dataset,Sy_word, postprocess_words, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
		test_predicted_words, test_audio_paths = trainer.get_word_SLU(test_dataset,Sy_word, postprocess_words, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
		predicted_words=np.concatenate([train_predicted_words, valid_predicted_words, test_predicted_words], axis=0)
		audio_paths=np.concatenate([train_audio_paths, valid_audio_paths, test_audio_paths], axis=0)
	else:
		predicted_words, audio_paths = trainer.get_word_SLU(test_dataset,Sy_word, postprocess_words, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
	df=pd.DataFrame({'audio path': audio_paths, 'predicted_words': predicted_words}) # Save predicted utterances
	df.to_csv(args.save_words_path, index=False)

if pipeline_train: # Train model in pipeline manner
	# Generate datasets
	Sy_word = []
	with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
		for line in f.readlines():
			Sy_word.append(line.rstrip("\n"))
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config, single_label=single_label)

	if postprocess_words:
		log_file="log_pipeline_postprocess.csv"
	else:
		if finetune_embedding:
			log_file="log_pipeline_finetune.csv"
		else:
			log_file="log_pipeline.csv"
	
	# Initialize final model
	model = Model(config=config,pipeline=True,finetune=finetune_embedding)

	# Train the final model
	trainer = Trainer(model=model, config=config)
	if restart: trainer.load_checkpoint()

	for epoch in range(config.training_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		train_intent_acc, train_intent_loss = trainer.pipeline_train_decoder(train_dataset, postprocess_words,log_file=log_file)
		valid_intent_acc, valid_intent_loss = trainer.pipeline_test_decoder(valid_dataset,  postprocess_words,log_file=log_file)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )
		if postprocess_words:
			trainer.save_checkpoint(model_path="model_state_postprocess.pth")
		else:
			if finetune_embedding:
				trainer.save_checkpoint(model_path="model_state_pipeline_finetune.pth")
			else:
				trainer.save_checkpoint(model_path="model_state_pipeline.pth")

	test_intent_acc, test_intent_loss = trainer.pipeline_test_decoder(test_dataset, postprocess_words,log_file=log_file)
	print("========= Test results =========")
	print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, valid_intent_acc, valid_intent_loss) )

if pipeline_gold_train: # Train model in pipeline manner by using gold set utterances
	# Generate datasets
	if (resplit_style=="unseen" or resplit_style=="challenge"):
		train_dataset, valid_dataset, test_closed_utterance_dataset, test_closed_speaker_dataset = get_SLU_datasets(config,use_gold_utterances=True,data_str=data_str,split_style=resplit_style, single_label=single_label,use_all_gold=True)
	else:
		train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config,use_gold_utterances=True,data_str=data_str,split_style=resplit_style, single_label=single_label,use_all_gold=True)

	print(valid_dataset)
	# print(test_dataset)
	# Initialize final model
	if use_semantic_embeddings: # Load Glove embedding
		Sy_word = []
		with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
			for line in f.readlines():
				Sy_word.append(line.rstrip("\n"))
		glove_embeddings=obtain_glove_embeddings(semantic_embeddings_path, Sy_word )
		model = Model(config=config,pipeline=True, use_semantic_embeddings = use_semantic_embeddings, glove_embeddings=glove_embeddings, finetune_semantic_embeddings= finetune_semantics_embedding, seperate_RNN=seperate_RNN, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
	elif use_FastText_embeddings: # Load FastText embedding
		Sy_word = []
		with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
			for line in f.readlines():
				Sy_word.append(line.rstrip("\n"))
		FastText_embeddings=obtain_fasttext_embeddings(semantic_embeddings_path, Sy_word)
		model = Model(config=config,pipeline=True, use_semantic_embeddings = use_FastText_embeddings, glove_embeddings=FastText_embeddings,glove_emb_dim=300, finetune_semantic_embeddings= finetune_semantics_embedding, seperate_RNN=seperate_RNN, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
	else:
		model = Model(config=config,pipeline=True)

	# Train the final model
	trainer = Trainer(model=model, config=config)
	if restart: trainer.load_checkpoint()

	log_file="log_pipeline_gold"
	only_model_path="only_gold_model_state"
	with_model_path="with_gold_model_state"

	if postprocess_words:
		log_file=log_file+"_postprocess"
		only_model_path=only_model_path + "_postprocess"
		with_model_path=with_model_path + "_postprocess"
	log_file=log_file+"_"+f"{resplit_style}"
	only_model_path=only_model_path+"_"+f"{resplit_style}"
	with_model_path=with_model_path+"_"+f"{resplit_style}"
	if utility:
		log_file=log_file+"_utility"
		only_model_path=only_model_path+"_"+f"{resplit_style}"
		with_model_path=with_model_path+"_"+f"{resplit_style}"
	
	if use_semantic_embeddings:
		log_file=log_file+"_glove"
		only_model_path=only_model_path + "_glove"
	elif use_FastText_embeddings:
		log_file=log_file+"_FastText"
		only_model_path=only_model_path + "_FastText"

	log_file=log_file+".csv"
	only_model_path=only_model_path + ".pth"
	with_model_path=with_model_path + ".pth"

	if save_best_model:
		best_model_path=only_model_path + "_best.pth"
		best_valid_acc=0.0

	for epoch in range(config.training_num_epochs): # Train intent model on gold set utterances
		print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		train_intent_acc, train_intent_loss = trainer.pipeline_train_decoder(train_dataset,gold=True,log_file=log_file)
		valid_intent_acc, valid_intent_loss = trainer.pipeline_test_decoder(valid_dataset,gold=True, log_file=log_file)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )
		trainer.save_checkpoint(model_path=only_model_path)
		if save_best_model: # Save best model observed till now
			if (valid_intent_acc>best_valid_acc):
				best_valid_acc=valid_intent_acc
				best_valid_loss=valid_intent_loss
				trainer.save_checkpoint(model_path=best_model_path)
	# train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config,random_split=random_split, disjoint_split=disjoint_split)
	# for epoch in range(config.training_num_epochs): # Train intent model on predicted utterances
	# 	print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
	# 	train_intent_acc, train_intent_loss = trainer.pipeline_train_decoder(train_dataset, postprocess_words,log_file=log_file)
	# 	valid_intent_acc, valid_intent_loss = trainer.pipeline_test_decoder(valid_dataset, postprocess_words, log_file=log_file)

	# 	print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
	# 	print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )
	# 	trainer.save_checkpoint(model_path=with_model_path)
	if (resplit_style=="unseen" or resplit_style=="challenge"):
		test_utterance_intent_acc, test_utterance_intent_loss = trainer.pipeline_test_decoder(test_closed_utterance_dataset, gold=True, log_file=log_file)
		test_speaker_intent_acc, test_speaker_intent_loss = trainer.pipeline_test_decoder(test_closed_speaker_dataset, gold=True, log_file=log_file)
		print("========= Test results =========")
		print("*intents*| test speaker accuracy: %.2f| test speaker loss: %.2f| test utterance accuracy: %.2f| test utterance loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_speaker_intent_acc, test_speaker_intent_loss,test_utterance_intent_acc, test_utterance_intent_loss, valid_intent_acc, valid_intent_loss) )
	else:
		test_intent_acc, test_intent_loss = trainer.pipeline_test_decoder(test_dataset, gold=True, log_file=log_file)
		print("========= Test results =========")
		print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, valid_intent_acc, valid_intent_loss) )
	if save_best_model:
		trainer.load_checkpoint(model_path=best_model_path) # Compute performance of best model on test set
		if (resplit_style=="unseen" or resplit_style=="challenge"):
			test_utterance_intent_acc, test_utterance_intent_loss = trainer.pipeline_test_decoder(test_closed_utterance_dataset, gold=True, log_file=log_file)
			test_speaker_intent_acc, test_speaker_intent_loss = trainer.pipeline_test_decoder(test_closed_speaker_dataset, gold=True, log_file=log_file)
			print("========= Test results =========")
			print("*intents*| test speaker accuracy: %.2f| test speaker loss: %.2f| test utterance accuracy: %.2f| test utterance loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_speaker_intent_acc, test_speaker_intent_loss,test_utterance_intent_acc, test_utterance_intent_loss, valid_intent_acc, valid_intent_loss) )
		else:
			test_intent_acc, test_intent_loss = trainer.pipeline_test_decoder(test_dataset, gold=True, log_file=log_file)
			print("========= Test results =========")
			print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, best_valid_acc, best_valid_loss) )

