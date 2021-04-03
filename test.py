# Create error analysis file for model
import torch
import numpy as np
from models import PretrainedModel, Model, obtain_fasttext_embeddings
from data import get_ASR_datasets, get_SLU_datasets, read_config
from training import Trainer
import argparse
import os
if __name__ == '__main__':
	# Get args
	parser = argparse.ArgumentParser()
	parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
	parser.add_argument('--config_path', type=str, required=True, help='path to config file with hyperparameters, etc.')
	parser.add_argument('--error_path', type=str, required=True, help='path to store list of files with predicted errors.')
	parser.add_argument('--model_path', type=str, required=True, help='path of model to load')
	parser.add_argument('--use_FastText_embeddings', action='store_true', help='use FastText embeddings')
	parser.add_argument('--semantic_embeddings_path', type=str, help='path for semantic embeddings')
	parser.add_argument('--resplit_style', required=True, choices=['original','random', 'utterance_closed', "speaker_or_utterance_closed", "mutually_closed"], help='Path to root of fluent_speech_commands_dataset directory')
	parser.add_argument('--utility', action='store_true', help='Use utility driven splits')
	parser.add_argument('--smooth_semantic', action='store_true', help='sum semantic embedding of top k words')
	parser.add_argument('--smooth_semantic_parameter', type=str, default="5",help='value of k in smooth_smantic')
	parser.add_argument('--nlu_setup', action='store_true', help='use Gold utterances to run an NLU test pipeline')
	parser.add_argument('--asr_setup', action='store_true', help='use Gold utterances to run an ASR test pipeline')
	parser.add_argument('--single_label', action='store_true',help='Whether our dataset contains a single intent label (or a full triple). Only applied for the FSC dataset.')
	parser.add_argument('--noBLEU', action='store_true', help='compute results on split not optimised on BLEU score')
	
	args = parser.parse_args()
	restart = args.restart
	config_path = args.config_path
	model_path = args.model_path
	use_FastText_embeddings = args.use_FastText_embeddings
	semantic_embeddings_path = args.semantic_embeddings_path
	resplit_style = args.resplit_style
	utility = args.utility
	smooth_semantic = args.smooth_semantic
	smooth_semantic_parameter = int(args.smooth_semantic_parameter)
	nlu_setup = args.nlu_setup
	asr_setup = args.asr_setup
	single_intent = args.single_label
	noBLEU=args.noBLEU


	data_str=f"{resplit_style}_splits"

	# Read config file
	config = read_config(config_path)
	torch.manual_seed(config.seed); np.random.seed(config.seed)

	# Generate datasets
	use_gold_utterances = False
	use_all_gold=False
	if nlu_setup:
		use_gold_utterances = True
		use_all_gold=True

	if utility:
		data_str=data_str+"_utility"
	if noBLEU:
		data_str=data_str+"_noBLEU"

	if resplit_style=="speaker_or_utterance_closed":
		train_dataset, valid_dataset, test_closed_utterance_dataset, test_closed_speaker_dataset = get_SLU_datasets(config,data_str=data_str,split_style=resplit_style, use_all_gold = use_all_gold, use_gold_utterances = use_gold_utterances, single_label=single_intent, asr_setup = asr_setup)
	else:
		train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config,data_str=data_str,split_style=resplit_style, use_all_gold = use_all_gold, use_gold_utterances = use_gold_utterances, single_label=single_intent, asr_setup = asr_setup)

	# Initialize model
	if use_FastText_embeddings:
		Sy_word = []
		with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
			for line in f.readlines():
				Sy_word.append(line.rstrip("\n"))
		FastText_embeddings=obtain_fasttext_embeddings(semantic_embeddings_path, Sy_word)
		model = Model(config=config,pipeline=False, use_semantic_embeddings = use_FastText_embeddings, glove_embeddings=FastText_embeddings,glove_emb_dim=300, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
	else:
		model = Model(config=config)

	# Load the trained model
	trainer = Trainer(model=model, config=config)
	if restart: trainer.load_checkpoint(model_path)
	# Create csv file containing errors made by model
	if not asr_setup:
		if resplit_style=="speaker_or_utterance_closed":
			# test_utterance_intent_acc, test_utterance_intent_loss = trainer.test(test_closed_utterance_dataset,log_file="oo")
			# test_speaker_intent_acc, test_speaker_intent_loss = trainer.test(test_closed_speaker_dataset,log_file="oo")
			# print("========= Test results =========")
			# print("*intents*| test speaker accuracy: %.2f| test speaker loss: %.2f| test utterance accuracy: %.2f| test utterance loss: %.2f" % (test_speaker_intent_acc, test_speaker_intent_loss,test_utterance_intent_acc, test_utterance_intent_loss) )
			# if restart: trainer.load_checkpoint(model_path)
			test_utterance_intent_acc, test_utterance_intent_loss =  trainer.get_error(test_closed_utterance_dataset, error_path=args.error_path+"_utterance.csv")
			test_speaker_intent_acc, test_speaker_intent_loss = trainer.get_error(test_closed_speaker_dataset, error_path=args.error_path+"_semantic.csv")
			print("========= Test results =========")
			print("*intents*| test speaker accuracy: %.2f| test speaker loss: %.2f| test utterance accuracy: %.2f| test utterance loss: %.2f" % (test_speaker_intent_acc, test_speaker_intent_loss,test_utterance_intent_acc, test_utterance_intent_loss) )
		else:
			test_intent_acc, test_intent_loss = trainer.get_error(test_dataset, error_path=args.error_path)
			print("========= Test results =========")
			print("*intents*| test accuracy: %.2f| test loss: %.2f\n" % (test_intent_acc, test_intent_loss) )

	else:
		# Initialize model
		if use_FastText_embeddings:
			# Load FastText Embedding
			Sy_word = []
			with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
				for line in f.readlines():
					Sy_word.append(line.rstrip("\n"))
			FastText_embeddings=obtain_fasttext_embeddings(semantic_embeddings_path, Sy_word)
			model = Model(config=config,pipeline=False, use_semantic_embeddings = use_FastText_embeddings, glove_embeddings=FastText_embeddings,glove_emb_dim=300, smooth_semantic= smooth_semantic, smooth_semantic_parameter= smooth_semantic_parameter)
		else:
			model = Model(config=config)

		# Load the trained model
		trainer = Trainer(model=model, config=config)
		if restart: trainer.load_checkpoint(model_path)
		# Create csv file containing errors made by model
		if not asr_setup:
			test_intent_acc, test_intent_loss = trainer.get_error(test_dataset, error_path=args.error_path, nlu_setup = nlu_setup)
			print("========= Test results =========")
			print("*intents*| test accuracy: %.2f| test loss: %.2f\n" % (test_intent_acc, test_intent_loss) )
		else:
			# note, error path not logging errors yet 
			test_wer = trainer.get_asr_error(test_dataset, error_path=args.error_path)
			print("========= Test results =========")
			print("Average WER (weighted by length) : {:.2f}".format(test_wer))

