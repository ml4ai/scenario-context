"""
A regulat seq2seq app, aiming to:
    (1) Take "input", which is a concatenation of text and context
    (2) Generate "output", which represents the contents

The model can be trained on a mix of original, paraphrased, and synthetic data
"""
from typing import Dict, Tuple
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed

import glob
import json
import tqdm
import random

import torch
import datasets

from datetime import datetime
from src.metrics import evaluate_sets
from src.make_data_structured import line_to_dict
from src.utils import get_hash, remove_unnecessary_content, preprocess_function
from src.metrics import evaluate_individual_sets_per_token
from src.parser_utils import get_parser

import wandb

# from allennlp.predictors import Predictor
# import allennlp_models.tagging
import re
import json
from tqdm import tqdm
# import spacy

# wandb.init(mode='disabled')

# nlp=spacy.load('en_core_web_sm')
# predictor = Predictor.from_path("structured-prediction-srl-bert.2020.12.15.tar.gz")


def count(data):
	# Compute TP, FP, TN, FN for LOC and TMP

	ret = {
		"loc": {
			"tp":0,
			"tn":0,
			"fp":0,
			"fn":0,
		},
		"tmp": {
			"tp":0,
			"tn":0,
			"fp":0,
			"fn":0,
		}
	}

	try:
	
		event = data["event"]
		

		preds = data.get("preds", {})
		#See if the event is attached
		event_attached = True

		# First, LOC
		# If gold is missing and pred is missing, TN
		if len(data["locations"]) == 0 and len(preds.get("locations", [])) == 0:
			ret['loc']["tn"] += 1
		# If gold is missing and there is a pred, FP
		elif len(data["locations"]) == 0 and len(preds.get("locations", [])) > 0:
			ret['loc']["fp"] += len(preds["locations"])
		# If gold exists and pred is missing, FN
		elif len(data["locations"]) > 0 and len(preds.get("locations", [])) == 0:
			ret['loc']["fn"] += len(preds.get("locations", []))
		else:
			# match = False
			
			for ref in data["locations"]:
				for pred in preds.get("locations", []):
					# To check for agreement, if any of the  references match lets consider it a hit, this is lenient
					ref = ref.lower().strip().replace(",", "")
					pred = pred.lower().strip().replace(",", "")
					match = ref == pred#ref in pred or pred in ref

					# If gold exists and pred is wrong, FP
					if not match:
						ret['loc']["fp"] += 1
					# If gold exists and pred agree and the event is attached, TP
					else:
						ret['loc']["tp"] += 1

		# Second, TMP
		
		# If gold is missing and pred is missing, TN
		if "time_periods" in preds:
			preds["time periods"] = preds["time_periods"]

		
		# If gold is missing and pred is missing, TN
		if len(data["temporals"]) == 0 and len(preds.get("time periods", [])) == 0:
			ret['tmp']["tn"] += 1
		# If gold is missing and there is a pred, FP
		elif len(data["temporals"]) == 0 and len(preds.get("time periods", [])) > 0:
			ret['tmp']["fp"] += len(preds["time periods"])
		# If gold exists and pred is missing, FN
		elif len(data["temporals"]) > 0 and len(preds.get("time periods", [])) == 0:
			ret['tmp']["fn"] += len(data["temporals"])
		else:
			# match = False
			
			for ref in data["temporals"]:
				for pred in preds.get("time periods", []):
					try:
						# To check for agreement, if any of the  references match lets consider it a hit, this is lenient
						ref = ref.lower().strip().replace(",", "")
						pred = pred.lower().strip().replace(",", "")
						match = ref == pred #ref in pred or pred in ref

						# If gold exists and pred is wrong, FP
						if not match:
							ret['tmp']["fp"] += 1
						# If gold exists and pred agree and the event is attached, TP
						else:
							ret['tmp']["tp"] += 1
					except:
						pass
	except:
		pass

	return ret


def accumulate(counts):
	c = counts[0]
	c["overall"] = {
		"tp":0,
		"fp":0,
		"tn":0,
		"fn":0
	}
	for d in counts[1:]:
		for type_ in d:
			for k, v in d[type_].items():
				c[type_][k] += v
				c["overall"][k] += v

	return c

def metrics(d):
	precision = d["tp"]/ ((d["tp"] + d["fp"])+1e-9)
	recall = d["tp"]/(( d["tp"] + d["fn"])+1e-9)
	f1 = (precision*recall)/((precision + recall)+1e-9)

	return {"p": precision, "r": recall, "f1":f1}


def compute(data):
	counts = accumulate([count(d) for d in data])
	ret = {}
	for type_, vals in counts.items():
		ret[type_] = metrics(vals)

	return ret



source_id2name = {
    0: 'original',
    1: 'paraphrase',
    2: 'synthetic',
    3: 'curated',
}

source_name2id = {v:k for (k, v) in source_id2name.items()}

parser = get_parser()

args = vars(parser.parse_args())

seed           = args['seed']
weight_decay   = args['weight_decay']
model_name     = args['model_name']
training_steps = args['training_steps']
learning_rate  = args['learning_rate']
saving_path    = args['saving_path']

set_seed(seed)
r = random.Random(seed)
original_data = []

##########################
### START READING DATA ###
##########################

# Read original
for f in glob.glob('data/original/*.json'):
    with open(f) as fin:
        original_data += json.load(fin)
        original_data = [{**x, 'source': source_name2id['original']} for x in original_data]

# Read the curated data
for f in glob.glob('data/curated/*.json'):
    with open(f) as fin:
        curated_data = json.load(fin)
        original_data += [{**x, 'source': source_name2id['curated']} for x in curated_data]

# Read paraphrases (if needed)
if args['use_paraphrase']:
    for f in glob.glob('data/paraphrases/240605/*.jsonl'):
        with open(f) as fin:
            for line in fin:
                original_data.append({**json.loads(line), 'source': source_name2id['paraphrase']})

# Read synthetic (if needed)
if args['use_synthetic']:
    for f in glob.glob('data/synthetic/*.jsonl'):
        with open(f) as fin:
            loaded = json.load(fin)
            original_data += [{**x, 'source': source_name2id['synthetic']} for x in loaded]

########################
### END READING DATA ###
########################



####################
### START FILTER ###
####################


# Keep only the data that contains all the fields: ['contents', 'text', 'pre_context', 'post_context']
original_data = [x for x in original_data if all(y in x.keys() for y in ['contents', 'text', 'pre_context', 'post_context'])]
# Skip some of the data
banned_contents_words = ['AUTHOR_INST', 'DATE', 'STRENGTHS', 'DESCRIPTION', 'USAGE', 'ASSUMPTIONS', 'All these instances are very', 'is relative temporal', 'AUTHOR_AUTHOR', 'AUTHOR', 'DATASET', 'DATASET', 'SCHEMA', "variation"]
for w in banned_contents_words:
    original_data = [x for x in original_data if w not in x['contents']]

##################
### END FILTER ###
##################



###########################
### START PREPROCESSING ###
###########################

# Set the "text" as the "original" field, which will be used as some form of "hash"
original_data = [{**x, 'original': x['text']} for x in original_data]
original_data = [remove_unnecessary_content(x) for x in original_data]
original_data = [x for x in original_data if len(x['contents'].split(" ")) < 8]

# Get all hashes, then shuffle
all_hashes = sorted(list(set([x['original'] for x in original_data])))
all_hashes = r.sample(all_hashes, k=len(all_hashes))

# The hashes of human annotated data
original_hashes = [x['original'] for x in original_data if x['source'] == source_name2id['original']]

if args['use_original'] is False:
    all_hashes = [x for x in all_hashes if x not in original_hashes]
    train_hashes = set(all_hashes[:int(len(all_hashes) * 0.8)])
    test_hashes  = set(all_hashes[len(train_hashes):] + original_hashes)
else:
    train_hashes = set(all_hashes[:int(len(all_hashes) * 0.8)])
    test_hashes  = set(all_hashes[len(train_hashes):])


test  = [x for x in original_data if x['original'] in test_hashes]


new_test = list()
for t in test:
    paragraph = t['pre_context'] + t['text'] + t['post_context']
    # paragraph = paragraph.replace("\\n", " ").replace("\n", " ")

    # sents_index = split_sentences(paragraph)

    sstart = len(t['pre_context'])
    send = sstart + len(t['text'])

    locs =  list()
    tmps = list()

    for x in t['contents'].split(";"):
        if x:
            ctx_type, vals = x.split(":")
            vals = [v.strip() for v in vals.split(",")]
            if ctx_type == "location":
                locs.extend(vals)
            else:
                tmps.extend(vals)

    # sents = [x[1] for x in sorted({get_sentence_from_char(s, sents_index) for s in (sstart, send)}, key=lambda t: t[0])]

    new_test.append({
        # 'sentence': "".join(sents),
        'pre_context': t['pre_context'],
		'post_context': t['post_context'],
        'event': t['text'],
        'locations': locs,
        'temporals': tmps,
        'source': source_id2name[t['source']]
    })


test = new_test

# data = datasets.DatasetDict({
#     'test' : datasets.Dataset.from_list(test),
# })

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenized = data.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

# print(tokenized)

#########################
### END PREPROCESSING ###
#########################


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOpenAI(
	base_url="https://llm1.cyverse.ai/v1",
    # model="gpt-4o",
    model="Mistral-7B-Instruct-v0.2",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="...",
    # base_url="...",
    # organization="...",
    # other params...
)



# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("human", 	"""For the following phrase, look at the event or concept surrounded by ``` and tell me all the locations and time periods, relevant to the focused element.
# 						The output format should be a json object with an array of strings for type of context. If there is not any element of a specific type, you will put an empty array in its value.
# 						Output format:
# 							{{
# 								"locations": [],
# 								"time periods": []
# 							}}
# 						Phrase:
# 						{pre_context} ```{event}``` {post_context}"""),
#     ]
# )

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", 	"""For the following phrase, look at the event or concept surrounded by ``` and tell me  the locations and time periods that relevant to the element surrounded by ```.
						The output format should be a json object with an array of strings for type of context. If there is not any element of a specific type, you will put an empty array in its value.
						Output format:
							{{
								"locations": [],
								"time periods": []
							}}
						Phrase:
						{pre_context} ```{event}``` {post_context}"""),
    ]
)

chain = prompt | llm | JsonOutputParser()




######################
### START EVALUATING ###
######################

def run_gpt(data):
    """Returns a dictionary with all semantic role labels for an input sentence
       outputs a role-label : word dictionary structure.
    """


    preds = chain.invoke({
         "pre_context": data["pre_context"],
         "event": data["event"],
         "post_context": data["post_context"],
    })
    
    return preds

for t in tqdm(test, desc="Running GPT"):
	try:
		preds = run_gpt(t)
		t["preds"] = preds
	except:
		pass


# with open("gpt4.json") as f:
# 	test = json.load(f)["data"]


scores = compute(test)
# print(scores)
print(json.dumps({"data":test, "scores":scores}))

####################
### END EVALUATING ###
####################





