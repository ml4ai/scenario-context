# When and Where Did it Happen? An Encoder-Decoder Model to Identify Scenario Context
Source code and data for `When and Where Did it Happen? An Encoder-Decoder Model to Identify Scenario Context` published in Findings of EMNLP 2024.

## Usage example
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "enoriega/scenario_context"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


passage_one = """On October 7, 2024, the city of Jakarta, Indonesia, declared a public health emergency following a sudden outbreak of dengue fever.
This outbreak cause significant contagion.
The rapid rise in cases during the outbreak, particularly in the northern and western districts, overwhelmed local hospitals and clinics,
with nearly 2,000 confirmed infections reported within the first week. 
Meanwhile, in Singapore, routine health surveillance continued smoothly despite concerns about regional disease spread.
In Jakarta, health authorities, in collaboration with the World Health Organization, initiated emergency vector control measures,
including widespread fumigation and public awareness campaigns. 
The government has urged residents to eliminate standing water around homes and public spaces,
as heavy monsoon rains have exacerbated the breeding of Aedes mosquitoes, the primary carriers of the virus.""".strip()


passage_two = """
While I was doing my PhD research, I had to do a lot of field work. In December of 2020, I traveled to egypt to conduct my last excavation. 
It was during this time that found the unsealed tomb of the great pharaon Tiktokhamon. In contrast, during my time in London the prior year, I wasn' able to do a lot of progress.
"""

prompt_template = "Text:\n{focus}\n\nContext:\n{passage}"

prompt = prompt_template.format(**{
	"focus": "found the unsealed tomb of the great pharaon Tiktokhamon",
	"passage": passage_two
	})

inputs = tokenizer(prompt, max_length=1024, truncation=True, return_tensors="pt")
generations = model.generate(input_ids=inputs['input_ids'], max_length=50)
predictions = tokenizer.batch_decode(generations, skip_special_tokens=True)

print(predictions)
# Output: ['time: December of 2020;location: egypt']
```
## Dataset
The data used to fine tune our models is located within the `data` directory.
```
data
├── curated
│   ├── annotations.json
├── paraphrases
|   ├── paraphrased.jsonl
├── syntethic
|   ├── annotations.jsonl
```
- `annotations.json`: The hand curated scenario context annotation files
- `paraphrased.jsonl`: Paraphrased versions of the hand curated annotations used for data augmentation
- `synthetic.jsonl`:  Proceduraly generated instances used for data augmentation

## HuggingFace Hub release 🤗
You can download the [fine-tuned models](https://huggingface.co/enoriega/scenario_context) from the HuggingFace Hub

## Training the model
This project contains the necessary code to train a T5 model to extract specific details (hereafter called contents) related to a given input text (hereafter called text).
The contents we are interested in are "location" and "time". 
There are two approaches investigated in this repository:

(1) Extracting (or generating) the full contents in one go, token by token.

(2) Generating the text for a specific type of content (e.g., "location", "time"). The full contents are generated, then, by generating the output individually, then assembling.


### Running Commands

```
CUDA_VISIBLE_DEVICES=0 python -i -m src.t5_specific_event --seed 1 --weight_decay 0.1 --model_name t5-base --saving_path results/240517/results_paraphrase_synthetic_240517 --use_paraphrase --use_synthetic --training_steps 1000  --use_original >> results/240517/results_paraphrase_synthetic_240517.txt
```

#### Flags
- `seed` -> The random seed to use; Used for: (1) Setting `transformers` seed (with `transformers.set_seed()`) and (2) Random object, for shuffling and data splitting
- `weight_decay` -> The weight decay parameter, used with Huggingface transformers `Trainer`
- `model_name` -> The name of the model, to be loaded (with `AutoModelForSeq2SeqLM.from_pretrained(model_name)`)
- `saving_path` -> Where to save the results. Used in (1) `Seq2SeqTrainingArguments` (`outputs/{saving_path}`), (2) Saving debug lines
- `training_steps` -> For how many steps to train for
- `learning_rate` -> The learning rate
- `use_original` -> Whether to use the original data for training; It is used for testing regardless of the status of this flag
- `use_paraphrase` -> Whether to use the paraphrase data for training and testing
- `use_synthetic` -> Whether to use the synthetic data for training and testing

## Citing this work
TODO
