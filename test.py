import random
# import logging
from loguru import logger
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

from os.path import join

# from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
import time

# 设置工作目录为当前文件路径
os.chdir(os.path.dirname(__file__))

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from prompt_bert.models import RobertaForCL, BertForCL
from prompt_bert.trainers import CLTrainer

# logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)





@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    middle_sentence: bool = field(
        default=False,
        metadata={
            "help": "Use middle sentence for unsupervised datasets."
        }
    )
    

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default="bert-base-uncased",
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    two_bert: bool = field(
        default=False,
        metadata={
        }
    )

    freeze_layers: str= field(
        default='',
        metadata={
        }
    )
    freeze_lm_head: bool = field(
        default=False,
        metadata={
        }
    )
    freeze_embedding: bool = field(
        default=False,
        metadata={
        }
    )
    label_smoothing: bool = field(
        default=False,
        metadata={
        }
    )
    two_bert_one_freeze: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_infomax: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_template: str = field(
        default='*cls*_This_sentence_:_\'*sent_0*\'_means*mask*.*sep+*',
        metadata={
        }
    )
    mask_embedding_sentence_bs: str = field(
        default='This sentence of "',
        metadata={
        }
    )
    mask_embedding_sentence_es: str = field(
        default='" means [MASK].',
        metadata={
        }
    )
    mask_embedding_sentence_with_mlm: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_different_template: str= field(
        default='*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*',
        metadata={
        }
    )
    mask_embedding_sentence_delta_freeze: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta_cross_stream: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta_no_position: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta_cotrain: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta_no_delta_eval: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_do_oxford: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_add_period: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_num_masks: int = field(
        default=1,
        metadata={
        }
    )
    mask_embedding_sentence_avg: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_add_template_in_batch: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_random_init: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_continue_training: str= field(
        default='',
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_continue_training_as_positive: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_freeze_prompt: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_org_mlp: bool = field(
        default=False,
        metadata={
        }
    )
    only_embedding_training: bool = field(
        default=False,
        metadata={
        }
    )
    roberta_with_special_token: bool = field(
        default=False,
        metadata={
        }
    )
    roberta_auto_weight_special_token: bool = field(
        default=False,
        metadata={
        }
    )
    roberta_special_token_as_cls: bool = field(
        default=False,
        metadata={
        }
    )
    remove_last_layer: bool = field(
        default=False,
        metadata={
        }
    )

    dot_sim: bool = field(
        default=False,
        metadata={
        }
    )

    norm_instead_temp: bool = field(
        default=False,
        metadata={
        }
    )
    add_pseudo_instances: bool = field(
        default=False,
        metadata={
        }
    )
    add_pseudo_instances_from_other_model: bool = field(
        default=False,
        metadata={
        }
    )

    only_negative_loss: bool = field(
        default=False,
        metadata={
        }
    )
    untie_weights_roberta: bool = field(
        default=False,
        metadata={
        }
    )
    token_classification: bool = field(
        default=False,
        metadata={
        }
    )
    add_rdrop: bool = field(
        default=False,
        metadata={
        }
    )

    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=10,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    train_file: Optional[str] = field(
        default="data/wiki1m_for_simcse.txt", 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    per_device_train_batch_size: int = field(
        default=256,
        metadata={"help": "The batch size per GPU for training."}
    )

    learning_rate: float = field(
        default=1e-5, metadata={"help": "The initial learning rate for Adam."}
    )

    num_train_epochs: float = field(default=1.0, metadata={"help": "Total number of training epochs to perform."})

    output_dir: str = field(
        default="output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )

    # reset follow flag type Optional[bool] -> bool
    # to fix typing error for TrainingArguments Optional[bool] in transformers==4.2.1
    # https://github.com/huggingface/transformers/pull/10672
    ddp_find_unused_parameters: bool = field(
        default=None,
        metadata={
            "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
            "`DistributedDataParallel`."
        },
    )
    disable_tqdm: bool = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    remove_unused_columns: bool = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    greater_is_better: bool = field(
        default=True, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )

    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )

    metric_for_best_model: str = field(
        default="stsb_spearman",
        metadata={"help": "The metric to use to compare two different models."},
    )

    eval_steps: int = field(
        default=125, metadata={"help": "Run an evaluation every X steps."}
    )

    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory."},
    )


    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})

    fp16: bool = field(default=False, metadata={"help": "Whether to use 16-bit (mixed) precision training."})

    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device





# main
if __name__ == "__main__":
        # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    use_vscode = True
    if (use_vscode):
        training_args.fp16 = True
        model_args.mask_embedding_sentence_delta = False
        model_args.mask_embedding_sentence = False
        model_args.mlp_only_train = True
        model_args.mask_embedding_sentence_template = '*cls*_This_sentence_:_\'*sent_0*\'_means*mask*.*sep+*'
        model_args.mask_embedding_sentence_different_template = '*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*'
        training_args.do_train = True
        training_args.load_best_model_at_end = True
        training_args.num_train_epochs = 1
        training_args.overwrite_output_dir = True


    # outpur_dir: The output directory where the model predictions and checkpoints will be written.
    training_args.output_dir = join(training_args.output_dir, model_args.model_name_or_path, 'RCL2_bsz-{}-lr-{}'.format(training_args.per_device_train_batch_size, training_args.learning_rate))


    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)


    model = BertForCL.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        model_args=model_args
    )

    # model = BertModel(config)

    model.train()

    ss = ['Today is a nice day!']
    inputs = tokenizer(ss, padding=True, return_tensors='pt')
    model.set_dropout_prob(0.0)


    # input_ids = torch.tensor([[101, 2003, 2002, 102]])  # 示例输入
    # attention_mask = torch.tensor([[1, 1, 1, 1]])  # 示例输入  
    outputs1 = model(inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state
    outputs2 = model(inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state

    if torch.equal(outputs1, outputs2):
        print('outputs1==outputs2')

    # print(outputs1)
    # print(outputs2)


   