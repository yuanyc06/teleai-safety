# advprompter_refactored.py (file header patch)
import os
import sys
sys.path.append(os.getcwd())
import json
import time
import logging
import warnings
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import List, Optional, Dict, Any, Union
from tqdm import tqdm
import torch
import pytorch_lightning as pl
import importlib
from types import SimpleNamespace

# project imports (unchanged)
from utils import ConfigManager, parse_arguments, Metrics, dotdict
from models import load_model, Seq, MergedSeq, stack_seqs
from evaluation import PatternScorer
from torchrl.data import ListStorage, ReplayBuffer

# Try importing PrioritizedSampler normally (keep original behavior if available)
try:
    from torchrl.data.replay_buffers.samplers import PrioritizedSampler
except Exception:
    PrioritizedSampler = None  # we'll provide fallback if needed

from dataset import get_dataloader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------- monkey-patch for torchrl name mismatch --------------------
# Some torchrl releases may have renamed/moved the SumSegmentTreeFp32 class.
# We attempt to alias a compatible implementation into the samplers module so
# PrioritizedSampler (which expects SumSegmentTreeFp32) won't NameError.
try:
    samplers_mod = importlib.import_module("torchrl.data.replay_buffers.samplers")
except Exception:
    samplers_mod = None

if samplers_mod is not None and not hasattr(samplers_mod, "SumSegmentTreeFp32"):
    patched = False
    candidates = [
        "torchrl.data.replay_buffers.segment_tree",
        "torchrl.data.replay_buffers.segment_trees",
        "torchrl.data.replay_buffers.segment_tree_impl",
        "torchrl.data.replay_buffers",
        "torchrl.data.utils",
    ]
    for cand in candidates:
        try:
            m = importlib.import_module(cand)
        except Exception:
            m = None
        if m is None:
            continue
        # Prefer an exact name, otherwise alias a generic SumSegmentTree if exists
        if hasattr(m, "SumSegmentTreeFp32"):
            setattr(samplers_mod, "SumSegmentTreeFp32", getattr(m, "SumSegmentTreeFp32"))
            patched = True
            break
        if hasattr(m, "SumSegmentTree"):
            setattr(samplers_mod, "SumSegmentTreeFp32", getattr(m, "SumSegmentTree"))
            patched = True
            break
    if patched:
        logger.info("Patched torchrl.data.replay_buffers.samplers.SumSegmentTreeFp32 -> alias applied.")
    else:
        logger.warning(
            "Could not patch SumSegmentTreeFp32 for torchrl samplers. "
            "If PrioritizedSampler initialization fails, consider installing a compatible torchrl version or using the fallback sampler."
        )




column_names = [
    "step",
    "split",
    "batch_idx",
    "sample_idx",
    # Prompt prediction
    "prompter/ar/query",
    "prompter/ar/response",  # auto-regressive prompter generation
    "prompter/ar/response_perplexity_basemodel",
    #
    # --- Evaluation of predicted prompt ---
    "target_llm/ar/query",
    "target_llm/ar/response",
    "target_llm/ar/jailbroken",
]

# setproctitle.setproctitle("advprompter-train")
def collate_fn(list_of_data):
    (
        instruct_batch,
        target_batch,
        suffix_batch,
        priority_batch,
    ) = zip(*list_of_data)
    context = dotdict()
    context.instruct = stack_seqs(instruct_batch)
    context.target = stack_seqs(target_batch)
    context.suffix = stack_seqs(suffix_batch)
    return context, priority_batch

def log_data(
    log_table,
    metrics,
    step,
    split,
    batch_idx,
    test_prefixes,
    affirmative_prefixes,
    log_sequences_to_wandb,
    log_metrics_to_wandb,
    batch_size=None,
    target_llm_tf=None,
    target_llm_ar=None,
    prompter_ar=None,
    basemodel_tf=None,
    prompter_tf_opt=None,
):
    if batch_size is None and prompter_ar is None:
        raise ValueError("either batch_size or prompter_ar must be provided")
    bs = batch_size if batch_size is not None else prompter_ar.response_sample.bs
    log_dct = {}
    log_seqs = {
        "step": [step] * bs,
        "split": [split] * bs,
        "batch_idx": [batch_idx] * bs,
        "sample_idx": list(range(bs)),
    }

    if prompter_ar is not None:
        log_seqs["prompter/ar/query"] = prompter_ar.query.to_html()
        if basemodel_tf is not None:
            log_dct["prompter/ar/response_perplexity_basemodel"] = (
                basemodel_tf.perplexity.mean().item()
            )

            log_seqs["prompter/ar/response"] = prompter_ar.response_sample.to_html(
                colors=basemodel_tf.loss_masked, normalize=True, color_scheme=2
            )
            log_seqs["prompter/ar/response_perplexity_basemodel"] = (
                basemodel_tf.perplexity
            )
        else:
            log_seqs["prompter/ar/response"] = prompter_ar.response_sample.to_html()

    if target_llm_tf is not None:
        target_llm_tf_affirmative_avg, target_llm_tf_affirmative_list = (
            check_affirmative(
                seq=target_llm_tf.response_dist,
                affirmative_prefixes=affirmative_prefixes,
            )
        )

        log_dct["target_llm/tf/response_entropy"] = (
            target_llm_tf.response_dist.get_entropy().item()
        )
        log_dct["target_llm/tf/affirmative"] = target_llm_tf_affirmative_avg
        log_dct["target_llm/tf/loss"] = target_llm_tf.loss.item()

    if target_llm_ar is not None:
        target_llm_ar_jailbroken_avg, target_llm_ar_jailbroken_list = check_jailbroken(
            seq=target_llm_ar.response_sample, test_prefixes=test_prefixes
        )

        # log_dct["target_llm/ar/jailbroken"] = target_llm_ar_jailbroken_avg
        log_dct["target_llm/ar/jailbroken_sum"] = sum(target_llm_ar_jailbroken_list)

        log_seqs["target_llm/ar/query"] = target_llm_ar.query.to_html()
        log_seqs["target_llm/ar/response"] = target_llm_ar.response_sample.to_html()
        log_seqs["target_llm/ar/jailbroken"] = target_llm_ar_jailbroken_list

    if prompter_tf_opt is not None:
        log_dct["prompter/tf/opt/response_dist_entropy"] = (
            prompter_tf_opt.response_dist.get_entropy().item()
        )
        log_dct["prompter/tf/opt/loss"] = prompter_tf_opt.loss.item()

    metrics.log_dict(log_dct, step=step, log_to_wandb=log_metrics_to_wandb)
    if log_sequences_to_wandb:
        log_data_to_table(log_table, bs, log_seqs)


def log_data_to_table(log_table, bs, log_seqs):
    log_list = []

    for column_name in column_names:
        if column_name in log_seqs:
            log_list.append(log_seqs[column_name])
        else:
            log_list.append([None] * bs)

    for bi in range(bs):
        log_l = [x[bi] for x in log_list]
        log_table.add_data(*log_l)


# -------------------------
# Configuration dataclasses
# -------------------------
@dataclass
class WandbParams:
    enable_wandb: bool = False
    entity: Optional[str] = None
    project: Optional[str] = None
    log_sequences_every: Dict[str, int] = field(default_factory=lambda: {"train": 100})


@dataclass
class ReplayBufferConfig:
    size: int = 200000
    priority_alpha: float = 0.6
    priority_factor: Any = field(default_factory=lambda: argparse_namespace_like({
        "loss_delta": 1.0,
        "jailbreaking": 1.0
    }))

# small helper for default nested-like args placeholder (if you don't have argparse Namespace)
def argparse_namespace_like(d: dict):
    class X:
        def __init__(self, dd):
            for k, v in dd.items():
                setattr(self, k, v)
    return X(d)

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 8
    prompter_optim_params: dict = field(default_factory=lambda: {"lr": 1e-4})
    q_params: Any = field(default_factory=lambda: argparse_namespace_like({
        "max_new_tokens": 64,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "top_k": 50,
        "num_chunks": 1,
        "candidates": argparse_namespace_like({
            "do_sample": True,
            "temperature": 1.0,
            "always_include_best": True,
        }),
        "lambda_val": 1.0,
        "beams": argparse_namespace_like({
            "do_sample": False,
            "temperature": 1.0,
            "always_include_best": True,
        }),
    }))
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    eval_every: Optional[int] = None
    model_save_dir: Optional[str] = "./ckpts"
    save_pretrain: bool = True
    always_save_before_eval: bool = True
    dataset_key: str = "train"

@dataclass
class EvalConfig:
    batch_size: int = 8
    num_trials: int = 1
    prompter: Any = field(default_factory=lambda: argparse_namespace_like({
        "max_new_tokens_list": [16],
        "do_sample": True
    }))
    data: Any = field(default_factory=lambda: argparse_namespace_like({"dataset_pth_dct": {}}))
    suffix_dataset_pth_dct: dict = field(default_factory=dict)

@dataclass
class GenParams:
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.9

@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "lm_head"])

@dataclass
class LoraParams:
    warmstart: Optional[bool] = False
    lora_checkpoint: Optional[str] = None
    lora_config: Optional[LoraConfig] = None

@dataclass
class LLMParams:
    device: str = "cuda:1"
    freeze: bool = False
    dtype: str = "float32"
    model_name: str = "llama2-7b"
    checkpoint: str = "meta-llama/Llama-2-7b-hf"
    lora_params: LoraParams = field(default_factory=LoraParams)

@dataclass
class PrompterConfig:
    llm_params: LLMParams = field(default_factory=LLMParams)
    allow_non_ascii: bool = False
    gen_params: GenParams = field(default_factory=GenParams)
    prompt_manager: Optional[Any] = None  # will be set in __post_init__ if missing


@dataclass
class PromptTemplate:
    key: str
    msg: str

@dataclass
class PromptManager:
    prompt_template: List[PromptTemplate]

@dataclass
class Prompter:
    llm_params: LLMParams = field(default_factory=LLMParams)  # Default LLMParams if not provided
    allow_non_ascii: Optional[bool] = False  # Default from YAML
    gen_params: Optional[dict] = None  # Default GenParams if not provided
    prompt_manager: PromptManager = None  # PromptManager should be passed when initializing

    def __post_init__(self):
        if self.prompt_manager is None:
            if "llama2" in self.llm_params.model_name.lower():
                # Default prompt manager with templates
                self.prompt_manager = PromptManager(
                    prompt_template=[
                        PromptTemplate(key="system_message", msg="<s>"),
                        PromptTemplate(key="hyper_instruct", msg="{instruct}"),
                        PromptTemplate(key="suffix", msg="{suffix}")
                    ]
                )
            else:
                raise ValueError("Unsupported model name: {}, you must set prompt_manager manually".format(self.llm_params.model_name))
        if self.gen_params is None:
            self.gen_params = dict(
                do_sample=True,
                temperature=1.0,
                top_p=0.9
            )


@dataclass
class AdvPrompterConfig:
    # general
    seed: int = 42
    verbose: bool = True
    wandb_params: WandbParams = field(default_factory=WandbParams)
    mode: str = "train"  # train / eval / eval_suffix_dataset

    # data / io
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompter: PrompterConfig = field(default_factory=PrompterConfig)

    # unified model descriptors for loading via load_model (optional)
    prompter_model_type: Optional[str] = None
    prompter_model_name: Optional[str] = None
    prompter_model_path: Optional[str] = None

    target_model_type: Optional[str] = None
    target_model_name: Optional[str] = None
    target_model_path: Optional[str] = None

    # unified auth / urls (used by load_model)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    azure_key: Optional[str] = None
    azure_url: Optional[str] = None
    grok_key: Optional[str] = None
    grok_url: Optional[str] = None

    # checkpoints / outputs
    train: TrainConfig = field(default_factory=TrainConfig)
    res_save_path: str = "./results/advprompter.jsonl"


# -------------------------
# BaseWorkspace (refactored)
# -------------------------
class BaseWorkspace:
    def __init__(self, cfg: AdvPrompterConfig):
        self.cfg = cfg
        self.step = 0
        self.verbose = cfg.verbose
        self.enable_wandb = cfg.wandb_params.enable_wandb
        self._init_seed_and_device()
        self._load_models()
        self._load_prefixes_data()
        if self.enable_wandb:
            self._init_wandb()
        else:
            self.train_table = None
            self.eval_table = None

    def _init_seed_and_device(self):
        pl.seed_everything(self.cfg.seed)

    def _init_wandb(self):
        import wandb

        tqdm.write("Initializing Wandb...")
        wandb_id = wandb.util.generate_id()
        wandb.init(
            entity=self.cfg.wandb_params.entity,
            project=self.cfg.wandb_params.project,
            config=self.cfg,
            id=wandb_id,
            resume="allow",
        )
        self.train_table = wandb.Table(columns=column_names)
        self.eval_table = wandb.Table(columns=column_names)

    @torch.no_grad()
    def _load_models(self):
        """
        Load prompter and target LLMs via unified load_model.
        Note: your original code used a custom LLM wrapper. Here we try to load a model via load_model,
        then adapt to expected LLM wrapper if necessary.
        """
        logger.info("Initializing Prompter via load_model...")
        # if prompter_model_type provided, use load_model; else fallback to original LLM wrapper if available
        if getattr(self.cfg, "prompter_model_type", None):
            self.prompter = load_model(
                model_type=self.cfg.prompter_model_type,
                model_name=self.cfg.prompter_model_name,
                model_path=self.cfg.prompter_model_path,
                config=self.cfg,
            )
        else:
            # Fallback: try to initialize your project's LLM wrapper with PrompterConfig
            try:
                from models import LLM  # original project wrapper
                self.prompter = LLM(self.cfg.prompter, verbose=self.verbose)
            except Exception:
                raise RuntimeError(
                    "Failed to load prompter: provide prompter_model_type/model_name or ensure LLM wrapper is available"
                )

        logger.info("Initializing Target LLM via load_model...")
        if getattr(self.cfg, "target_model_type", None):
            self.target_llm = load_model(
                model_type=self.cfg.target_model_type,
                model_name=self.cfg.target_model_name,
                model_path=self.cfg.target_model_path,
                config=self.cfg,
            )
        else:
            try:
                from models import LLM
                self.target_llm = LLM(self.cfg.prompter, verbose=self.verbose)  # fallback; adjust if needed
            except Exception:
                raise RuntimeError("Failed to load target_llm; set target_model_type/name/path in config")

    @torch.no_grad()
    def _load_prefixes_data(self):
        # Use PatternScorer as original code to load prefixes
        try:
            ps = PatternScorer()
            self.test_prefixes = ps.pattern_dict["fail"]
            self.affirmative_prefixes = ps.pattern_dict["pass"]
        except Exception:
            self.test_prefixes = []
            self.affirmative_prefixes = []

    def batch_to_context(self, batch):
        """
        Convert raw batch dict into Seq objects for prompter/target LLMs.
        This preserves the original mapping: instruct -> prompter/tokenizer, suffix -> prompter, target/full_instruct -> target_llm
        """
        model_map = dict(
            instruct=self.prompter,
            suffix=self.prompter,
            target=self.target_llm,
            full_instruct=self.target_llm,
        )
        context = argparse_namespace_like({})
        # Seq is project-specific; assume it's available
        try:
            from models import Seq
        except Exception:
            Seq = None

        for key, model in model_map.items():
            if key in batch.keys():
                if Seq is None:
                    # simple fallback: provide raw strings in a minimal container
                    class _S:
                        def __init__(self, text):
                            self.text = text
                            self.bs = len(text)
                            self.to_html = lambda *a, **k: text
                    context.__dict__[key] = _S(batch[key])
                else:
                    context.__dict__[key] = Seq(text=batch[key], tokenizer=model.tokenizer, device=model.device)
            else:
                context.__dict__[key] = None
        return context

    @torch.no_grad()
    def save_prompter(self):
        # Save prompter model — if the loaded prompter supports `.save_pretrained`, call it
        save_dir = getattr(self.cfg.train, "model_save_dir", None) or "./ckpts"
        save_path = os.path.join(save_dir, f"step_{self.step}")
        tqdm.write(f" Saving prompter to {save_path}...")
        try:
            if hasattr(self.prompter, "save_pretrained"):
                self.prompter.save_pretrained(save_path=save_path)
            elif hasattr(self.prompter, "save"):
                self.prompter.save(save_path)
            else:
                logger.warning("Prompter object has no save method; skipping.")
        except Exception:
            logger.exception("Failed to save prompter.")


# -------------------------
# EvalSuffixDatasetsWorkspace and EvalWorkspace
# -------------------------
class EvalSuffixDatasetsWorkspace(BaseWorkspace):
    def __init__(self, cfg: AdvPrompterConfig):
        super().__init__(cfg)
        # set prompter/target to eval mode if wrappers provide .eval()
        if hasattr(self.prompter, "eval"):
            self.prompter.eval()
        if hasattr(self.target_llm, "eval"):
            self.target_llm.eval()

    @torch.no_grad()
    def eval_suffix_datasets(self, suffix_dataset_pth_dct: Dict[str, str]):
        for k, p in suffix_dataset_pth_dct.items():
            self.eval_suffix_dataset(k, p)

    @torch.no_grad()
    def eval_suffix_dataset(self, suffix_dataset_key: str, suffix_dataset_pth: str):
        split = re.sub("[^a-zA-Z]", "", suffix_dataset_key)
        eval_loader = get_dataloader(
            suffix_dataset_pth,
            shuffle=False,
            augment_target=False,
            batch_size=self.cfg.eval.batch_size,
        )
        num_batches = len(eval_loader)
        eval_metrics = Metrics(prefix=split + "_eval/")

        instruct_jb_dict = defaultdict(list)
        processed_samples, ppl_sum = 0, 0

        logger.info(f"Start evaluating suffix dataset {suffix_dataset_key}, {num_batches} batches in total")

        for batch_idx, batch in enumerate(eval_loader):
            context = self.batch_to_context(batch)
            instruct, suffix, full_instruct, target = context.instruct, context.suffix, context.full_instruct, context.target

            target_llm_tf, target_llm_ar, basemodel_tf = evaluate_prompt(
                instruct=instruct,
                suffix=suffix,
                full_instruct=full_instruct,
                target=target,
                prompter=self.prompter,
                target_llm=self.target_llm,
                generate_target_llm_response=True,
                reweight_loss=self.cfg.reweight_loss if hasattr(self.cfg, "reweight_loss") else None,
                verbose=self.verbose,
                print_idx=0,
            )

            _, jailbroken_list = check_jailbroken(seq=target_llm_ar.response_sample, test_prefixes=self.test_prefixes)
            assert instruct.bs == len(jailbroken_list)
            for i in range(instruct.bs):
                instruct_jb_dict[instruct.text[i]].append(jailbroken_list[i])

            log_data(
                log_table=self.eval_table,
                metrics=eval_metrics,
                step=self.step,
                split=split,
                batch_idx=batch_idx,
                test_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
                batch_size=self.cfg.eval.batch_size,
                log_sequences_to_wandb=False,
                log_metrics_to_wandb=False,
                target_llm_tf=target_llm_tf,
                target_llm_ar=target_llm_ar,
                basemodel_tf=basemodel_tf,
            )
            processed_samples += instruct.bs
            if basemodel_tf is not None:
                ppl_sum += basemodel_tf.perplexity.sum().item()
            total_jailbroken = sum(eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"])
            logger.info(f"Evaluating {suffix_dataset_key} | {batch_idx+1}/{num_batches} batches completed | {total_jailbroken}/{processed_samples} of processed samples are jailbroken |  PPL: {float(ppl_sum) / processed_samples:.2f}")

        avg_metrics = eval_metrics.get_avg(step=self.step, log_to_wandb=False)
        avg_metrics["avg/" + split + "_eval/target_llm/ar/jailbroken_sum"] = float(sum(eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"])) / processed_samples
        logger.info(f" Loss: {avg_metrics['avg/' + split + '_eval/target_llm/tf/loss']:.2f}")
        logger.info(f" Jailbroken: {avg_metrics['avg/' + split + '_eval/target_llm/ar/jailbroken_sum']:.2f}")
        logger.info(f" PPL: {float(ppl_sum) / processed_samples:.2f}")

        if self.enable_wandb:
            import wandb
            wandb.log(avg_metrics, step=self.step)
            wandb.log(dict(eval_examples=copy(self.eval_table)), step=self.step)


class EvalWorkspace(EvalSuffixDatasetsWorkspace):
    def __init__(self, cfg: AdvPrompterConfig):
        super().__init__(cfg)

    @torch.no_grad()
    def eval(self):
        suffix_dataset_pth_dct = self.generate_suffix_datasets()
        self.eval_suffix_datasets(suffix_dataset_pth_dct)

    @torch.no_grad()
    def generate_suffix_datasets(self):
        suffix_dataset_pth_dct = {}
        for dataset_key, dataset_pth in self.cfg.eval.data.dataset_pth_dct.items():
            suffix_dataset = self.generate_suffix_dataset(dataset_key, dataset_pth)
            suffix_dataset_pth = self.save_suffix_dataset(suffix_dataset, dir=self.cfg.eval.data.suffix_dataset_dir)
            suffix_dataset_pth_dct[suffix_dataset.suffix_dataset_key] = suffix_dataset_pth
        return suffix_dataset_pth_dct

    @torch.no_grad()
    def generate_suffix_dataset(self, dataset_key, dataset_pth):
        if self.cfg.prompter.gen_params.do_sample:
            num_trials = self.cfg.eval.num_trials
        else:
            if self.cfg.eval.num_trials != 1:
                warnings.warn("Prompter generation is deterministic, but num_trials > 1. Setting num_trials to 1.")
            num_trials = 1

        data = []
        suffix_dataset_key = f"{dataset_key}_{self.step}"
        eval_loader = get_dataloader(
            data_pth=dataset_pth,
            shuffle=False,
            augment_target=False,
            batch_size=self.cfg.eval.batch_size,
        )
        logger.info(f"Generating suffix dataset {suffix_dataset_key}...")
        for batch in tqdm(eval_loader, desc=f"Generating suffix dataset {suffix_dataset_key}"):
            batch['instruct'] = batch['query']
            context = self.batch_to_context(batch)
            instruct = context.instruct
            target = context.target
            batch_data = []
            for max_new_tokens in self.cfg.eval.prompter.max_new_tokens_list:
                trial_data = []
                for trial in range(num_trials):
                    prompter_ar = self.prompter.generate_autoregressive(key="suffix", max_new_tokens=max_new_tokens, instruct=instruct)
                    suffix = prompter_ar.response_sample
                    full_instruct = MergedSeq(seqs=[instruct, suffix]).to_seq(merge_dtype="ids")
                    assert instruct.bs == target.bs == suffix.bs
                    datapoint = []
                    for i in range(instruct.bs):
                        datapoint.append((instruct.text[i], target.text[i], suffix.text[i], full_instruct.text[i]))
                    trial_data.append(datapoint)
                batch_data.append(trial_data)
            # aggregate
            for i in range(instruct.bs):
                for j in range(len(self.cfg.eval.prompter.max_new_tokens_list)):
                    for k in range(num_trials):
                        data.append(batch_data[j][k][i])
        suffix_dataset = argparse_namespace_like({
            "data": data,
            "fields": ["instruct", "target", "suffix", "full_instruct"],
            "suffix_dataset_key": suffix_dataset_key
        })
        return suffix_dataset

    @torch.no_grad()
    def save_suffix_dataset(self, suffix_dataset, dir):
        import pandas as pd
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        suffix_dataset_pth = os.path.join(dir, suffix_dataset.suffix_dataset_key + ".csv")
        logger.info(f" Saving {suffix_dataset.suffix_dataset_key} to {suffix_dataset_pth}")
        df = pd.DataFrame(suffix_dataset.data, columns=suffix_dataset.fields)
        df.to_csv(suffix_dataset_pth, index=False)
        return suffix_dataset_pth


# ---------- robust helper functions for optimizer setup ----------
from types import SimpleNamespace
from dataclasses import is_dataclass, asdict

def _try_cast_number(v):
    """把字符串数字尝试转成 int/float，否则返回原值（保留原类型）。"""
    if not isinstance(v, str):
        return v
    # 整数（只含数字）
    if v.isdigit():
        return int(v)
    # 尝试浮点 / 科学计数法
    try:
        f = float(v)
        return f
    except Exception:
        return v

def _to_primitive(obj):
    """
    把 SimpleNamespace / dataclass 转为 dict，递归处理 list/tuple/dict。
    如果已经是 dict/list/tuple/primitive，直接返回（但会对字符串数字尝试转换）。
    """
    # dataclass -> dict
    if is_dataclass(obj):
        return _to_primitive(asdict(obj))

    # SimpleNamespace -> dict
    if isinstance(obj, SimpleNamespace):
        return _to_primitive(vars(obj))

    # dict -> 递归转换 keys/values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _to_primitive(v)
        return out

    # list / tuple -> 转 list，递归
    if isinstance(obj, (list, tuple)):
        return [_to_primitive(x) for x in obj]

    # primitive (str/int/float/bool/None) -> 尝试把数字字符串转换
    return _try_cast_number(obj)

def _cast_optim_params(raw_params):
    """
    将任意可能的 config 类型（dict / SimpleNamespace / dataclass / nested lists）规范化为一个 dict，
    并把数值形式的字符串转换成数值。返回一个 dict（适合直接传给 torch.optim.Adam(..., **params)）。
    """
    # 把 namespace / dataclass / nested 类型 -> 原生 dict/list/primitive
    prim = _to_primitive(raw_params)

    # 如果结果是 dict，直接返回（keys 保持原样）
    if isinstance(prim, dict):
        return prim

    # 如果结果是 list/tuple，但我们需要 dict（例如用户把 optim param 写成列表？） -> 直接返回 {}
    # 也有可能 prim 是 None 或空 -> 返回 {}
    return {}

def _gather_model_parameters(prompter_obj):
    """
    尝试智能定位 prompter 对象中真实的 torch.nn.Parameter 迭代器。
    支持：prompter.parameters()、prompter.model.parameters()、prompter.prompter/model/peft_model 等常见字段。
    会返回一个 iterator（可交给 optimizer 使用），找不到时抛 RuntimeError。
    """
    # 1) 对象本身实现 parameters()
    if hasattr(prompter_obj, "parameters") and callable(getattr(prompter_obj, "parameters")):
        try:
            # 确保能迭代（即返回可迭代对象）
            params = list(prompter_obj.parameters())
            # 允许 0-length params （比如部分 wrapper），但返回 iterator 以保持行为一致
            return prompter_obj.parameters()
        except Exception:
            pass

    # 2) 常见 wrapper: .model
    if hasattr(prompter_obj, "model") and hasattr(prompter_obj.model, "parameters"):
        return prompter_obj.model.parameters()

    # 3) 其他常见属性名
    for attr in ("prompter", "prompter_model", "net", "module", "_model", "peft_model"):
        if hasattr(prompter_obj, attr):
            cand = getattr(prompter_obj, attr)
            if hasattr(cand, "parameters") and callable(getattr(cand, "parameters")):
                return cand.parameters()

    # 4) 未找到 -> raise
    raise RuntimeError(
        "Cannot find trainable parameters for prompter. "
        "Ensure 'self.prompter' is a trainable model (torch.nn.Module) or a wrapper exposing `.parameters()`.\n"
        f"Detected prompter type: {type(prompter_obj)}. Available attrs: {', '.join([a for a in dir(prompter_obj) if not a.startswith('_')])}"
    )

# -------------------------
# FinetuneWorkspace + AdvprompterWorkspace
# -------------------------
class FinetuneWorkspace(EvalWorkspace):
    def __init__(self, cfg: AdvPrompterConfig):
        super().__init__(cfg)
        self.mutator = None  # user should set AdvPrompterOpt or equivalent

    # def _init_train_components(self):
    #     self.prompter_optimizer = torch.optim.Adam(self.prompter.parameters(), **self.cfg.train.prompter_optim_params)
    #     sampler = PrioritizedSampler(max_capacity=self.cfg.train.replay_buffer.size, alpha=self.cfg.train.replay_buffer.priority_alpha, beta=1.0)
    #     self.replay_buffer = ReplayBuffer(storage=ListStorage(self.cfg.train.replay_buffer.size), batch_size=self.cfg.train.batch_size, sampler=sampler, collate_fn=collate_fn)

    def _init_train_components(self):
        # 1) 取并清洗 optimizer 参数（把字符串数字转为数值）
        # raw_optim_params = getattr(self.cfg.train, "prompter_optim_params", {}) or {}
        raw_optim_params = getattr(self.cfg, "train", None) and getattr(self.cfg.train, "prompter_optim_params", {}) or {}
        optim_params = _cast_optim_params(raw_optim_params)

        # 2) 从 self.prompter 中智能获取参数 iterator
        try:
            params_iter = _gather_model_parameters(self.prompter)
        except RuntimeError:
            logger.exception("Failed to gather prompter parameters for optimizer. Please ensure `self.prompter` exposes trainable parameters.")
            raise

        # 3) 创建 optimizer
        self.prompter_optimizer = torch.optim.Adam(params_iter, **optim_params)

        # 4) replay buffer & sampler as before
        sampler = PrioritizedSampler(
            max_capacity=self.cfg.train.replay_buffer.size,
            alpha=self.cfg.train.replay_buffer.priority_alpha,
            beta=1.0,
        )
        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(self.cfg.train.replay_buffer.size),
            batch_size=self.cfg.train.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    def add_to_replay_buffer(self, instruct, suffix, target, target_llm_tf, target_llm_tf_opt, target_llm_ar_opt, replay_buffer):
        loss_batch = target_llm_tf.loss_batch
        loss_opt_batch = target_llm_tf_opt.loss_batch
        priority = torch.relu(loss_batch - loss_opt_batch) * self.cfg.train.replay_buffer.priority_factor.loss_delta
        if self.cfg.train.replay_buffer.priority_factor.jailbreaking > 0:
            _, target_llm_ar_opt_jailbroken_list = check_jailbroken(seq=target_llm_ar_opt.response_sample, test_prefixes=self.test_prefixes)
            jailbroken = torch.tensor(target_llm_ar_opt_jailbroken_list, device=loss_batch.device)
            priority += jailbroken * self.cfg.train.replay_buffer.priority_factor.jailbreaking
        for i, prio in enumerate(priority):
            if prio > 0:
                datapoint = (instruct[i], target[i], suffix[i], priority[i])
                idx = self.replay_buffer.add(datapoint)
                self.replay_buffer.update_priority(index=idx, priority=prio.item())

    def finetune_prompter_step(self, instruct, suffix, prompter_optimizer, step=0):
        self.prompter_optimizer = prompter_optimizer
        self.step = step
        self.prompter_optimizer.zero_grad()
        prompter_tf_opt = self.prompter.compute_pred_loss_teacher_forced(key="suffix", instruct=instruct, suffix=suffix, loss_params=dict(hard_labels=True))
        loss = prompter_tf_opt.loss
        loss.backward()
        self.prompter_optimizer.step()
        if self.enable_wandb:
            import wandb
            wandb.log({"regression_loss": loss.item()}, step=self.step)
        return prompter_tf_opt

    def finetune_prompter_with_data_sampled_from_replay_buffer(self, prompter_optimizer, replay_buffer):
        if len(self.replay_buffer) < self.cfg.train.batch_size:
            logger.info("Replay buffer size is less than batch size, skipping finetuning step")
            return None
        if self.verbose:
            logger.info(f"Step: {self.step} | Sampling from replay buffer and finetuning prompter...")
        num_updates = min(self.cfg.train.replay_buffer.num_updates, len(self.replay_buffer) // self.cfg.train.batch_size)
        for _ in range(num_updates):
            context, priority_batch = self.replay_buffer.sample(batch_size=self.cfg.train.batch_size)
            prompter_tf_opt = self.finetune_prompter_step(instruct=context.instruct, suffix=context.suffix, prompter_optimizer=prompter_optimizer, step=self.step)
            if self.verbose:
                logger.info(f"Step: {self.step} | {_+1}/{num_updates} updates completed | Loss {prompter_tf_opt.loss:.3f}, Sample priorities {[p.item() for p in priority_batch]}")
        return prompter_tf_opt


class AdvprompterWorkspace(FinetuneWorkspace):
    def __init__(self, cfg: AdvPrompterConfig):
        super().__init__(cfg)
        # mutator should be provided from user code (AdvPrompterOpt). Keep placeholder.
        try:
            from mutation import AdvPrompterOpt
            self.mutator = AdvPrompterOpt()
        except Exception:
            self.mutator = None
            logger.warning("AdvPrompterOpt not found; set self.mutator before training.")

    def train(self):
        logger.info("Initializing optimizer and replay buffer...")
        self._init_train_components()
        if getattr(self.cfg, "train", None) and getattr(self.cfg.train, "do_initial_eval", False):
            logger.info("Doing initial eval before optionally pretraining and training...")
            self.eval()
        if getattr(self.cfg, "pretrain", None) and getattr(self.cfg.pretrain, "enable", False):
            logger.info("Starting pretraining...")
            self.pretrain()
            if self.cfg.train.model_save_dir is not None and self.cfg.train.save_pretrain:
                logger.info("Saving pretraining ckpts...")
                self.save_prompter()
        logger.info("Start training...")
        for epoch in range(self.cfg.train.epochs):
            logger.info(f"Epoch {epoch}/{self.cfg.train.epochs}")
            self.train_epoch(epoch=epoch)
            if self.cfg.train.eval_every is not None and (epoch + 1) % self.cfg.train.eval_every == 0 and (epoch + 1) < self.cfg.train.epochs:
                if self.cfg.train.model_save_dir is not None and self.cfg.train.always_save_before_eval:
                    self.save_prompter()
                self.eval()
        if self.cfg.train.model_save_dir is not None:
            self.save_prompter()
        self.eval()

    def pretrain(self):
        logger.info("Starting pretraining...")
        for pretrain_epoch in tqdm(range(self.cfg.pretrain.epochs), desc="Warmstarting (epochs)"):
            self.pretrain_epoch()
        if getattr(self.cfg.pretrain, "do_eval_after", False):
            self.eval()


# -------------------------
# Lightweight AdvPrompter helper (mutate-only)
# -------------------------
class AdvPrompter:
    def __init__(self, attacker_model_name: str, attacker_model_path: str, lora_checkpoint: Optional[str] = None, device: str = "cuda:0", dtype: str = "float32", verbose: bool = False):
        # build a minimal LLM-like Prompter object using load_model if possible
        self.verbose = verbose
        # Prefer to load prebuilt prompter via load_model if model descriptors are provided
        try:
            # If you have a LLM wrapper expecting the PrompterConfig, initialize it
            from models import LLM
            cfg = Prompter()
            self.prompter = LLM(cfg, verbose=self.verbose)
        except Exception:
            # fallback to load_model to obtain a model-like object (may not have generate_autoregressive)
            try:
                self.prompter = load_model(model_type="local", model_name=attacker_model_name, model_path=attacker_model_path, config=None)
            except Exception:
                raise RuntimeError("Cannot initialize prompter; please provide LLM wrapper or valid load_model params")

    def batch_to_context(self, batch):
        # keep same behaviour as workspace.batch_to_context for mutate-only usage
        model_map = dict(instruct=self.prompter)
        context = argparse_namespace_like({})
        try:
            from models import Seq
        except Exception:
            Seq = None
        for key, model in model_map.items():
            if key in batch.keys():
                if Seq is None:
                    class _S:
                        def __init__(self, text):
                            self.text = text
                            self.bs = len(text)
                    context.__dict__[key] = _S(batch[key])
                else:
                    context.__dict__[key] = Seq(text=batch[key], tokenizer=model.tokenizer, device=model.device)
            else:
                context.__dict__[key] = None
        return context

    def mutate(self, prompt: str, max_new_tokens: int = 50):
        batch = argparse_namespace_like({"instruct": [prompt]})
        context = self.batch_to_context({"instruct": [prompt]})
        instruct = context.instruct
        # prefer existing generate_autoregressive if wrapper provides it
        if hasattr(self.prompter, "generate_autoregressive"):
            prompter_ar = self.prompter.generate_autoregressive(key="suffix", max_new_tokens=max_new_tokens, instruct=instruct)
            suffix = prompter_ar.response_sample
            full_instruct = MergedSeq(seqs=[instruct, suffix]).to_seq(merge_dtype="ids")
            return full_instruct.text[0]
        else:
            # fallback: call model.chat or generate with load_model wrapper and return a string
            try:
                out = self.prompter.chat([{"role": "user", "content": prompt}])
                # normalize if dict/list
                def norm(x):
                    if isinstance(x, str):
                        return x
                    if isinstance(x, dict) and "choices" in x:
                        c = x["choices"][0]
                        if isinstance(c, dict) and "message" in c and "content" in c["message"]:
                            return c["message"]["content"]
                        if "text" in c:
                            return c["text"]
                    return str(x)
                return norm(out)
            except Exception:
                return prompt



from types import SimpleNamespace

def _try_cast_number(s):
    """如果是数字形式的字符串则转换为 int/float，否则保持原字符串。"""
    if not isinstance(s, str):
        return s
    # 整数
    if s.isdigit():
        return int(s)
    # 浮点、科学计数法等
    try:
        f = float(s)
        return f
    except Exception:
        return s

def dict_to_namespace(obj):
    """
    递归把 dict/list 转换为 SimpleNamespace/list，方便用点式访问。
    同时对字符串形式的数字做尝试性转换（'1e-4' -> 0.0001）。
    """
    if isinstance(obj, dict):
        ns = SimpleNamespace()
        for k, v in obj.items():
            # 合法的 python 属性名：若 key 含非法字符可保留原 key（但点访问会失败）
            safe_v = dict_to_namespace(v)
            setattr(ns, k, safe_v)
        return ns
    elif isinstance(obj, list):
        return [dict_to_namespace(x) for x in obj]
    else:
        # 尝试把数字字符串转换为数值
        return _try_cast_number(obj)

def main():
    try:
        args = parse_arguments()
        config_path = args.config_path or "./configs/advprompter.yaml"
        config_manager = ConfigManager(config_path=config_path)
        cfg = config_manager.config

        # 如果 config_manager 返回 dict -> 递归转为 namespace，方便用 adv_cfg.mode/adv_cfg.train.batch_size
        if isinstance(cfg, dict):
            adv_cfg = dict_to_namespace(cfg)
        else:
            # 如果 config_manager 已经返回 SimpleNamespace / dataclass / 自定义对象，则直接使用
            adv_cfg = cfg

        logger.info("Starting run...")
        logger.info(f"Using parameters: {adv_cfg}")

        # 下面按原逻辑，adv_cfg 已支持点式访问
        if getattr(adv_cfg, "mode", None) == "train":
            logger.info("Start training advprompter")
            workspace = AdvprompterWorkspace(adv_cfg)
            workspace.train()
        elif getattr(adv_cfg, "mode", None) == "eval":
            logger.info("Start evaluating advprompter, generating suffix using existing advprompter LLM")
            workspace = EvalWorkspace(adv_cfg)
            workspace.eval()
        elif getattr(adv_cfg, "mode", None) == "eval_suffix_dataset":
            logger.info("Start evaluating existing suffix datasets")
            workspace = EvalSuffixDatasetsWorkspace(adv_cfg)
            workspace.eval_suffix_datasets(adv_cfg.eval.suffix_dataset_pth_dct)
        else:
            raise ValueError(f"Mode {getattr(adv_cfg, 'mode', None)} not recognized.")
        logger.info("Finished!")
    except Exception as e:
        logger.exception("AdvPrompter run failed.")
        raise


if __name__ == "__main__":
    main()
