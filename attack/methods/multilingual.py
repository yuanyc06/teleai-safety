"""
Multilingual Attack Method
============================================
This class implements a multilingual jailbreak method that uses translation
to bypass safety mechanisms in language models.
"""

import os
import sys
import json
import time
import random
from typing import Optional, List, Dict, Any

# Add current working directory to path for relative imports
sys.path.append(os.getcwd())

# Third-party imports
import torch
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from dataset import AttackDataset
from utils import BaseAttackManager, ConfigManager, parse_arguments
from models import load_model, FORMATTER_REGISTRY, DefaultFormatter
from logger import setup_logger
from mutation import Translate

@dataclass
class MultilingualConfig:
    """Configuration class for Multilingual attack parameters."""
    attack_data_path: str
    target_model_path: str  
    target_model_name: str
    model_type: str = 'local'  # openai, azure, local
    mutation_method: str = 'zh-CN'
    system_prompt: str = 'You are a helpful assistant.'
    device: str = 'cuda'
    
    # API keys for different services
    api_key: str = ''  # OpenAI API key
    base_url: str = ''  # OpenAI base URL
    azure_key: str = ''  # Azure API key
    azure_url: str = ''  # Azure endpoint URL
    grok_key: str = ''  # Grok API key  
    grok_url: str = ''  # Grok endpoint URL
    
    res_save_path: Optional[str] = None
    log_path: Optional[str] = None


class ModelInitializer:
    """Handles model and tokenizer initialization."""
    
    def __init__(self, config: MultilingualConfig) -> None:
        """Initialize model components with configuration."""
        self.config = config
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        logger.info("Model and tokenizer initialized successfully")
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer based on configuration."""
        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.config.target_model_path,
                device_map="auto"
            ).eval()
            
            # Load tokenizer
            tokenizer = self._load_tokenizer()
            
            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load tokenizer with fallback mechanism."""
        tokenizer_path = self.config.target_model_path
        tokenizer_files = ["tokenizer_config.json", "tokenizer.model", "vocab.json", "merges.txt"]
        has_tokenizer = any(os.path.exists(os.path.join(tokenizer_path, f)) for f in tokenizer_files)

        if has_tokenizer:
            logger.info(f"Loading local tokenizer: {tokenizer_path}")
            return AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            logger.warning(f"Local tokenizer doesn't exist, downloading from HuggingFace: {self.config.target_model_name}")
            return AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-70B-Instruct",
                use_auth_token=""
            )


class TranslationHandler:
    """Handles multilingual translation mutations."""
    
    def __init__(self, config: MultilingualConfig) -> None:
        """Initialize translation handler with configuration."""
        self.config = config
        self.mutations = {
            # Chinese
            'zh-CN': Translate(language='zh-CN'),
            # Italian
            'it': Translate(language='it'),
            # Vietnamese
            'vi': Translate(language='vi'),
            # Arabic
            'ar': Translate(language='ar'),
            # # Korean
            # 'ko': Translate(language='ko'),
            # # Thai
            # 'th': Translate(language='th'),
            # # Bengali
            # 'bn': Translate(language='bn'),
            # # Swahili
            # 'sw': Translate(language='sw'),
            # # Javanese
            # 'jv': Translate(language='jv'),
        }
        
        if self.config.mutation_method not in self.mutations:
            raise ValueError(f"Unsupported mutation method: {self.config.mutation_method}")
            
        self.mutation = self.mutations[self.config.mutation_method]
        logger.info(f"Translation handler initialized for language: {self.config.mutation_method}")

    def mutate(self, query: str) -> List[str]:
        """Apply translation mutation to the query."""
        try:
            mutated_queries = self.mutation.translate(query)
            if isinstance(mutated_queries, str):
                mutated_queries = [mutated_queries]
            return mutated_queries
        except Exception as e:
            logger.error(f"Translation failed for query '{query}': {e}")
            return [query]  # Return original query if translation fails


class PromptFormatter:
    """Formats prompts for the multilingual attack."""
    
    @staticmethod
    def format_messages(target_model, system_prompt: str, raw_query: str) -> str:
        """Format the query with system prompt and conversation template."""
        # Check if model supports conversation interface
        if hasattr(target_model, "conversation") and hasattr(target_model.conversation, "append_message"):
            # Use conversation mechanism
            target_model.conversation.messages = []
            if system_prompt:
                target_model.conversation.set_system_message(system_prompt)
            conv_template = target_model.conversation
            conv_template.append_message(conv_template.roles[0], raw_query)
            
            # Check if conversation has get_prompt method (fastchat style)
            if hasattr(conv_template, "get_prompt"):
                prompt = conv_template.get_prompt()
                return prompt
            # Check if conversation has to_openai_api_messages method (OpenAI/Azure style)
            elif hasattr(conv_template, "to_openai_api_messages"):
                messages = conv_template.to_openai_api_messages()
                # Convert messages to a simple prompt format
                prompt_parts = []
                for msg in messages:
                    if msg['role'] == 'system':
                        prompt_parts.append(f"System: {msg['content']}")
                    elif msg['role'] == 'user':
                        prompt_parts.append(f"User: {msg['content']}")
                    elif msg['role'] == 'assistant':
                        prompt_parts.append(f"Assistant: {msg['content']}")
                prompt = "\n".join(prompt_parts)
                return prompt
            else:
                # Fallback if no known method exists
                if system_prompt:
                    return system_prompt + "\n" + raw_query
                return raw_query
        else:
            # Fallback: handle models without conversation interface
            if system_prompt:
                return system_prompt + "\n" + raw_query
            return raw_query


class MultilingualManager(BaseAttackManager):
    """
    Main class for managing and executing Multilingual attacks.
    
    This attack method uses translation to different languages to create adversarial prompts
    that can potentially bypass safety mechanisms in language models.
    """
    
    def __init__(
        self,
        attack_data_path: str = "thu-coai/AISafetyLab_Datasets/harmbench_standard",
        target_model_path: str = "lmsys/vicuna-7b-v1.5",
        target_model_name: str = "vicuna_1.5",
        model_type: str = 'local',
        mutation_method: str = "zh-CN",
        system_prompt: str = "",
        device: str = "cuda",
        evaluator_type: str = 'pattern',
        evaluator_path: str = 'cais/HarmBench-Llama-2-13b-cls',
        res_save_path: str = "results/multilingual.jsonl",
        log_path: str = '../logs/',
        api_key: str = "",
        base_url: str = "",
        azure_key: str = "",
        azure_url: str = "",
        grok_key: str = "",
        grok_url: str = "",
        *args,
        **kwargs
    ):
        """Initialize the Multilingual attack manager."""
        super().__init__(res_save_path)

        # Create configuration object
        self.config = MultilingualConfig(
            attack_data_path=attack_data_path,
            target_model_path=target_model_path,
            target_model_name=target_model_name,
            model_type=model_type,
            mutation_method=mutation_method,
            system_prompt=system_prompt,
            device=device,
            api_key=api_key,
            base_url=base_url,
            azure_key=azure_key,
            azure_url=azure_url,
            grok_key=grok_key,
            grok_url=grok_url,
            res_save_path=res_save_path,
            log_path=log_path
        )

        # Initialize components
        self._initialize_components()
        
        # Setup logging
        if self.config.log_path:
            setup_logger(log_file_path=self.config.log_path)
        
        logger.info(f"Multilingual Manager initialized with {len(self.attack_dataset)} attack examples")
        logger.info(f"Using model type: {self.config.model_type}")
        logger.info(f"Target model: {self.config.target_model_name}")
        logger.info(f"Translation language: {self.config.mutation_method}")
    
    def _initialize_components(self) -> None:
        """Initialize all components needed for the attack."""
        # Load attack dataset
        self.attack_dataset = AttackDataset(self.config.attack_data_path)
                
        # Initialize target model
        self.target_model = load_model(
            model_type=self.config.model_type,
            model_name=self.config.target_model_name,
            model_path=self.config.target_model_path,
            config=self.config
        )
        
        # Initialize translation handler
        self.translation_handler = TranslationHandler(self.config)
        
        # Initialize prompt formatter
        self.prompt_formatter = PromptFormatter()

    def generate_attack_prompt(self, raw_query: str) -> List[str]:
        """Generate attack prompts using translation."""
        return self.translation_handler.mutate(raw_query)
    
    def mutate(self, query: str) -> List[str]:
        """Generate mutated versions of the query using translation."""
        return self.generate_attack_prompt(query)
    
    def format_msgs(self, raw_query: str) -> str:
        """Format messages using the prompt formatter."""
        return self.prompt_formatter.format_messages(
            self.target_model, 
            self.config.system_prompt, 
            raw_query
        )
    
    def attack(self) -> None:
        """Execute the Multilingual attack on all examples in the dataset."""
        logger.info("Starting Multilingual attack...")
        
        for example_idx, example in enumerate(tqdm(self.attack_dataset, desc="Executing Multilingual attack")):
            try:
                # Generate attack prompts
                raw_query = example.query
                mutated_queries = self.generate_attack_prompt(raw_query)
                
                for mutated_query in mutated_queries:
                    # Format the prompt
                    final_input = self.format_msgs(mutated_query)
                    
                    # Tokenize input
                    response = self.target_model.chat(final_input)
                    
                    # Log results
                    result_data = {
                        'example_idx': example_idx,
                        'query': raw_query,
                        'final_query': mutated_query,
                        'response': response,
                        'model_type': self.config.model_type,
                        'model_name': self.config.target_model_name,
                        'mutation_method': self.config.mutation_method
                    }
                    
                    self.log(result_data, save=True)
                    
            except Exception as e:
                logger.error(f"Error processing example {example_idx}: {e}")
                continue
        
        logger.info("Multilingual attack completed!")

    @classmethod
    def from_config(cls, config: Dict[str, Any], mutation_method: Optional[str] = None) -> 'MultilingualManager':
        """Create MultilingualManager from configuration dictionary."""
        config = config.copy()  # Avoid modifying original config
        if mutation_method is not None:
            config["mutation_method"] = mutation_method
        return cls(**config)


def main():
    """Main function to run the Multilingual attack."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        config_path = args.config_path or './configs/multilingual.yaml'
        
        # Load configuration
        config_manager = ConfigManager(config_path=config_path)
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Create and run attack manager
        attack_manager = MultilingualManager.from_config(config=config_manager.config)
        attack_manager.attack()
        
    except Exception as e:
        logger.error(f"Failed to run Multilingual attack: {e}")
        raise


if __name__ == "__main__":
    main()
