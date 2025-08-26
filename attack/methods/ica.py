"""
In-Context-Learning Attack Method
============================================
This class implements a jailbreak method described in the paper below.

Paper title: Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations
arXiv link: https://arxiv.org/pdf/2310.06387
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
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass

# Local imports
from dataset import AttackDataset
from utils import BaseAttackManager, ConfigManager, parse_arguments
from models import load_model, FORMATTER_REGISTRY, DefaultFormatter
from logger import setup_logger

@dataclass
class ICAConfig:
    """Configuration class for ICA attack parameters."""
    attack_data_path: str
    target_model_path: str  
    target_model_name: str
    model_type: str = 'openai'  # openai, azure, local
    demo_num: int = 1
    lang: str = 'en'
    system_prompt: str = 'You are a helpful assistant.'
    device: str = 'cpu'
    
    # API keys for different services
    api_key: str = ''  # OpenAI API key
    base_url: str = ''  # OpenAI base URL
    azure_key: str = ''  # Azure API key
    azure_url: str = ''  # Azure endpoint URL
    grok_key: str = ''  # Grok API key  
    grok_url: str = ''  # Grok endpoint URL
    
    evaluator_type: str = 'pattern'
    evaluator_path: str = 'cais/HarmBench-Llama-2-13b-cls'
    res_save_path: Optional[str] = None
    log_path: Optional[str] = None
    template_path: Optional[str] = None


class DemoLoader:
    """Handles loading and sampling of demonstration data."""
    
    def __init__(self, config: ICAConfig) -> None:
        """Initialize demo loader with configuration."""
        self.config = config
        self.demos = self._load_demos()
        logger.info(f"Loaded {len(self.demos)} demonstration examples")
    
    def _load_demos(self) -> List[Dict[str, Any]]:
        """Load demonstration data based on language configuration."""
        demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../utils/icl/")
        
        if self.config.lang == 'en':
            demo_file = os.path.join(demo_dir, "demo.json")
        elif self.config.lang == 'zh':
            demo_file = os.path.join(demo_dir, "data_zh.json")
        else:
            raise ValueError(f"Unsupported language: {self.config.lang}")
            
        try:
            with open(demo_file, "r", encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Demo file not found: {demo_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in demo file: {demo_file}")
            raise
    
    def sample_demos(self) -> List[Dict[str, Any]]:
        """Sample demonstration examples according to configured number."""
        if self.config.demo_num > len(self.demos):
            logger.warning(f"Requested {self.config.demo_num} demos but only {len(self.demos)} available")
            return self.demos.copy()
        
        return random.sample(self.demos, self.config.demo_num)


class PromptFormatter:
    """Formats prompts for the ICA attack."""
    
    @staticmethod
    def format_messages(demos: List[Dict[str, Any]], raw_query: str) -> str:
        """Format demonstration examples with the target query."""
        prompt_parts = []
        
        for demo in demos:
            query = demo.get("query", "")
            response = demo.get("unsafe_response", "")
            prompt_parts.append(f"User: {query}\n\nAssistant: {response}\n\n")
        
        prompt_parts.append(f"User: {raw_query}")
        return "".join(prompt_parts)
    
    
class ICAManager(BaseAttackManager):
    """
    Main class for managing and executing In-Context-Learning attacks.
    
    This attack method uses few demonstration examples to create adversarial prompts
    that can potentially bypass safety mechanisms in language models.
    """
    
    def __init__(
        self,
        attack_data_path: str,
        target_model_path: str,
        target_model_name: str,
        model_type: str = 'openai',
        demo_num: int = 1,
        lang: str = 'en',
        evaluator_type: str = 'pattern',
        evaluator_path: str = 'cais/HarmBench-Llama-2-13b-cls',
        system_prompt: str = 'You are a helpful assistant.',
        device: str = 'cpu',
        res_save_path: str = '../results/ica_openai.jsonl',
        log_path: str = '../logs/',
        template_path: str = "",
        api_key: str = "",
        base_url: str = "",
        azure_key: str = "",
        azure_url: str = "",
        grok_key: str = "",
        grok_url: str = ""
    ):
        """Initialize the ICA attack manager."""
        super().__init__(res_save_path)

        # Create configuration object
        self.config = ICAConfig(
            attack_data_path=attack_data_path,
            target_model_path=target_model_path,
            target_model_name=target_model_name,
            model_type=model_type,
            demo_num=demo_num,
            lang=lang,
            system_prompt=system_prompt,
            device=device,
            api_key=api_key,
            base_url=base_url,
            azure_key=azure_key,
            azure_url=azure_url,
            grok_key=grok_key,
            grok_url=grok_url,
            evaluator_type=evaluator_type,
            evaluator_path=evaluator_path,
            res_save_path=res_save_path,
            log_path=log_path,
            template_path=template_path
        )

        # Initialize components
        self._initialize_components()
        
        # Setup logging
        if self.config.log_path:
            setup_logger(log_file_path=self.config.log_path)
        
        logger.info(f"ICA Manager initialized with {len(self.attack_dataset)} attack examples")
        logger.info(f"Using model type: {self.config.model_type}")
        logger.info(f"Target model: {self.config.target_model_name}")
    
    def _initialize_components(self) -> None:
        """Initialize all components needed for the attack."""
        # Load attack dataset
        attack_dataset = AttackDataset(self.config.attack_data_path)
        self.attack_dataset = attack_dataset
        
        # Initialize demo loader
        self.demo_loader = DemoLoader(self.config)
        
        # Initialize local model if needed
        self.target_model = load_model(
            model_type=self.config.model_type,
            model_name=self.config.target_model_name,
            model_path=self.config.target_model_path,
            config=self.config
        )

        # Initialize prompt formatter
        self.prompt_formatter = PromptFormatter()
    
    
    def generate_attack_prompt(self, raw_query: str) -> str:
        """Generate an attack prompt using demonstration examples."""
        demos = self.demo_loader.sample_demos()
        return self.prompt_formatter.format_messages(demos, raw_query)
    
    def mutate(self, query: str) -> str:
        """Generate a mutated version of the query using ICA method."""
        return self.generate_attack_prompt(query)
    
    def attack(self) -> None:
        """Execute the ICA attack on all examples in the dataset."""
        logger.info("Starting ICA attack...")
        
        for example_idx, example in enumerate(tqdm(self.attack_dataset, desc="Executing ICA attack")):
            try:
                # Generate attack prompt
                raw_query = example.query
                attack_prompt = self.generate_attack_prompt(raw_query)
                
                # Get model response based on model type
                response = self.target_model.chat(attack_prompt)

                # Log results
                result_data = {
                    'example_idx': example_idx,
                    'query': raw_query,
                    'final_query': attack_prompt,
                    'response': response,
                    'model_type': self.config.model_type,
                    'model_name': self.config.target_model_name
                }
                
                self.log(result_data, save=True)
                
            except Exception as e:
                logger.error(f"Error processing example {example_idx}: {e}")
                continue
        
        logger.info("ICA attack completed!")


def main():
    """Main function to run the ICA attack."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        config_path = args.config_path or './configs/ica.yaml'
        
        # Load configuration
        config_manager = ConfigManager(config_path=config_path)
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Create and run attack manager
        attack_manager = ICAManager.from_config(config=config_manager.config)
        attack_manager.attack()
        
    except Exception as e:
        logger.error(f"Failed to run ICA attack: {e}")
        raise


if __name__ == "__main__":
    main()