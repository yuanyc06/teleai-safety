"""
DeepInception Attack Method
============================================
This class implements a jailbreak method described in the paper below.

Paper title: DeepInception: Hypnotize Large Language Model to Be Jailbreaker
arXiv link: https://arxiv.org/pdf/2311.03191
Source repository: https://github.com/tmlr-group/DeepInception
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
from initialization import InitTemplates
from mutation import DeepInception

@dataclass
class DeepInceptionConfig:
    """Configuration class for DeepInception attack parameters."""
    attack_data_path: str
    target_model_path: str  
    target_model_name: str
    model_type: str = 'openai'  # openai, azure, local
    layer_num: int = 5
    character_num: int = 5
    scene: str = "science fiction"
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


class TemplateLoader:
    """Handles loading and management of DeepInception templates."""
    
    def __init__(self, config: DeepInceptionConfig) -> None:
        """Initialize template loader with configuration."""
        self.config = config
        self.template = self._load_template()
        logger.info("DeepInception template loaded successfully")
    
    def _load_template(self) -> str:
        """Load DeepInception template."""
        try:
            templates = InitTemplates().get_templates('DeepInception', 1)
            return templates[0] if templates else ""
        except Exception as e:
            logger.error(f"Failed to load template: {e}")
            raise

class MutationHandler:
    """Handles mutation operations for DeepInception attack."""
    
    def __init__(self, config: DeepInceptionConfig, template: str) -> None:
        """Initialize mutation handler with configuration and template."""
        self.config = config
        self.mutator = DeepInception(
            config.layer_num,
            config.character_num,
            config.scene,
            template,
            config.lang
        )
        logger.info("DeepInception mutator initialized")
    
    def mutate(self, query: str) -> str:
        """Apply DeepInception mutation to the query."""
        try:
            return self.mutator.apply_template(query)
        except Exception as e:
            logger.error(f"Mutation failed for query '{query}': {e}")
            return query  # Return original query if mutation fails


class PromptFormatter:
    """Formats prompts for the DeepInception attack."""
    
    @staticmethod
    def format_attack_prompt(query: str, template: str) -> str:
        """Format the query using DeepInception template."""
        # This will be handled by the mutator, keeping for consistency
        return query


class DeepInceptionManager(BaseAttackManager):
    """
    Main class for managing and executing DeepInception attacks.
    
    This attack method uses deep nested storytelling techniques to create adversarial prompts
    that can potentially bypass safety mechanisms in language models.
    """
    
    def __init__(
        self,
        attack_data_path: str,
        target_model_path: str,
        target_model_name: str,
        model_type: str = 'openai',
        layer_num: int = 5,
        character_num: int = 5,
        scene: str = "science fiction",
        lang: str = 'en',
        evaluator_type: str = 'pattern',
        evaluator_path: str = 'cais/HarmBench-Llama-2-13b-cls',
        system_prompt: str = 'You are a helpful assistant.',
        device: str = 'cpu',
        res_save_path: str = '../results/deepinception_openai.jsonl',
        log_path: str = '../logs/',
        template_path: str = "",
        api_key: str = "",
        base_url: str = "",
        azure_key: str = "",
        azure_url: str = "",
        grok_key: str = "",
        grok_url: str = ""
    ):
        """Initialize the DeepInception attack manager."""
        super().__init__(res_save_path)

        # Create configuration object
        self.config = DeepInceptionConfig(
            attack_data_path=attack_data_path,
            target_model_path=target_model_path,
            target_model_name=target_model_name,
            model_type=model_type,
            layer_num=layer_num,
            character_num=character_num,
            scene=scene,
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
        
        logger.info(f"DeepInception Manager initialized with {len(self.attack_dataset)} attack examples")
        logger.info(f"Using model type: {self.config.model_type}")
        logger.info(f"Target model: {self.config.target_model_name}")
    
    def _initialize_components(self) -> None:
        """Initialize all components needed for the attack."""
        # Load attack dataset
        attack_dataset = AttackDataset(self.config.attack_data_path)
        self.attack_dataset = attack_dataset
        
        # Initialize template loader
        self.template_loader = TemplateLoader(self.config)
        
        # Initialize local model if needed
        self.target_model = load_model(
            model_type=self.config.model_type,
            model_name=self.config.target_model_name,
            model_path=self.config.target_model_path,
            config=self.config
        )

        # Initialize mutation handler
        self.mutation_handler = MutationHandler(self.config, self.template_loader.template)

        # Initialize prompt formatter
        self.prompt_formatter = PromptFormatter()
    
    def generate_attack_prompt(self, raw_query: str) -> str:
        """Generate an attack prompt using DeepInception method."""
        return self.mutation_handler.mutate(raw_query)
    
    def mutate(self, query: str) -> str:
        """Generate a mutated version of the query using DeepInception method."""
        return self.generate_attack_prompt(query)
    
    def attack(self) -> None:
        """Execute the DeepInception attack on all examples in the dataset."""
        logger.info("Starting DeepInception attack...")
        
        for example_idx, example in enumerate(tqdm(self.attack_dataset, desc="Executing DeepInception attack")):
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
                    'model_name': self.config.target_model_name,
                    'layer_num': self.config.layer_num,
                    'character_num': self.config.character_num,
                    'scene': self.config.scene
                }
                
                self.log(result_data, save=True)
                
            except Exception as e:
                logger.error(f"Error processing example {example_idx}: {e}")
                continue
        
        logger.info("DeepInception attack completed!")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DeepInceptionManager':
        """Create DeepInceptionManager from configuration dictionary."""
        return cls(**config)


def main():
    """Main function to run the DeepInception attack."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        config_path = args.config_path or './configs/deepinception.yaml'
        
        # Load configuration
        config_manager = ConfigManager(config_path=config_path)
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Create and run attack manager
        attack_manager = DeepInceptionManager.from_config(config=config_manager.config)
        attack_manager.attack()
        
    except Exception as e:
        logger.error(f"Failed to run DeepInception attack: {e}")
        raise


if __name__ == "__main__":
    main()

