from typing import List

import requests
from mutation import BaseMutation

from transformers import MarianMTModel, MarianTokenizer
import torch

# class Translate(BaseMutation):
#     """
#     Translate is a class for translating the query to another language.
#     """
#     def __init__(self, attr_name='query', language='en'):
#         self.attr_name = attr_name
#         self.language = language
#         languages_supported = {
#             'en': 'English',
#             'zh-CN': 'Chinese',
#             'it': 'Italian',
#             'vi': 'Vietnamese',
#             'ar': 'Arabic',
#             'ko': 'Korean',
#             'th': 'Thai',
#             'bn': 'Bengali',
#             'sw': 'Swahili',
#             'jv': 'Javanese'
#         }
#         if self.language in languages_supported:
#             self.lang = languages_supported[self.language]
#         else:
#             raise ValueError(f"Unsupported language: {self.language}")

    
#     def translate(self, text, src_lang='auto'):
#         """
#         translate the text to another language
#         """
#         googleapis_url = 'https://translate.googleapis.com/translate_a/single'
#         url = '%s?client=gtx&sl=%s&tl=%s&dt=t&q=%s' % (googleapis_url,src_lang,self.language,text)
#         data = requests.get(url).json()
#         res = ''.join([s[0] for s in data[0]])
#         return res
    



class Translate(BaseMutation):
    """
    Translate is a class for translating the query to another language using a local MarianMT model.
    """
    def __init__(self, attr_name='query', language='en'):
        self.attr_name = attr_name
        self.language = language
        self.languages_supported = {
            'en': 'Helsinki-NLP/opus-mt-zh-en',      # 中文 → 英文
            'zh-CN': 'Helsinki-NLP/opus-mt-en-zh',   # 英文 → 中文
            'vi': 'Helsinki-NLP/opus-mt-en-vi',
            'ar': 'Helsinki-NLP/opus-mt-en-ar',
            # 'ko': 'Helsinki-NLP/opus-mt-en-ko',
            # 'th': 'Helsinki-NLP/opus-mt-en-th',
            'it': 'Helsinki-NLP/opus-mt-en-it'
        }

        # if self.language not in self.languages_supported:
        #     raise ValueError(f"Unsupported language: {self.language}")

        # model_name = self.languages_supported[self.language]
        # self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        # self.model = MarianMTModel.from_pretrained(model_name)
        if self.language not in self.languages_supported:
            warnings.warn(f"Warning: Unsupported language '{self.language}'. Translation model will not be loaded.")
            self.tokenizer = None
            self.model = None
        else:
            model_name = self.languages_supported[self.language]
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text, src_lang='auto'):
        """
        Translate the text to the target language using local MarianMT model.
        """
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = self.model.generate(**inputs)
        res = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        return res