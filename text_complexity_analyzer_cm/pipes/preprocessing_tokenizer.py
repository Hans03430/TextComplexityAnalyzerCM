import re

from spacy.tokens import Doc
from typing import Callable

class PreprocessingTokenizer:
    '''Class that calls a function that preprocess a text before sending it to Spacy's default tokenizer.
    '''
    def __init__(self, tokenizer: object = None, preprocessing_func: Callable = lambda text: text) -> None:
        '''
        Creates the preprocessing tokenizer by receiving spacy's tokenizer.

        Parameters:
        tokenizer(Spacy.tokenizer.Tokenizer): Spacy tokenizer.
        
        Returns:
        None
        '''
        self._tokenizer = tokenizer
        self._preprocessing_func = preprocessing_func

    def __call__(self, text: str) -> Doc:
        '''When called, the class will preprocess a text and return a Spacy Doc.
        
        Paramters:
        text(str): The text to preprocess.
        
        Returns:
        Doc: The spacy document that is the text.
        '''
        clean_text = self._preprocessing_func(text)
        return self._tokenizer(clean_text)
