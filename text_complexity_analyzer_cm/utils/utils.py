import spacy

from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens import Token
from typing import List

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES


def split_text_into_paragraphs(text: str) -> List[str]:
    """
    This function splits a text into paragraphs. It assumes paragraphs are separated by two line breaks.

    Parameters:
    text(str): The text to be split into paragraphs.

    Returns:
    List[str]: A list of paragraphs.
    """
    text_aux = text.strip()
    paragraphs = text_aux.split('\n\n') # Strip any leading whitespaces

    for p in paragraphs:
        p = p.strip()

    return [p.strip() for p in paragraphs if len(p) > 0] # Don't count empty paragraphs


def split_text_into_sentences(text: str, language: str='es') -> List[str]:
    """
    This function splits a text into sentences.

    Parameters:
    text(str): The text to be split into sentences.
    language(str): The language of the text.

    Returns:
    List[str]: A list of sentences.
    """
    if not language in ACCEPTED_LANGUAGES[language]:
        raise ValueError(f'Language {language} is not supported yet')

    nlp = spacy.load(language, disable=['tagger', 'parser', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    text_spacy = nlp(text)
    return [str(sentence) for sentence in text_spacy.sents]


def is_content_word(token: Token) -> bool:
    '''
    This function checks if a token is a content word: Substantive, verb, adverb or adjective.

    Parameters:
    token(Token): A Spacy token to analyze.

    Returns:
    bool: True or false.
    '''
    return token.is_alpha and token.pos_ in ['PROPN', 'NOUN', 'VERB', 'ADJ', 'ADV', 'AUX']


def is_word(token: Token) -> bool:
    '''
    This function checks if a token is a word. All characters will be alphabetic.

    Parameters:
    token(Token): A Spacy token to analyze.

    Returns:
    bool: True or false.
    '''
    return token.is_alpha

def split_doc_into_sentences(doc: Doc) -> List[Span]:
    """
    This function splits a text into sentences.

    Parameters:
    text(str): The text to be split into sentences.

    Returns:
    List[Span]: A list of sentences represented by spacy spans.
    """
    return [s
            for s in doc.sents
            if len(s.text.strip()) > 0]