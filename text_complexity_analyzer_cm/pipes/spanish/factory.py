from spacy.lang.es import Spanish
from spacy.language import Language

from text_complexity_analyzer_cm.pipes.auxiliaries.alphanumeric_word_identifier import AlphanumericWordIdentifier

@Spanish.factory('alphanumeric_word_identifier')
def create_es_alphanumeric_word_identifier(nlp: Language, name: str) -> AlphanumericWordIdentifier:
    '''
    Function that creates an alphanumeric word identifier for spanish.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    AlphanumericWordIdentifier: The pipe that finds alphanumeric words.
    '''
    return AlphanumericWordIdentifier(nlp)
