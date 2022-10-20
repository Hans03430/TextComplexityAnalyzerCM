import statistics

from spacy.language import Language
from spacy.tokens import Doc
from time import time


class SyntacticComplexityIndices:
    '''
    This class will handle all operations to find the synthactic complexity indices of a text according to Coh-Metrix.
    '''

    name = 'syntactic_complexity_indices'
    
    def __init__(self, nlp: Language) -> None:
        '''
        The constructor will initialize this object that calculates the synthactic complexity indices for a specific language of those that are available.

        Parameters:
        nlp: The spacy model that corresponds to a language.
        
        Returns:
        None.
        '''
        required_pipes = ['noun_phrase_tagger', 'words_before_main_verb_counter']
        if not all((
            pipe in nlp.pipe_names
            for pipe in required_pipes
        )):
            message = 'Syntactic complexity indices pipe need the following pipes: ' + ', '.join(required_pipes)
            raise AttributeError(message)
        
        self._nlp = nlp
        Doc.set_extension('syntactic_complexity_indices', default={})

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will calculate the syntactic complexity indices.

        Parameters:
        doc(Doc): A Spacy document.

        Returns:
        Doc: The spacy document analyzed.
        '''
        if len(doc.text) == 0:
            raise ValueError('The text is empty.')

        print('Analyzing syntactic complexity indices')
        start = time()
        doc._.syntactic_complexity_indices['SYNNP'] = self.__get_mean_number_of_modifiers_per_noun_phrase(doc)
        doc._.syntactic_complexity_indices['SYNLE'] = self.__get_mean_number_of_words_before_main_verb(doc)
        end = time()
        print(f'Syntactic complexity indices analyzed in {end - start} seconds.')
        return doc

    def __get_mean_number_of_modifiers_per_noun_phrase(self, doc: Doc) -> float:
        '''
        This method calculates the mean number of modifiers per noun phrase in a text.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The mean of modifiers per noun phrases.
        '''
        return statistics.mean([np._.noun_phrase_modifiers_count for np in doc._.noun_phrases])

    def __get_mean_number_of_words_before_main_verb(self, doc: Doc) -> float:
        '''
        This method calculates the mean number of words before the main verb of sentences.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The mean of words before the main verb of sentences.
        '''
        return statistics.mean([sent._.count_of_words_before_main_verb for sent in doc._.non_empty_sentences])