import multiprocessing
import pickle
import spacy
import time

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.constants import BASE_DIRECTORY
from text_complexity_analyzer_cm.pipes.preprocessing_tokenizer import PreprocessingTokenizer
from text_complexity_analyzer_cm.pipes.independent_factory import *
from text_complexity_analyzer_cm.pipes.spanish.factory import *

from typing import Callable, Dict
from typing import List


class TextComplexityAnalyzer:
    '''
    This class groups all of the indices in order to calculate them in one go. It works for a specific language.

    To use this class, instantiate an object with it. For example:
    tca = TextComplexityAnalyzer('es')

    Notice that a short version of the language was passed. The only languages available for now are: 'es'.

    To calculate the implemented coh-metrix indices for a text, do the following:
    m1, m2, m3, m4, m5, m6, m7, m8 = tca.calculate_all_indices_for_one_text(text='Example text', workers=-1)

    Here, all available cores will be used to analyze the text passed as parameter.

    To predict the category of a text, do the following:
    prediction = tca.predict_text_category(text='Example text', workers=-1)

    The example uses the default classifier stored along the library.
    '''
    def __init__(self, language:str = 'es', load_classifier=True, paragraph_delimiter: str='\n\n', preprocessing_func: Callable = lambda text:text) -> None:
        '''
        This constructor initializes the analizer for a specific language. It initializes all used pipes forthe analysis.

        Parameters:
        language(str): The language that the texts are in.
        load_classifier(bool): Flag to load the default prediction model or not.
        paragraph_delimiter(str): Separator to consider for the paragraphs.
        
        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')
        
        self.language = language
        self._nlp = spacy.load(ACCEPTED_LANGUAGES[language], exclude=['ner'])
        self._nlp.tokenizer = PreprocessingTokenizer(self._nlp.tokenizer, preprocessing_func)
        self._nlp.max_length = 3000000
        self._nlp.add_pipe('sentencizer')
        self._nlp.add_pipe('paragraphizer', config={'paragraph_delimiter': paragraph_delimiter})
        self._nlp.add_pipe('alphanumeric_word_identifier')
        self._nlp.add_pipe('syllablelizer', config={'language': language})
        self._nlp.add_pipe('descriptive_indices')
        self._nlp.add_pipe('content_word_identifier')
        self._nlp.add_pipe('lexical_diversity_indices')
        self._nlp.add_pipe('readability_indices')
        self._nlp.add_pipe('noun_phrase_tagger')
        self._nlp.add_pipe('words_before_main_verb_counter')
        self._nlp.add_pipe('syntactic_complexity_indices')
        self._nlp.add_pipe('verb_phrase_tagger')
        self._nlp.add_pipe('negative_expression_tagger')
        self._nlp.add_pipe('syntactic_pattern_density_indices')
        self._nlp.add_pipe('causal_connectives_tagger')
        self._nlp.add_pipe('logical_connectives_tagger')
        self._nlp.add_pipe('adversative_connectives_tagger')
        self._nlp.add_pipe('temporal_connectives_tagger')
        self._nlp.add_pipe('additive_connectives_tagger')
        self._nlp.add_pipe('connective_indices')
        self._nlp.add_pipe('cohesion_words_tokenizer')
        self._nlp.add_pipe('referential_cohesion_indices')
        self._nlp.add_pipe('informative_word_tagger')
        self._nlp.add_pipe('word_information_indices')
        self._nlp.add_pipe('wrapper_serializer', last=True)
        # Load default classifier if enabled
        if load_classifier:
            self.load_default_classifier()

        self._indices = ['CNCADC', 'CNCAdd', 'CNCAll', 'CNCCaus', 'CNCLogic', 'CNCTemp', 'CRFANP1', 'CRFANPa', 'CRFAO1', 'CRFAOa', 'CRFCWO1', 'CRFCWO1d', 'CRFCWOa', 'CRFCWOad', 'CRFNO1', 'CRFNOa', 'CRFSO1', 'CRFSOa', 'DESPC', 'DESPL', 'DESPLd', 'DESSC', 'DESSL', 'DESSLd', 'DESWC', 'DESWLlt', 'DESWLltd', 'DESWLsy', 'DESWLsyd', 'DRNEG', 'DRNP', 'DRVP', 'LDTTRa', 'LDTTRcw', 'RDFHGL', 'SYNLE', 'SYNNP', 'WRDADJ', 'WRDADV', 'WRDNOUN', 'WRDPRO', 'WRDPRP1p', 'WRDPRP1s', 'WRDPRP2p', 'WRDPRP2s', 'WRDPRP3p', 'WRDPRP3s', 'WRDVERB']

    def load_default_classifier(self) -> None:
        '''
        Method that loads the default classifier used to calculate the text complexity of new texts.
        '''
        self._classifier = pickle.load(open(f'{BASE_DIRECTORY}/model/classifier.pkl', 'rb'))
        self._scaler = pickle.load(open(f'{BASE_DIRECTORY}/model/scaler.pkl', 'rb'))

    def calculate_all_indices_for_texts(self, texts: List[str], workers: int=-1, batch_size: int=1) -> List[Dict]:
        '''
        This method calculates all indices for a list of texts using multiprocessing, if available, and stores them in a list of dictionaries.

        Parameters:
        texts(List[str]): The texts to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.
        batch_size(int): Amount of texts that each worker will analyze sequentially until no more texts are left.

        Returns:
        List[Dict]: A list with the dictionaries containing the indices for all texts sent for analysis.
        '''
        if workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0.')
        else:
            print('Analyzing texts.')
            start = time.time()
            threads = multiprocessing.cpu_count() if workers == -1 else workers  
            # Process all texts using multiprocessing
            metrics = [
                doc._.coh_metrix_indices
                for doc in self._nlp.pipe(texts, batch_size=batch_size, n_process=threads)
            ]
                
            end = time.time()
            print(f'Texts analyzed in {end - start} seconds.')
            return metrics


    def predict_text_category(self, text: str, workers: int=-1, classifier=None, scaler=None, indices: List=None) -> int:
        '''
        This method receives a text and predict its category based on the classification model trained.

        Parameters:
        text(str): The text to predict its category.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.
        classifier: Optional. A supervised learning model that implements the 'predict' method. If None, the default classifier is used.
        scaler: Optional. A object that implements the 'transform' method that scales the indices of the text to analyze. It must be the same as the one used in the classifier, if a scaler was used. Pass None if no scaler was used during the custom classifier's training.
        indices(List): Optional. Ignored if the default classifier is used. The name indices which the classifier was trained with. They must be in the same order as the ones that were used at training and also be the same. 

        Returns:
        int: The category of the text represented as a number
        '''
        if workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0.')
        if classifier is not None and not hasattr(classifier, 'predict'):
            raise ValueError('The custom surpervised learning model (classifier) must have the \'predict\' method.d')
        if classifier is not None and indices is None:
            raise ValueError('You must provide the names of the metrics used to train the custom classifier in the same order and amount that they were at the time of training said classifier.')
        if classifier is not None and scaler is not None and not hasattr(scaler, 'transform'):
            raise ValueError('The custom scaling model (scaler) for the custom classifier must have the \'transform\' method.')
        else:
            descriptive, word_information, syntactic_pattern, syntactic_complexity, connective, lexical_diversity, readability, referential_cohesion = self.calculate_all_indices_for_one_text(text, workers)
            metrics = {**descriptive, **word_information, **syntactic_pattern, **syntactic_complexity, **connective, **lexical_diversity, **readability, **referential_cohesion}
            if classifier is None: # Default indices
                indices_values = [[metrics[key] for key in self._indices]]
                # Check that the classifier was loaded
                if self._classifier is None:
                    raise AttributeError('The default classifier was not loaded when this object was created.')
                else:
                    return self._classifier.predict(self._scaler.transform(indices_values))

            else: # Indices used by the custom classifier
                indices_values = [[metrics[key] for key in indices]]
                    
                return list(classifier.predict(indices_values if scaler is None else scaler.transform(indices_values)))
