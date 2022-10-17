from typing import Text
from text_complexity_analyzer_cm.text_complexity_analyzer import TextComplexityAnalyzer

if __name__ == '__main__':
    texts = ['El perro está caminando.']
    tca = TextComplexityAnalyzer('es', False)
    tca.calculate_all_indices_for_texts(texts, 1)
