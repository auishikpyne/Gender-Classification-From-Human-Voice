import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")

class GenderIdentifier:
    def __init__(self, file_path, females_model_path, males_model_path):
        self.file_path = file_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        # load models
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        
    def process(self):
        file = self.file_path
        # read the test directory and get the list of test audio files
        
        self.total_sample += 1
        print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

        vector = self.features_extractor.extract_features(file)
        winner = self.identify_gender(vector)
        
        print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

    def identify_gender(self, vector):
        # female hypothesis scoring
        is_female_scores         = np.array(self.females_gmm.score(vector))
        is_female_log_likelihood = is_female_scores.sum()
        # male hypothesis scoring
        is_male_scores         = np.array(self.males_gmm.score(vector))
        is_male_log_likelihood = is_male_scores.sum()

        print("%10s %5s %1s" % ("+ FEMALE SCORE",":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_log_likelihood,3))))

        if is_male_log_likelihood > is_female_log_likelihood: winner = "male"
        else                                                : winner = "female"
        return winner
    
if __name__== "__main__":
    gender_identifier = GenderIdentifier("/home/auishik/gender_classification/tts.wav", "females.gmm", "males.gmm")
    gender_identifier.process()
