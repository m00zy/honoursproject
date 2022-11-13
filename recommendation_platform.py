import pandas as pd
import numpy as np
from numpy import random
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
import os
import time

#THIS IS THE MAIN CODE
#FOR EVALUATIONS AND TESTING, PLEASE REFER TO THE FILE "EVALUATION_TESTS.PY)


class PrimeGenerator:
    #This class generates a replicatable sequence of prime numbers to be used as the random seeds of each epoch

    def __init__(self):
        #Initlalizer for Prime Generator Class
        self.current = 1

    def __iter__(self):
        #Return self
        return self

    @staticmethod
    def is_prime(n):
        #Returns True if input number is prime, otherwise False
        for i in range(2, n):
            if n % i == 0:
                return False
        return True

    #Returns the next prime number in the sequence
    def __next__(self):
        next_prime = self.current + 1
        while not self.is_prime(next_prime):
            next_prime += 1
        self.current = next_prime
        return self.current


class Preprocessor:
    #This class transforms a dataset into a database of profiles

    @staticmethod
    def get_data_from_filename(filename):
        #Takes in a filename, and reads the file to a pandas df
        file_path = os.path.join(os.path.dirname(__file__), filename)
        return pd.read_csv(file_path)

    @staticmethod
    def split_names_by_gender(names):
        #Takes in a file of names, and splits them into male names and female names
        return names["male names"], names["female names"]

    @staticmethod
    def get_relevant_data(data, columns):
        #Takes in a pandas df, and drops duplicate rows as well as rows with missing data
        data = data.drop_duplicates()
        return data[columns].dropna()

    @staticmethod
    def replace_values_with_dictionary(data, dictionaries, valid_input):
        #Takes in a pandas df and a dictionary of values, and replaces the relevant df columns with the values
        for dictionary in dictionaries:
            label = dictionary["attribute"]
            if label not in valid_input.get_categorical_inputs():
                raise ValueError(f"{label} is not a valid attribute!")
            dictionary.pop("attribute")
            data.replace({label: dictionary}, inplace=True)
        return data

    @staticmethod
    def convert_numerical_data_to_scaled_array(data, maximum=10):
        #Takes in a pandas df, and scales numerical attributes to a given range using MinMax
        normalizer = MinMaxScaler(feature_range=(0, maximum))
        return normalizer.fit_transform(data.to_numpy())

    @staticmethod
    def combine_attributes_and_names(names, numeric_attributes, english_attributes):
        #For each of the profiles in the two attribute arrays(numeric and english),
        #this function assigns to them a random name from the names dataset provided
        profile_numerical = np.hstack(([[None] for attribute in numeric_attributes], numeric_attributes))
        profile_english = np.hstack(([[None] for attribute in english_attributes], english_attributes))
        for i in range(0, np.size(profile_numerical, axis=0)):
            random_name = np.random.choice(names)
            profile_numerical[i][0] = random_name
            profile_english[i][0] = random_name
        return profile_numerical, profile_english


class DatingApp:
    #Mimics an interface for a real dating application
    #Selects profiles to recommend based on the user's behaviour
    def __init__(self, male_names, female_names, profiles_normalized, profiles_english):
        self.gender = ""
        self.male_names = male_names
        self.female_names = female_names
        self.profiles_normalized = profiles_normalized
        self.profiles_english = profiles_english
        self.recommendation_number = 1
        self.like_list = []
        self.dislike_list = []
        self.like_array_in_memory = None
        self.dislike_array_in_memory = None
        self.current_profile = None
        self.current_english = None
        self.random_recommendation = False
        self.recommended_profiles = []

    def set_gender(self, preprocessor, gender):
        #Takes in a specified gender, and assigns a gender-specific name to each profile in the database
        if gender == "Female":
            self.profiles_normalized, self.profiles_english = preprocessor.combine_attributes_and_names(self.male_names, self.profiles_normalized, self.profiles_english)

        elif gender == "Male":
            self.profiles_normalized, self.profiles_english = preprocessor.combine_attributes_and_names(self.female_names, self.profiles_normalized, self.profiles_english)
        else:
            raise ValueError("Invalid Gender Specified!")

    def get_a_recommendation(self, random_recommend_chance=5, top_choices=30, number_of_stored_profiles=30):
        #Calculates the best profile, and recommends it
        #There is a chance of a random_recommendation recommendation, which decreases as iterations increase
        self.recommendation_number += 1

        if self.recommendation_number <= 10:  # Random recommendations for the first ten recommendations
            rand_index = np.random.randint(0, np.size(self.profiles_normalized, axis=0))
            self.current_english = self.profiles_english[rand_index]
            self.current_profile = self.profiles_normalized[rand_index]
        else:
            if not self.like_list:  # Contingency in the case that no profiles have been liked after 10 recommendations
                dislike_array = np.array(self.dislike_list)
                worst_array = np.average(dislike_array[:, 1:], axis=0)
                self.like_list.append(np.array(["Sam", 10-worst_array[0], 10-worst_array[1], 10-worst_array[2], 10-worst_array[3], 10-worst_array[4]], dtype="O"))
            if not self.dislike_list: # Contingency in the case that no profiles have been disliked for 10 recommendations
                like_array = np.array(self.like_list)
                best_array = np.average(like_array[:, 1:], axis=0)
                self.dislike_list.append(np.array(["Sam", 10-best_array[0], 10-best_array[1], 10-best_array[2], 10-best_array[3], 10-best_array[4]], dtype="O"))

            self.like_array_in_memory = np.array(self.like_list)[-number_of_stored_profiles:] # Limit cap on how many profiles the algorithm uses to calculate average
            self.dislike_array_in_memory = np.array(self.dislike_list)[-number_of_stored_profiles:]

            random_recommend_chance = random_recommend_chance / (self.recommendation_number + random_recommend_chance) # Chance of a random recommendation

            best_array = np.average(self.like_array_in_memory[:, 1:], axis=0, weights=np.arange(1, np.size(self.like_array_in_memory[:, 1:], axis=0) + 1))  # Weighted average of the liked profiles
            worst_array = np.average(self.dislike_array_in_memory[:, 1:], axis=0, weights=np.arange(1, np.size(self.dislike_array_in_memory[:, 1:], axis=0) + 1))  # Weighed average of the disliked profiles
            profiles_normalized_enumerated = np.array(sorted(enumerate(self.profiles_normalized), key = lambda x: (np.linalg.norm(x[1][1:] - best_array) - (np.linalg.norm(x[1][1:] - worst_array)))), dtype="O") # Sorts according to distance to liked and disliked profiles
            indices = profiles_normalized_enumerated[:, 0]
            profiles_norm = profiles_normalized_enumerated[:, 1]
            index = np.random.choice(top_choices, 1)[0]
            self.profiles_english = np.array([self.profiles_english[i] for i in indices])
            self.profiles_normalized = profiles_norm

            random_number = np.random.random()
            if random_number > random_recommend_chance:
                self.current_profile = profiles_norm[index]
                self.current_english = self.profiles_english[index]
                self.random_recommendation = False
            else:  # Random recommendation
                random_index = np.random.choice(profiles_norm.shape[0], 1)[0]
                self.current_profile = profiles_norm[random_index]
                self.current_english = self.profiles_english[random_index]
                self.random_recommendation = True

        return self.current_profile, self.current_english

    def receive_like_or_dislike(self, preference):
        #Takes in either a "like" or "dislike", and appends the current profile to the appropriate list
        if preference == "like":
            self.like_list.append(self.current_profile)
        elif preference == "dislike":
            self.dislike_list.append(self.current_profile)


class ValidInputs:
    #Ensures that inputs passed into the system are valid

    def __init__(self):
        self.numerical_inputs = None
        self.categorical_inputs = None

    def set_limits(self, filename):
        #Takes in a filename, and sets the appropriate inputs
        file = open(filename, 'r')
        lines = file.readlines()
        file.close()

        categorical_inputs = lines[0].split(',')[1:]
        numerical_inputs = lines[1].split(',')[1:]

        self.numerical_inputs = numerical_inputs
        self.categorical_inputs = categorical_inputs

    def get_numerical_inputs(self):
        #Returns the specified valid numerical inputs
        return self.numerical_inputs

    def get_categorical_inputs(self):
        #Returns the specified valid categorical inputs
        return self.categorical_inputs


class Bot:
    #A model of a user
    #It has its own implicit preferences and target profile

    def __init__(self, preference_attributes, target_attributes, max_recommendations, number_of_stored_profiles, number_of_top_choices,
                 total_num_of_epochs, random_recommendation_weight, get_mean_recommendations, get_mean_distance, get_std_recommendations,
                 get_std_distance, get_time):
        self.num_of_recommendations_per_epoch = []
        self.current_num_of_epochs = 1
        self.ideal_arrays = []
        self.recommended_profiles = []
        self.max_recommendations = max_recommendations
        self.stored_profiles= number_of_stored_profiles
        self.top_choices = number_of_top_choices
        self.epochs = total_num_of_epochs
        self.random_recommendation = random_recommendation_weight
        self.get_mean_recommendations = get_mean_recommendations
        self.get_std_recommendations = get_std_recommendations
        self.get_mean_distance = get_mean_distance
        self.get_std_distance = get_std_distance
        self.get_time = get_time
        self.preference_attributes = preference_attributes
        self.target_attributes = target_attributes

    @staticmethod
    def preference(profile, preference_attributes):
        #Implicit preference

        if isinstance(preference_attributes, str):
            if preference_attributes == "Liker":
                return True
            elif preference_attributes == "Disliker":
                return False
            elif preference_attributes == "Random":
                if random.random() >= 0.5:
                    return True
                else:
                    return False

        age, drugs, drinks, smokes, height = profile[1], profile[2], profile[3], profile[4], profile[5]
        attributes = [age, drugs, drinks, smokes, height]
        preference = True
        for i in range(6):
            try:
                print(attributes[i], preference_attributes[i])
                if not attributes[i] >= preference_attributes[i]:
                    preference = False
            except IndexError:
                return preference

    @staticmethod
    def target_profile(profile, target_attributes):

        if not target_attributes:
            return False

        age, drugs, drinks, smokes, height = profile[1], profile[2], profile[3], profile[4], profile[5]
        attributes = [age, drugs, drinks, smokes, height]
        goal = True
        for i in range(6):
            try:
                if not attributes[i] >= target_attributes[i]:
                    goal = False
            except IndexError:
                return goal

    def run(self):
        #Runs the program
        gender = "Male"

        #Encodings of categorical data -> Numerical values
        attributes = ["age", "drugs", "drinks", "smokes", "height"]
        drugs_dictionary = {"attribute": "drugs", "never": 1, "sometimes": 2, "often": 3}
        drinks_dictionary = {"attribute": "drinks", "not at all": 1, "rarely": 2, "socially": 3, "often": 4, "very often": 5, "desperately": 6}
        smokes_dictionary = {"attribute": "smokes", "no": 1, "trying to quit": 2, "when drinking": 3, "sometimes": 4, "yes": 5}
        dictionary_list = [drugs_dictionary, drinks_dictionary, smokes_dictionary]

        #Valid Inputs
        valid_input = ValidInputs()
        valid_input.set_limits("ValidInputs")

        # Preprocessing
        preprocessor = Preprocessor()
        okcupid_data = preprocessor.get_data_from_filename("okcupid_profiles.csv")
        names_data = preprocessor.get_data_from_filename("baby_names.csv")
        okcupid_relevant_data = preprocessor.get_relevant_data(okcupid_data, attributes)
        okcupid_relevant_data= okcupid_relevant_data.drop_duplicates()
        profiles_english = okcupid_relevant_data.to_numpy()
        okcupid_relevant_data_numerical = preprocessor.replace_values_with_dictionary(okcupid_relevant_data, dictionary_list, valid_input)
        normalized_okcupid_data_array = preprocessor.convert_numerical_data_to_scaled_array(okcupid_relevant_data_numerical)
        male_names, female_names = preprocessor.split_names_by_gender(names_data)

        #Interacting with App
        interface = DatingApp(male_names, female_names, normalized_okcupid_data_array, profiles_english)
        interface.set_gender(preprocessor, gender)
        target_reached = False

        primes = PrimeGenerator()

        start = time.time()

        while self.current_num_of_epochs <= self.epochs: # Check to see if 30 epochs have been reached
            print(f"Epoch Number: {self.current_num_of_epochs}")
            np.random.seed(next(primes))  # Setting the seed of the epoch to be next prime number
            while interface.recommendation_number <= self.max_recommendations:
                if target_reached and not interface.random_recommendation and interface.recommendation_number > 10:  # Random recommendations cannot be target profiles
                    break

                print("Recommendation No.", interface.recommendation_number, sep="")

                profile, profile_english = interface.get_a_recommendation(self.random_recommendation, self.top_choices, self.stored_profiles)

                if interface.recommendation_number == 11:
                    self.ideal_arrays.append(profile[1:])

                print(profile_english)
                print(profile)

                target_reached = self.target_profile(profile, self.target_attributes)

                if self.preference(profile, self.preference_attributes):  # Passing a like or dislike into the face based on implicit preferences
                    interface.receive_like_or_dislike("like")
                    print("Like")
                else:
                    interface.receive_like_or_dislike("dislike")
                    print("Dislike")

            print("")
            print("Number of recommendations taken:", interface.recommendation_number-1)
            print("Final Recommendation: ", profile)
            print("")
            self.num_of_recommendations_per_epoch.append(interface.recommendation_number)
            self.recommended_profiles.append(profile[1:])
            self.current_num_of_epochs += 1
            interface = DatingApp(male_names, female_names, normalized_okcupid_data_array, profiles_english)
            interface.set_gender(preprocessor, gender)
            target_reached = False

        #Various metric outputs
        if self.get_mean_recommendations:
            print("Average Number of Recommendations Across Epochs:", np.average(self.num_of_recommendations_per_epoch))

        if self.get_std_recommendations:
            print("Average Standard Deviation Across Number of Recommendations:", np.std(self.num_of_recommendations_per_epoch))

        if self.get_time:
            print("Total Time Taken for All Epochs:", time.time()-start)

        if self.get_mean_distance:
            distances = [np.linalg.norm(self.recommended_profiles[i] - self.ideal_arrays[i]) for i in range(len(self.recommended_profiles))]
            print("Average Distance from Initial Implicit Preferences:", np.average(distances))

        if self.get_std_distance:
            distances = [np.linalg.norm(self.recommended_profiles[i] - self.ideal_arrays[i]) for i in range(len(self.recommended_profiles))]
            print("Average Standard Deviation Across Distance From Implicit Preferences", np.std(distances))

        print("")


