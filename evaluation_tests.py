from recommendation_platform import Bot
import time

######################################################################
# Experiment No 1

# Uncomment the following block of code to run Experiment 1

# max_recommendations = 2000
# number_of_stored_profiles = 30
# number_of_top_choices = 30  # Represents rate of overfitting
# total_num_of_epochs = 30
# outlier_weight = 5
# get_mean_recommendations = True
# get_mean_distance = False
# get_std_recommendations = True
# get_std_distance = False
# get_time = False
# preference_attributes = [3.21, 2.17, 3.84, 3.38, 7.17]
# target_attributes = [4.12, 5, 4, 5, 7.44]
#
# print("Experiment 1: Vague Bot")
# time.sleep(1)
# for i in range(1, 6):
#     print(f"Constraint(s) on {i} Attribute(s)")
#     time.sleep(1)
#     print()
#     my_bot = Bot(preference_attributes=preference_attributes[:i],
#                  target_attributes=target_attributes[:i],
#                  max_recommendations=max_recommendations,
#                  number_of_stored_profiles=number_of_stored_profiles,
#                  number_of_top_choices=number_of_top_choices,
#                  total_num_of_epochs=total_num_of_epochs,
#                  random_recommendation_weight=outlier_weight,
#                  get_mean_recommendations = get_mean_recommendations,
#                  get_mean_distance = get_mean_distance,
#                  get_std_recommendations=get_std_recommendations,
#                  get_std_distance=get_std_distance,
#                  get_time=get_time)
#
#     my_bot.run()
#     time.sleep(5)

######################################################################
#Experiment 2
#Uncomment the following block of code to run Experiment 2

# max_recommendations = 2000
# number_of_stored_profiles = 30
# number_of_top_choices = 30  # Represents rate of overfitting
# total_num_of_epochs = 30
# random_recommendation_weight = 5
# get_mean_recommendations = True
# get_mean_distance = False
# get_std_recommendations = True
# get_std_distance = False
# get_time = True
# preference_attributes = [4.12, 5, 4, 5, 7.44]
# target_attributes = [4.12, 5, 4, 5, 7.44]
#
# print("Experiment 2: Assertive User")
# time.sleep(1)
# for i in range(1, 6):
#     print(f"Constraint(s) on {i} Attribute(s)")
#     time.sleep(1)
#     print()
#     my_bot = Bot(preference_attributes=preference_attributes[:i],
#                  target_attributes=target_attributes[:i],
#                  max_recommendations=max_recommendations,
#                  number_of_stored_profiles=number_of_stored_profiles,
#                  number_of_top_choices=number_of_top_choices,
#                  total_num_of_epochs=total_num_of_epochs,
#                  random_recommendation_weight=random_recommendation_weight,
#                  get_mean_recommendations = get_mean_recommendations,
#                  get_mean_distance = get_mean_distance,
#                  get_std_recommendations=get_std_recommendations,
#                  get_std_distance=get_std_distance,
#                  get_time=get_time)
#
#     my_bot.run()
#     time.sleep(5)

######################################################################

# Experiment 3

# Uncomment the following block of code to run Experiment 3

# max_recommendations = [50, 100, 150, 200]
# number_of_stored_profiles = 30
# number_of_top_choices = 30  # Represents rate of overfitting
# total_num_of_epochs = 5
# random_recommendation_weight = 5
# get_mean_recommendations = False
# get_std_recommendations = False
# get_mean_distance = True
# get_std_distance = True
# get_time = False
# preference_attributes = ["Liker", "Disliker", "Random"]
# target_attributes = False
#
# print("Experiment 3")
# time.sleep(1)
# for num_recommendations in max_recommendations:
#     for personality in preference_attributes:
#         print(f"{personality} Personality for {num_recommendations} recommendations")
#         time.sleep(1)
#         my_bot = Bot(preference_attributes=personality,
#                      target_attributes=target_attributes,
#                      max_recommendations=num_recommendations,
#                      number_of_stored_profiles=number_of_stored_profiles,
#                      number_of_top_choices=number_of_top_choices,
#                      total_num_of_epochs=total_num_of_epochs,
#                      random_recommendation_weight=random_recommendation_weight,
#                      get_mean_recommendations=get_mean_recommendations,
#                      get_mean_distance=get_mean_distance,
#                      get_std_recommendations=get_std_recommendations,
#                      get_std_distance=get_std_distance,
#                      get_time=get_time)
#         my_bot.run()
#         time.sleep(5)
