##### Try to use map and reduce in the next 3 exercises
# 1)
# Create a function called "count_simba" that counts and returns
# the number of times that Simba appears in a list of
# strings. Example:
# ["Simba and Nala are lions.", "I laugh in the face of danger.",
#  "Hakuna matata", "Timon, Pumba and Simba are friends, but Simba could eat the other two."] 
#

# Define the function "count_simba" that accepts a list of strings
def count_simba(list_of_strings):
    # Initialize a counter to keep track of the number of occurrences of "Simba"
    simba_count = 0
    
    # Iterate over each string in the list
    for string in list_of_strings:
        # Use the .lower() method to make the search case-insensitive
        # Use the .count() method to count occurrences of "simba" in the string
        simba_count += string.lower().count("simba")
    
    # Return the total count of "Simba"
    return simba_count

# Example usage:
list_of_sentences = [
    "Simba and Nala are lions.",
    "I laugh in the face of danger.",
    "Hakuna matata",
    "Timon, Pumba and Simba are friends, but Simba could eat the other two."
]

# Call the function with the example list and print the result
result = count_simba(list_of_sentences)
print(f"The name 'Simba' appears {result} times.")



# 2)
# Create a function called "get_day_month_year" that takes 
# a list of datetimes.date and returns a pandas dataframe
# with 3 columns (day, month, year) in which each of the rows
# is an element of the input list and has as value its 
# day, month, and year.
# 

import pandas as pd

# Create a function called "get_day_month_year"
def get_day_month_year(date_list):
    # Create empty lists to store day, month, and year values
    days = []
    months = []
    years = []
    
    # Iterate over each date object in the input list
    for date in date_list:
        # Append the day, month, and year from the date object to the respective lists
        days.append(date.day)
        months.append(date.month)
        years.append(date.year)
    
    # Create a dictionary with the day, month, and year data
    data = {'day': days, 'month': months, 'year': years}
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Return the resulting DataFrame
    return df

# Example usage:
import datetime

# Create a list of datetime.date objects
dates = [
    datetime.date(2023, 5, 17),
    datetime.date(2022, 12, 25),
    datetime.date(2021, 7, 4)
]

# Call the function with the example list and print the resulting DataFrame
df = get_day_month_year(dates)
print(df)


# 3) 
# Create a function called "compute_distance" that takes
# a list of tuple pairs with latitude and longitude coordinates and 
# returns a list with the distance between the two pairs
# example input: [((41.23,23.5), (41.5, 23.4)), ((52.38, 20.1),(52.3, 17.8))]
# HINT: You can use geopy.distance in order to compute the distance
#

from geopy.distance import geodesic

# Define the function "compute_distance"
def compute_distance(coord_pairs):
    # Initialize an empty list to store the distances
    distances = []
    
    # Iterate over each tuple pair in the list
    for pair in coord_pairs:
        # Extract the two coordinates from each pair
        coord1, coord2 = pair
        
        # Calculate the distance using geodesic function from geopy.distance
        distance = geodesic(coord1, coord2).kilometers
        
        # Append the calculated distance to the distances list
        distances.append(distance)
    
    # Return the list of distances
    return distances

# Example usage:
coord_pairs = [((41.23, 23.5), (41.5, 23.4)), ((52.38, 20.1), (52.3, 17.8))]

# Call the function and print the result
distances = compute_distance(coord_pairs)
print(distances)


#################################################
# 4)
# Consider a list that each element can be an integer or
# a list that contains integers or more lists with integers
# example: [[2], 4, 5, [1, [2], [3, 5, [7,8]], 10], 1]. 
# create a recursive function called "sum_general_int_list"
# that takes as input this type of list 
# and returns the sum of all the integers within the lists
# for instance for list_1=[[2], 3, [[1,2],5]] 
# the result should be 13
#

# Define a recursive function "sum_general_int_list"
def sum_general_int_list(nested_list):
    # Initialize the total sum to 0
    total_sum = 0
    
    # Iterate over each element in the list
    for element in nested_list:
        # Check if the element is a list
        if isinstance(element, list):
            # Recursively call the function to sum the integers in the inner list
            total_sum += sum_general_int_list(element)
        else:
            # If the element is an integer, add it to the total sum
            total_sum += element
    
    # Return the total sum of integers
    return total_sum

# Example usage:
list_1 = [[2], 3, [[1, 2], 5]]
result = sum_general_int_list(list_1)
print(f"The sum of all integers in the list is: {result}")

