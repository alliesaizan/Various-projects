
# This file performs the following tasks:
#    1. Collects chocolate chip cookie recipes from the Internet 
#    2. Pulls out the 6 most common ingredients and standardizes them 
#    3. Calculates an average recipe, based on the ingredients list
#    4. Visualizes the distribution of ingredients over recipes
#       Ideas: boxplot

__author__ = 'A SAIZAN'

# Import modules

import requests
import bs4
import pandas as pd
import re
import json
from nltk import RegexpTokenizer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
import pickle


##################################################################################
# Scrape the Food Network database and retrieve URL links to chocolate chip cookie recipes

cookieLinks = []
for i in range(1, 57):
    if i < 2:
        url = "http://www.foodnetwork.com/search/chocolate-chip-cookie-"
    else:        
        url = "http://www.foodnetwork.com/search/chocolate-chip-cookie-/p/" + str(i) 
    res = requests.get(url, verify=False)
    soupText= bs4.BeautifulSoup(res.text, "html.parser")
    selected = soupText.find_all('a')
    for link in selected:
        if "cookie" in link.get('href').lower() and "facet" not in link.get('href').lower() and "search" not in link.get('href').lower():
            cookieLinks.append(link.get('href'))        

del i, url


# Remove duplicate links
cookieLinks = set(cookieLinks)
len(cookieLinks) # Total number of recipes

# Treat videos and recipes separately
cookieRecipeLinks = [i for i in cookieLinks if "com/recipes" in i]
cookieVideoLinks = [i for i in cookieLinks if "com/videos" in i]


# Pull out the metadata for each recipe, including ingredients, instructions, and amount
cookieRecipes = pd.DataFrame(columns = ["url","name", "amount","ingredients", "instructions", "rating"])

for link in cookieRecipeLinks:
    res = requests.get(link, verify=False)
    soupText= bs4.BeautifulSoup(res.text, "html.parser")
    js = json.loads(soupText.find("script", type="application/ld+json").text)
    if "name" in js.keys():
        name = js['name']
    else:
        name = ""
    if "recipeYield" in js.keys():
        amount = js['recipeYield']
    else:
        amount = ""
    if "recipeIngredient" in js.keys():
        ingredients = js['recipeIngredient']
    else:
        ingredients = ""
    if "recipeInstructions" in js.keys():
        instructions = js['recipeInstructions']
    else:
        instructions = ""
    if "review" in js.keys():
        review = js['review']
    else:
        review = ""
    cookieRecipes = cookieRecipes.append([{"url": link, 'name': name, 'amount': amount,
        'ingredients': ingredients, 'rating': review, 
        'instructions': instructions}], ignore_index=True)    

del name, amount, ingredients, review, instructions, js, link



# Drop duplicates
cookieRecipes = cookieRecipes.drop_duplicates(subset="name")

# Pull out ingredients from data frame
ingredients = ""
for i in cookieRecipes.ingredients.apply("".join).tolist():
    ingredients += i.lower()

text = RegexpTokenizer(r'\w+').tokenize(ingredients)
text = [x for x in text if x.isdigit() == False]
len(text)




# Save the data frame so we don't have to reload it every time we reopen the file
pickle.dump(cookieRecipes, open("/Users/alliesaizan/Documents/Python Tinkering/cookies.py", "wb"))
cookieRecipes = pickle.load(open("/Users/alliesaizan/Documents/Python Tinkering/cookies.py", "rb"))




##################################################################################
# Pull out the individual ingredients


# Defining functions we will use in the string cleaning process
def formatter(x):
    if len(x)==2:
        if "/" in x[0] and "/" in x[1]:
            return x[0]
        elif "/" not in x[1]:
            return x[0]
    else:
        return x
    
def split_up_words(x):
    if x:
        return " ".join(re.split('([A-Za-z]+)', x))
    else:
        return ""
        
def Checker(x):
    if len(x) ==2:
        if "/" not in x[1]:
            return [x[0]]
        else:
            return  x
    else:
        return x

# Chocolate chips

    # Determine the most common measurement for chocolate chips
chocolateChips = cookieRecipes.ingredients.apply(lambda x: 
    [i for i in x if "chocolate" in i.lower()]).tolist()

chocolateChips_str = ""

for i in chocolateChips:
    for j in i:
        chocolateChips_str += j

chocolateChips_str = RegexpTokenizer(r'\w+').tokenize(chocolateChips_str)
chocolateChips_str = set(chocolateChips_str)
chocolateChips_str = [word.lower() for word in chocolateChips_str]
chocolateChips_str = [word for word in chocolateChips_str if word not in stopwords.words()]
FreqDist(chocolateChips_str).most_common()

    # Pull out the ingredients and measurements
commonMeasurements = ["ounce", "ounces",  "bag", "bags","grams", "pkg", "cup", "teaspoon", "oz", "bars", "bar"]
cookieRecipes['chocolate_chips'] = cookieRecipes.ingredients.apply(lambda x: 
    [i for i in x if "chocolate" in i.lower() and "chips" in i.lower()]).apply("".join).apply(lambda x: 
        split_up_words(x)).apply(word_tokenize)
cookieRecipes['chocolate_chipsAmt'] = cookieRecipes.chocolate_chips.apply(lambda x: 
    [i for i in x if i.isdigit()==True or "/" in i or "One" in i or "Two" in i or "-" in i]).apply(lambda x:
        x[0:2] if len(x) >= 1 else "")
cookieRecipes['chocolate_chipsMeasurement'] = cookieRecipes.chocolate_chips.apply(lambda x: 
    [i for i in x if i in commonMeasurements])
cookieRecipes[["chocolate_chips", "chocolate_chipsAmt", "chocolate_chipsMeasurement"]]


# Vanilla
cookieRecipes['vanilla'] = cookieRecipes.ingredients.apply(lambda x: 
    [i for i in x if "vanilla extract" in i.lower()]).apply(
        "".join).str.replace("-", " ")
cookieRecipes['vanilla']= cookieRecipes['vanilla'].apply(lambda x: split_up_words(x)).apply(word_tokenize)
cookieRecipes['vanillaMeasurement'] = cookieRecipes.vanilla.apply(lambda 
    x: [i for i in x if i.isdigit()==False and i.isalpha()==True][0] if len(x) > 0 else "")
cookieRecipes['vanillaAmt'] = cookieRecipes.vanilla.apply(lambda x: 
    [i for i in x if i.isdigit()==True or "/" in i or "One" in i]).apply(lambda x:
        x[0:2] if len(x) >= 1 else "")
#cookieRecipes['vanillaAmt'] = cookieRecipes.vanillaAmt.apply(lambda x: " ".join([i for i in x if i.isalpha()==False])).str.replace("-", " ")


# Brown Sugar
cookieRecipes['brownsugar'] = cookieRecipes.ingredients.apply(lambda x: 
    [i for i in x if "brown sugar" in i.lower()]).apply("".join).apply(
        lambda x: split_up_words(x)).apply(word_tokenize)
cookieRecipes['brownsugarAmt'] = cookieRecipes.brownsugar.apply(lambda x: 
    [i for i in x if i.isdigit()==True or "/" in i]).apply(lambda x:
        x[0:2] if len(x) >= 1 else "").apply(Checker)
#cookieRecipes['brownsugarAmt'] = cookieRecipes.brownsugarAmt.apply(lambda x: " ".join([i for i in x if i.isalpha()==False]))
cookieRecipes['brownsugarMeasurement'] = cookieRecipes.brownsugar.apply(lambda 
    x: [i for i in x if i.isdigit()==False and i.isalpha()==True and i != "Recipe"][0] if len(x) > 0 else "")


# Granulated sugar
cookieRecipes['granulated_sugar'] = cookieRecipes.ingredients.apply(lambda x: 
    [i for i in x if "granulated sugar" in i.lower()]).apply("".join).apply(
        lambda x: split_up_words(x)).apply(word_tokenize)
cookieRecipes['granulated_sugarAmt'] = cookieRecipes.granulated_sugar.apply(lambda x: 
    [i for i in x if i.isdigit()==True or "/" in i]).apply(lambda x:
        x[0:2] if len(x) >= 1 else "")
#cookieRecipes['granulated_sugarAmt'] = cookieRecipes.granulated_sugarAmt.apply(lambda x: " ".join([i for i in x if i.isalpha()==False]))
cookieRecipes['granulated_sugarMeasurement'] = cookieRecipes.granulated_sugar.apply(lambda 
    x: [i for i in x if i.isdigit()==False and i.isalpha()==True and i != "Recipe"][0] if len(x) > 0 else "")
cookieRecipes[["granulated_sugar", "granulated_sugarAmt", "granulated_sugarMeasurement"]].tail()


# Butter
cookieRecipes['butter'] = cookieRecipes.ingredients.apply(lambda x: [i for i in
             x if "butter" in i.lower() and "peanut" not in i.lower()]).apply("".join).apply(lambda x: 
    split_up_words(x)).apply(word_tokenize)
cookieRecipes['butterAmt'] = cookieRecipes.butter.apply(lambda x: 
    [i for i in x if i.isdigit()==True or "/" in i or "One" in i]).apply(lambda x:
        x[0:2] if len(x) >= 1 else "").apply(Checker)
#cookieRecipes['butterAmt'] = cookieRecipes['butterAmt'].apply(lambda x: " ".join([i for i in x if i.isalpha()==False]))
cookieRecipes['butterMeasurement'] = cookieRecipes.butter.apply(lambda 
    x: [i for i in x if i.isalpha()==True and 
        "Softened" not in i][0] if len(x) > 0 else "").str.replace("Butter", "tablespoons")

    
# Baking soda
cookieRecipes['baking_soda'] = cookieRecipes.ingredients.apply(lambda x: 
    [i for i in x if "baking soda" in i.lower()]).apply(
        "".join).apply(lambda x: split_up_words(x)).apply(word_tokenize)
cookieRecipes['bsAmt'] = cookieRecipes.baking_soda.apply(lambda x: 
    [i for i in x if i.isdigit()==True or "/" in i]).apply(lambda x:
        x[0:2] if len(x) >= 1 else "")
#cookieRecipes['bsAmt'] = cookieRecipes.bsAmt.apply(lambda x: " ".join([i for i in x if i.isalpha()==False]))
cookieRecipes['bsMeasurement'] = cookieRecipes.baking_soda.apply(lambda 
    x: [i for i in x if i.isalpha()==True and "Heaping" not in i and "Rounded" not in i][0] if len(x) > 0 else "")
  

# Flour
def tbs2Cups(x):
    if len(x)==2:
        if "/" not in x[1]:
            return [x[0], str(int(x[1])/ 16)]
        else:
            return x
    else:
        return x
cookieRecipes['flour'] = cookieRecipes.ingredients.apply(lambda x: 
    [i for i in x if "flour" in i.lower()]).apply("".join).apply(lambda x: 
        split_up_words(x)).apply(word_tokenize)
cookieRecipes['flourAmt'] = cookieRecipes.flour.apply(lambda x: 
    [i for i in x if i.isdigit()==True or "/" in i]).apply(lambda x:
        x[0:2] if len(x) >= 1 else "").apply(tbs2Cups)
#cookieRecipes['flourAmt'] = cookieRecipes.flourAmt.apply(lambda x: " ".join([i for i in x if i.isalpha()==False and i != "435"]))
cookieRecipes['flourMeasurement'] = cookieRecipes.flour.apply(lambda x: 
    [i for i in x if i.isalpha()==True and "/" not in i and "heaping" not in i.lower()][0] 
    if len(x) > 0 else "")
cookieRecipes['flourMeasurement'] = cookieRecipes.flourMeasurement.str.replace("scant", "cups")


# Eggs
def eggstoInt(x):
    if x == "" :
        return 0
    elif "/" not in x:
        return int(x)
    elif "/" in x:
        return int(x[0]) / int(x[2])

cookieRecipes['eggs'] = cookieRecipes.ingredients.apply(lambda x: 
    [i for i in x if "eggs" in i.lower()]).apply("".join).apply(
            lambda x: split_up_words(x)).apply(word_tokenize)
cookieRecipes['eggsAmt'] = cookieRecipes.eggs.apply(lambda x: 
    [i for i in x if i.isdigit()==True or "/" in i]).apply(lambda x: 
        x[0] if len(x) >= 1 else "")
cookieRecipes['eggsNum'] = cookieRecipes.eggsAmt.apply(eggstoInt)
cookieRecipes['eggsMeasurement'] = cookieRecipes.eggs.apply(lambda 
    x: [i for i in x if i=="eggs" or i == "cups"][0] if len(x) > 0 else "")


##################################################################################
# Standardize and aggregate ingredients

# Turn the list of amounts into integers
def strToInt(x):
    num = 0
    for i in x:
        if i == "" :
           return 0
        elif "." in i:
            num += float(i)
        elif i=="One":
            num +=1 
        elif i=="Two":
            num +=2
        elif "/" not in i:
            num += int(i)
        elif "/" in i:
            try:
                num += int(i[0]) / int(i[2])
            except:
                num += int(i[1]) / int(i[3])
    return num

cookieRecipes['chocolate_chipsNum'] = cookieRecipes.chocolate_chipsAmt.apply(strToInt)
cookieRecipes['vanillaNum'] = cookieRecipes.vanillaAmt.apply(strToInt)
cookieRecipes['brownsugarNum'] = cookieRecipes.brownsugarAmt.apply(strToInt)
cookieRecipes['granulated_sugarNum'] = cookieRecipes.granulated_sugarAmt.apply(strToInt)
cookieRecipes['bsNum'] = cookieRecipes.bsAmt.apply(strToInt)
cookieRecipes['flourNum'] = cookieRecipes.flourAmt.apply(strToInt)


# Convert the ingredient amounts into normalized amount measurements:
 # tablespoons (tablespoon, tbsp)
 # teaspoons (teaspoon, tsp)
 # cup (cups)
 
def standardizer(x, y):
    # The standard measurement I am moving to is tablespoons
    if re.search('tbsp|tablespoon', y) != None:
        return x
    elif re.search('tsp|teaspoon', y) != None:
        return x * 0.333333
    elif re.search('cups', y) != None:
        return x / 16
    else:
        return 0

cookieRecipes.reset_index(inplace=True)

cookieRecipes[['chocolate_chips','chocolate_chipsMeasurement', 'chocolate_chipsAmt', 'chocolate_chipsNum']]

cookieRecipes['chocolate_chips_standardized'] = cookieRecipes.apply(
        lambda row: standardizer(cookieRecipes['chocolate_chipsNum'], 
            cookieRecipes['chocolate_chipsMeasurement']), axis=1)

cookieRecipes['vanilla_standardized'] = cookieRecipes.apply(
        lambda row: standardizer(row['vanillaNum'], 
        row['vanillaMeasurement']), axis=1)
cookieRecipes[['vanilla','vanillaMeasurement', 'vanillaAmt', 'vanillaNum', 'vanilla_standardized']]

cookieRecipes['brownsugar_standardized'] = cookieRecipes.apply(
        lambda row: standardizer(row['brownsugarNum'], 
        row['brownsugarMeasurement']), axis=1)

cookieRecipes['granulated_sugar_standardized'] = cookieRecipes.apply(
        lambda row: standardizer(row['granulated_sugarNum'], 
        row['granulated_sugarMeasurement']), axis=1)

cookieRecipes['bakingsoda_standardized'] = cookieRecipes.apply(
        lambda row: standardizer(row['bsNum'], 
        row['bsMeasurement']), axis=1)

cookieRecipes['flour_standardized'] = cookieRecipes.apply(
        lambda row: standardizer(row['flourNum'], 
        row['flourMeasurement']), axis=1)


# Aggregate the ingredients by the number of times they appear in recipes
    # (i.e., by relative importance)






# What is the average baking time? What is the average baking temperature?

# Retrieve baking temperatures
instructions = cookieRecipes.instructions.tolist()
temp = []
for i in instructions:
    temp.append("".join([x for x in i if "degree" in x]))

recipeTemperature = [i.partition("degrees")[0].split()[-1] for i in temp if i != ""]

# Retrieve baking times
temp = []
for i in instructions:
    for x in i:
        temp.append("".join(sent_tokenize(x)))
temp = [x.split(".") for x in temp]

bakingTime = []
for i in temp:
    for x in i:
        bakingTime.append("".join([x for x in i if "bake" in x.lower() and x != None]))

del temp, i, x

    # Remove null strings
times = []
bakingTime = set(list(filter(None, bakingTime)))
bakingTime = [i.partition("minutes")[0].split()[-3:] for i in bakingTime if i != ""]
for i in bakingTime:
    for item in i:
        if "to" in item or item.isdigit()==True:
            times.append(i)
        
times = list(map(list, set(map(lambda i: tuple(i), bakingTime))))

# NOTE: I am going to use the method of "throwing everything into a bowl and 
    # mixing it up", so I won't pull out specific instructions besides the 
    # baking temperature and time



##################################################################################
# Plot ingredients

# What is the distribution of each ingredient? want a boxplot here

# What is the correlation within recipe for each ingredient? Do recipes with more 
    # butter tend to have more sugar? etc?


