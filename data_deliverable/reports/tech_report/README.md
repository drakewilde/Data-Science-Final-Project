# Tech Report

### Tech Report ###
***Where is the data from?***

Our data is from a fivethirtyeight.com GitHub data repository. It was originally collected from a SurveyMonkey audience.

***How did you collect your data?***

We downloaded the raw GitHub CSV file from the FiveThirtyEight data repository. As stated above however it was originally collected from a SurveyMonkey Audience.

***Is the source reputable?***

FiveThirtyEight is known to be a reputable site for datasets. It was named the "Data Journalism Website of the Year" for 2016 by the Global Editors Network, a Paris-based organization that promotes innovation in newsrooms around the world and also won an additional award for "News Data App of the Year" that same year. SurveyMonkey is online survey software that helps you to create and run professional online surveys, it is very powerful and a well known online application. Surveys can be administered/sent to users via a number of ways – e-mail, web link, Facebook, embed link on web page, link via Twitter, and website pop-ups, while enhanced security is possible (used with our dataset to ensure respondent confidentiality).

***How did you generate the sample? Is it comparably small or large? Is it representative or is it likely to exhibit some kind of sampling bias?***

I use the pandas sample method to sample the data, it is about 1% of the dataset, so comparably small. You can see this sample (10 rows) by running the get_data.py file in the data folder. It is likely that there is some sampling bias for a plethora of the variables we are using in analysis such as age, income, and education. For that reason we will most likely look at majority categories conditionally (i.e. E(X|age > 60). However because the sleeping arrangements category is a relatively equal split the distribution of the sleeping arrangement variable (which is the main dependent variable in our analyses) will be represented well in a sample of a larger size.

***Are there any other considerations you took into account when collecting your data? This is open-ended based on your data; feel free to leave this blank. (Example: If its user data, is it public/are they consenting to have their data used? Is the data potentially skewed in any direction?)***

The original collection of this data was done with many considerations of the user in mind, taken care of by SurveyMonkey. No data point displays names, actual income values, or direct location (census region is used). Essentially a respondent could not be identified from participating in this survey. Pertaining to respondents being representative of our overall population.

***How clean is the data? Does this data contain what you need in order to complete the project you proposed to do? (Each team will have to go about answering this question differently, but use the following questions as a guide. Graphs and tables are highly encouraged if they allow you to answer these questions more succinctly.)***

The columns of the data we use are relatively clean, we were able to load and plot the categorical data without any problems. Our main task will be to convert certain attributes into indicator variables to run our analyses efficiently i.e. for income buckets: $0-$25,000 = 1 $25,000-$560,000 = 2, ..., etc. Each unique category can be found by converting a list of column values to a set. From there it is very simple to mutate the columns of the dataframe to reflect indicator variables. We will be able to calculate any relevant metric for hypothesis testing, prediction tasks, and chi-squared tests. Minimal cleaning will need to be done for getting the respondents responses for why they sleep in separate beds if we decide to run analyses pertaining to them.

***How many data points are there total? How many are there in each group you care about (e.g. if you are dividing your data into positive/negative examples, are they split evenly)? Do you think this is enough data to perform your analysis later on? Are there missing values? Do these occur in fields that are important for your project's goals?***

There are a total of 1,093 data points. There is about a 50% split of people who never sleep in separate beds and people who variably do. Some categories only contain small variable splits which would not be good for analysis, for example most respondents have been in a relationship for more than twenty years, so to look at a hypothesis pertaining to respondents who have been in a relationship one year or less we most likely would not see an equal distribution of sleeping arrangement categories so we could not do analyses with these. Please see the plots directory for more information on this as well.

**Are there duplicates? Do these occur in fields that are important for your project's goals?***

There are no duplicate data points in fields that are important to our project's goals. Our main focus will be on missing information.

***How is the data distributed? Is it uniform or skewed? Are there outliers? What are the min/max values? (focus on the fields that are most relevant to your project goals)
Are there any data type issues (e.g. words in fields that were supposed to be numeric)? Where are these coming from? (E.g. a bug in your scraper? User input?) How will you fix them?***

Please see the plots directory. All relevant variable distribution plots are included. :)

The dependent variable "sleeping arrangements" is majority skewed towards those who sleep in the same bed all the time, however when the buckets of people who sleep in separate beds are summed, they are relatively the same ratios of the dataset. The male and female ratios in this dataset are relatively equal as well. However there is no direct indicator for other genders other than not responding to the question (which is a downfall of this dataset), so there is virtually no representation for other genders in the dataset. Most respondents were married and have been together for more than twenty years. Also most respondents are college educated and over the age of 60. Lastly most respondents are from the mid Atlantic census region. We have no type errors currently in loading our dataset. To find the distributions of each category we just found the frequency of them as strings.

***Do you need to throw any data away? What data? Why? Any reason this might affect the analyses you are able to run or the conclusions you are able to draw?***

When we want to look at a specific age groups, income groups, education, etc., we'll have to filter our dataset to find the specific subset of the categories we are analysing. Some categories are not heavily represented in the dataset so drawing conclusions about these groups would not be statistically correct.  
For certain analyses we will have to get rid of any data points where respondents did not answer the survey question such as "how long have you been in a relationship with your significant other", however most rows are retained after filtering. The question "When both you and your partner are at home, how often do you sleep in separate beds?" is the main dependent variable we are analysing in our hypotheses; about 98% of participants answer this question

***Summarize any challenges or observations you have made since collecting your data. Then, discuss your next steps and how your data collection has impacted the type of analysis you will perform. (approximately 3-5 sentences)***

From the beginning we've known our data is categorical, and although this allows for freedom of response from participants it makes our preprocessing more involved. Data is very qualitative due to it being a survey, so for our analyses we will have to convert some of these bucket types to incremental indicator values as mentioned above. This will make calculations for hypothesis testing much simpler. Also, if we decide to work with the sleeping arrangement reason responses for each participant we will have to parse the untitled columns of the dataset and also group similar responses. This would require defining a metric of "similarity" (defining which sleeping arrangement reasons belong in the same category). When looking at responses directly only a subset of 483 participants answered so we would have to ensure that relevant attributes for that subset of the data are representative of the overall population we are looking at.
