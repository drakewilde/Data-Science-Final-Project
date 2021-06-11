import pandas as pd
import numpy as np


filepath = "C:/Users/wilde/Desktop/Spring 2021/cs1951a Data Science/Assignments/final_project_data_science/sleeping-alone-data/sleeping-alone-data.csv"

df = pd.read_csv(filepath, encoding = "ISO-8859-1")

regions_to_states = {
    'Pacific' : ["WA", "OR", "CA", "AK", "HI"],
    'Mountain' : ["MT", "ID", "WY", "NV", "UT", "CO", "AZ", "NM"],
    'West North Central' : ["ND", "SD", "NE", "KS", "MN", "IA", "MO"],
    'West South Central' : ["OK", "TX", "AR", "LA"],
    'East North Central' : ["WI", "MI", "IL", "IN", "OH"],
    'East South Central' : ["KY", "TN", "MS", "AL"],
    'Middle Atlantic' : ["NY", "PA", "NJ"],
    'South Atlantic' : ["WV", "MD", "DE", "DC", "VA", "NC", "SC", "GA", "FL"],
    'New England' : ["ME", "NH", "VT", "MA", "CT", "RI"]
    }

sleeping_arangements_sorted = [
    "Never", 
    "Once a year or less",
    "Once a month or less",
    "A few times per month",
    "A few times per week",
    "Every night"
    ]



def unique_regions(dataframe):
    '''
    Output all regions
    '''    
    u_regions = dataframe.columns[-1]
    out = df[u_regions].unique()
    out = np.delete(out, [0, 5]) #del 'response' and 'nan'
    return out

def unique_sleeping_arangements(dataframe):
    '''
    Output all sleeping arangements
    '''    
    u_arangements = dataframe.columns[4]
    out = df[u_arangements].unique()
    out = np.delete(out, [0, -1]) #del 'response' and 'nan'
    return out


#TODO: Create new dataframe with:
    #1. Every row duplicated with each state in that region
    #2. Sleeping arangements row now numerical 
        #[0.000, 0.003, 0.033, 0.100, 0.400, 1.000]

#TODO: Choropleth map with State and Sleeping arangements as scale

#TODO: Choropleth map with State and Income as scale

#TODO: Choropleth map with State and Age as scale

#TODO: Choropleth map with State and Education as scale



def main():
    
    list_regions = unique_regions(df)
    print(list_regions)
    
    list_arangements = unique_sleeping_arangements(df)
    print(list_arangements)


if __name__ == '__main__':
    main()
    

