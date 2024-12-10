from Pre_processing import *
import pandas as pd

def preProcessing(data):

    # miss data number
    missing_columns = ['wage', 'release_clause_euro', 'club_rating', 'club_jersey_number', 'value']
    # outlier in data
    """""
    for i in missing_columns:
        oulier=find_outliers_IQR(data[i])
        print("number of outliers: "+ i +" "+str(len(oulier)))
        print(oulier)
    """""
    mean_column = ['club_rating', 'club_jersey_number']
    for i in mean_column:
        data[i] = data[i].fillna(data[i].mean())
    medain_column = ['wage', 'release_clause_euro']

    for i in medain_column:
        data[i] = data[i].fillna(data[i].median())

    # split coloumn postion to 4 column each column express about one postaion
    data[['pos1', 'pos2', 'pos3', 'pos4']] = data['positions'].str.split(',', expand=True)
    postion = ['pos1', 'pos2', 'pos3', 'pos4', 'tags', 'club_team', 'traits']
    for i in postion:
        data[i] = data[i].replace(np.nan, 0)

    # convert category to numerical by encoder
    fencode = ['nationality', 'club_team', 'traits']
    data = Feature_Encoder(data, fencode)

    # convet category to numerical by one hot encoder
    categrey_columns = ['preferred_foot', 'work_rate', 'body_type', 'club_position', 'pos1', 'pos2', 'pos3', 'pos4',
                        'tags']
    for feature in categrey_columns:
        data = encode_and_bind(data, feature)

    # convert date to year in club_join_date
    data['club_join_date'] = data['club_join_date'].replace(np.nan, 0)
    for element in range(len(data)):
        x = data.at[element, 'club_join_date']
        if x == 0:
            continue
        s = x.split("/")
        data.at[element, 'club_join_date'] = int(s[2])

    # convert date to year in contract_end_year
    data['contract_end_year'] = data['contract_end_year'].replace(np.nan, 0)
    for element in range(len(data)):
        x = data.at[element, 'contract_end_year']
        if x == 0:
            continue
        l = len(x)
        if l == 4:
            data.at[element, 'contract_end_year'] = int(x)
            continue
        s = x.split("-")
        year = 2000 + int(s[2])
        data.at[element, 'contract_end_year'] = int(year)

    # convert categorical data to numerical then summation them
    pos = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB',
           'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
    for ele in pos:
        data[ele] = data[ele].replace(np.nan, 0)
        for element in range(len(data)):
            t = data.at[element, ele]
            if t == 0:
                continue
            t2 = t.split("+")
            res = int(t2[0]) + int(t2[1])
            data.at[element, ele] = res

    # fill zero value to median
    date_missing_columns = ['club_join_date', 'contract_end_year']
    for ele in date_missing_columns:
        data[ele] = data[ele].replace(0, data[ele].median())
    for j in pos:
        data[j] = data[j].replace(0, data[j].median())

    # Drop coloumn and make value coloumn =-1
    value = data['PlayerLevel']
    data = data.drop(labels=['id', 'name', 'full_name', 'birth_date', 'national_team',
                             'national_rating', 'national_team_position', 'national_jersey_number', 'positions',
                             'PlayerLevel'], axis=1)
    data = pd.concat([data, value], axis=1)
    X = data.drop(['PlayerLevel'], axis=1)

    # column value
    Y = data['PlayerLevel']
    return X, Y



