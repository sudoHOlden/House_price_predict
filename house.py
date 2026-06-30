import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


df = pd.read_csv('housedata.csv')
df_median = df['Area_SqFt'][df['Area_SqFt'] < 5000].median()
df['Area_SqFt'] = df['Area_SqFt'].fillna(df_median)
df = df[df['Area_SqFt'] < 5000]
df['Rooms'] = df['Rooms'].fillna(df['Rooms'].mean())
df['Furnishing'] = df['Furnishing'].fillna(0)

def fur_encod(df, name):
    if df[name] == 'Furnished':
        return 1
    elif df[name] == 'Semi-Furnished':
        return 0.5
    else:
        return 0

def prop_type_ordinal_encod(df, name):
    if df[name] == 'Apartment':
        return 1
    elif df[name] == 'Independent House':
        return 2
    elif df[name] == 'Duplex':
        return 3
    else:
        return 4

df['Furnishing'] = df.apply(lambda x: fur_encod(x, 'Furnishing'), axis=1)
df['Has_Pool'] = df.apply(lambda x: 1 if x['Has_Pool'] == 'Yes' else 0, axis=1)
grop_house = df.groupby('Property_Type')['Price'].mean().sort_values(ascending=False)
df['Property_Type'] = df.apply(lambda x: prop_type_ordinal_encod(x, 'Property_Type'), axis=1)
df = pd.get_dummies(df, columns=['Street_Type','Location'],dtype=int)


y = df['Price']
X = df.drop(columns='Price')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
columns_x_train = X_train.columns
model = RandomForestRegressor(n_estimators=100,max_depth=20,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
scores = cross_val_score(model, X_train, y_train, cv=5)

#draft for further telegran bot
def predict_price(area, rooms, year, location, street_type, furnishing, prop_type, has_pool):
    data = {
        'Area_SqFt': area,
        'Rooms': rooms,
        'Build_Year': year,
        'Furnishing': furnishing,
        'Property_Type': prop_type,
        'Has_Pool': has_pool,
        'Street_Type': street_type,
        'Location': location
    }
    df = pd.DataFrame(data, index=[0])
    df['Furnishing'] = df.apply(lambda x: fur_encod(x, 'Furnishing'), axis=1)
    df['Has_Pool'] = df.apply(lambda x: 1 if x['Has_Pool'] == 'Yes' else 0, axis=1)
    df['Property_Type'] = df.apply(lambda x: prop_type_ordinal_encod(x, 'Property_Type'), axis=1)
    df = pd.get_dummies(df, columns=['Street_Type', 'Location'], dtype=int)
    df = df.reindex(columns=columns_x_train, fill_value=0)
    return df


result = predict_price(2000, 4, 2015, 'Noida', 'Main Road', 'Furnished', 'Villa', 'Yes')
#end draft

print(df.head(5).to_string())
print('----' * 50)
print(df.shape)
print('----' * 50)
print(df.describe().to_string())
print('----' * 50)
print(df.info())
print('----' * 50)
print('----' * 50)
print(scores)
print('----'*50)
print(scores.mean())
print('----'*50)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

test_model = X_test.iloc[[1]]
model_test_ok = model.predict(test_model)
print(model_test_ok)
print(y_test.iloc[1])
print(test_model)

