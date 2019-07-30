import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression


raw_df = pd.read_csv('GlobalTemperatures.csv')
raw_df = raw_df[pd.notnull(raw_df['LandAverageTemperature'])]

START_YEAR = 1860
END_YEAR = int(re.split('-', raw_df.iloc[-1]['dt'])[0])

new_rows = []

for year in range(START_YEAR, END_YEAR + 1):
    all_rows_for_year = raw_df[
        raw_df['dt'].str.contains(str(year))
    ]['LandAverageTemperature']

    average_temperature = all_rows_for_year.mean()
    new_rows.append([year, average_temperature])

processed_df = pd.DataFrame(new_rows, columns=['year', 'average_temperature'])
processed_df = processed_df[pd.notnull(processed_df['average_temperature'])]

lm = LinearRegression()
lm.fit(
    X=processed_df.drop('average_temperature',axis=1),
    y=processed_df['average_temperature'])


def predict_temperature(year):
    if year <= END_YEAR:
        print(f'Year must be larger than {END_YEAR}.')
    else:
        value = round(lm.predict([[year]])[0], 3)
        print(f'The {year} predicted Average Earth Land Temperature is {value}°C')


def export_graph():
    plt.figure(figsize=(10,5))
    fig = sns.lineplot(
        x=processed_df['year'],
        y=processed_df['average_temperature']
    ).get_figure().savefig('ayelt')
    print("Graph exported successfully as 'ayelt.png'")


if __name__ == '__main__':
    if sys.argv[1] == 'export':
        export_graph()
    else:
        try:
            predict_temperature(int(sys.argv[1]))
        except ValueError:
            print('Year must be an integer')
