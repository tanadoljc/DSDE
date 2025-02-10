import pandas as pd

def main():
    file = 'scores_student.csv'
    func = input()
    df = pd.read_csv(file)

    if func == 'Q1':
        print(df.shape)
    elif func == 'Q2':
        print(df['score'].max())
    elif func == 'Q3':
        condition = df['score'] >= 80
        print(df[condition]['id'].count())
    else:
        print('No Output')

if __name__ == "__main__":
    main()