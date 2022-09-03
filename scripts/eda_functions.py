import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------------------------------     Cleaning Functions   ---------------------------------------------------
# Function to calculate missing values by column
def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(),df[column])
    
    return df[column]

def format_float(value):
    return f'{value:,.2f}'



# ----------------------------------------------------     Plotting Functions   ---------------------------------------------------
def plot_hist(df:pd.DataFrame, column:str, color:str, file_name= None)->None:
    # plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, figsize=(10, 5))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()

def plot_count(df:pd.DataFrame, column:str, hue= None,order=None, file_name= None) -> None:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, order= order, hue= hue)
    if hue:
        title = f'Distribution of {column} compared for different classes of {hue}'
    else:
        title = f'Distribution of {column}'
        
    plt.title(title, size=20, fontweight='bold')
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()
    
    
def plot_pie(df:pd.DataFrame, column:str, labels= None, file_name= None)->None:
    val = df[column].value_counts()
    lab = df[column].value_counts().index
    if labels:
        lab = [labels[x] for x in lab]
    fig, ax = plt.subplots(1, figsize=(7,7))
    ax.pie(val, labels= lab, autopct='%1.1f%%', textprops= {'fontsize': 14})
    plt.title(f'Pie Chart of {column}', size=20, fontweight='bold')
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()

    
def plot_count_compare(ax, df:pd.DataFrame, column:str, hue= None, order=None, group='full data'):
    sns.countplot(ax= ax, data=df, x=column, order= order, hue= hue)
    if hue:
        ax.set_title(f'{group}: Distribution of {column} for different {hue}s')
    else:
        ax.set_title(f'{group}: Distribution of {column}')
    
    
def plot_pie_compare(ax, df:pd.DataFrame, column:str, group='full data', labels=None)->None:
    val = df[column].value_counts()
    lab = df[column].value_counts().index
    if labels:
        lab = [labels[x] for x in lab]
    ax.pie(val, labels= lab, autopct='%1.1f%%', textprops= {'fontsize': 14})
    ax.set_title(f'{group}: Pie Chart of {column}', size=20, fontweight='bold')
        
    
pd.options.display.float_format = format_float