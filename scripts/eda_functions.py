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
    
def plot_bar(df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str, file_name= None)->None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()

def plot_heatmap(df:pd.DataFrame, title:str, cbar=False, file_name= None)->None:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
    plt.title(title, size=18, fontweight='bold')
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()

def plot_box(df:pd.DataFrame, x_col:str, title:str, file_name= None) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data = df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()

def plot_box_multi(df:pd.DataFrame, x_col:str, y_col:str, title:str, file_name= None) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str, file_name= None) -> None:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue, style=style)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks( fontsize=14)
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight')
    plt.show()


        
    
pd.options.display.float_format = format_float