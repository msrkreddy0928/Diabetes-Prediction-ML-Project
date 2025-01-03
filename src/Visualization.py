import matplotlib.pyplot as plt
import seaborn as sns   

# pair_plot function: This function creates a pair plot (scatterplot matrix) to visualize relationships 

def pair_plot(data,target):
    sns.pairplot(data,hue=target,diag_kind='kde')
    plt.show()
    
# count_plot_data function: This function generates count plots (bar charts) for each categorical feature in the data.
def count_plot_data(cleaned_df,categorical_features):
    i=1
    for col in categorical_features:
        plt.subplot(2,3,i)
        sns.countplot(x=col, data=cleaned_df)
        plt.title(f'Count plot of {col}')
        i+=1
        
    plt.show() 


# raw_data_distribution function: This function visualizes the distribution (histogram + KDE) 

def raw_data_distribution(data, columns):
    i=1
    for col in columns:
        plt.subplot(2,2,i)
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        i+=1
    plt.show()   
        
 
 # data_distribution_after_trans function: This function visualizes the distribution of continuous features
 
def data_distribution_after_trans(data, columns):
    i=1
    for col in columns:
        plt.subplot(2,2,i)
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        i+=1
    plt.show()
   
   
  # plot_box_plots function: This function generates boxplots for the specified columns to detect outliers   
    
def plot_box_plots(data,columns):
    i=1
    for col in columns:
        plt.subplot(2,3,i)
        sns.boxplot(data[col])
        plt.title(f'boxplot of {col}')
        i+=1
    plt.show()    
        
        
  # plot_correlation_matrix function: This function generates a heatmap of the correlation matrix for the features in the data.
        
def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


