import matplotlib.pyplot as plt
import seaborn as sns   
from cleaning import cleaned_df,categorical_features,continous_features,df_after_trans,correlation,initial_df


def pair_plot(data,target):
    sns.pairplot(data,hue=target,diag_kind='kde')
    plt.show()
    

def count_plot_data(cleaned_df,tegorical_features):
    i=1
    for col in categorical_features:
        plt.subplot(2,3,i)
        sns.countplot(x=col, data=cleaned_df)
        plt.title(f'Count plot of {col}')
        i+=1
        
    plt.show() 




def raw_data_distribution(data, columns):
    i=1
    for col in columns:
        plt.subplot(2,2,i)
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        i+=1
    plt.show()   
        
 
 
 
def data_distribution_after_trans(data, columns):
    i=1
    for col in columns:
        plt.subplot(2,2,i)
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        i+=1
    plt.show()
    
    
def plot_box_plots(data,columns):
    i=1
    for col in columns:
        plt.subplot(2,3,i)
        sns.boxplot(data[col])
        plt.title(f'boxplot of {col}')
        i+=1
    plt.show()    
        
        
 
        
def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()



# count_plot_data(cleaned_df,categorical_features)                    #count plots
# raw_data_distribution(cleaned_df,continous_features)                 #hist
# data_distribution_after_trans(df_after_trans,continous_features)     #hist after transformations
# plot_box_plots(df_after_trans,continous_features)                    #boxplots
# plot_correlation_matrix(correlation)                                 #correlation_matrix
pair_plot(initial_df,"diabetes")
