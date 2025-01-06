# import pandas as pd
# from pymongo import MongoClient


# #vNJBBbtr5B91l4n0

# uri = "mongodb+srv://smallreddy:vNJBBbtr5B91l4n0@myproject.riyfu.mongodb.net/?retryWrites=true&w=majority&appName=MyProject"

# client = MongoClient(uri)

# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)
    
# database = client["DiabetesMLProject"]    
# collection = database["DiabetesProject"]


# df = pd.read_csv("EDA/data/diabetes_prediction_dataset.csv")

# dict = df.to_dict(orient="records")
 
# result = collection.insert_many(dict)

# # print(result)

# client.close()






    
    
    
  