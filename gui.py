#import modules
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import customtkinter

#create dataframe
df = pd.read_csv(r'dataset.csv')

#make inputs and target
inputs = df.drop('RiskLevel',axis='columns')
target = df['RiskLevel']

#creat the model
model = tree.DecisionTreeClassifier()

#feed the model
model.fit(inputs,target)

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()
root.geometry("500x800")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

titleLabel = customtkinter.CTkLabel(master=frame, text="Dengue disease predictor", font=customtkinter.CTkFont(family="Helvetica", size=28))
titleLabel.pack(padx=0, pady=10)

ageLabel = customtkinter.CTkLabel(master=frame, text="Age : ")
ageLabel.pack(padx=0, pady=5)

ageET = customtkinter.CTkEntry(master=frame, placeholder_text="Enter your age")
ageET.pack(padx=0, pady=5)

sbpLabel = customtkinter.CTkLabel(master=frame, text="Systolic BP : ")
sbpLabel.pack(padx=0, pady=5)

sbpET = customtkinter.CTkEntry(master=frame, placeholder_text="Enter your Systolic BP")
sbpET.pack(padx=0, pady=5)

dbpLabel = customtkinter.CTkLabel(master=frame, text="Diastolic BP : ")
dbpLabel.pack(padx=0, pady=5)

dbpET = customtkinter.CTkEntry(master=frame, placeholder_text="Enter your Diastolic BP")
dbpET.pack(padx=0, pady=5)

bsLabel = customtkinter.CTkLabel(master=frame, text="Blood Sugar : ")
bsLabel.pack(padx=0, pady=5)

bsET = customtkinter.CTkEntry(master=frame, placeholder_text="Enter your Blood Sugar")
bsET.pack(padx=0, pady=5)

tempLabel = customtkinter.CTkLabel(master=frame, text="Temperature : ")
tempLabel.pack(padx=0, pady=5)

tempET = customtkinter.CTkEntry(master=frame, placeholder_text="Enter your body temperature")
tempET.pack(padx=0, pady=5)

hrLabel = customtkinter.CTkLabel(master=frame, text="Heart Rate : ")
hrLabel.pack(padx=0, pady=5)

hrET = customtkinter.CTkEntry(master=frame, placeholder_text="Enter your heart rate")
hrET.pack(padx=0, pady=5)

result = "result"

resultLabel = customtkinter.CTkLabel(master=frame, text=result)
resultLabel.pack(padx=0, pady=0)

def submit():
    age = int(ageET.get())
    sbp = int(sbpET.get())
    dbp = int(dbpET.get())
    bs = float(bsET.get())
    temp = int(tempET.get())
    hr = int(hrET.get())

    resultLabel.configure(text = model.predict([[age, sbp, dbp, bs, temp, hr]])[0])

submitBT = customtkinter.CTkButton(master=frame, text="Submit", command=submit)
submitBT.pack(padx=0, pady=20)

root.mainloop()
