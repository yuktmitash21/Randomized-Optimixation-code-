import sklearn.preprocessing as sk
import pandas as pd

filename = "Admission_Predict_Ver1.csv"
label = "Chance of Admit"
instances = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]

row_vector = instances[:].append(label)
print(row_vector)
print(instances)

file = pd.read_csv(filename, usecols=row_vector)

for field in instances:
    file[field] = sk.normalize(file[field])[0]

file.to_csv("predictAdmissionNormalized.csv", index=False, header=False)