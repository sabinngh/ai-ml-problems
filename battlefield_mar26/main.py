import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

df = pd.read_csv("data.csv")

# SUBTASK 1


team_0 = df[df['Team'] == 0]
X0 = team_0[['x','y']]
db = DBSCAN(eps = 0.02, min_samples = 10)
labels0 = db.fit_predict(X0)
unique,counts = np.unique(labels0,return_counts = True)
smallest_cluster = unique[np.argmin(counts)]

special = (labels0 == smallest_cluster).astype(int)
subtask1 = np.zeros(len(df), dtype = int)
subtask1[team_0.index] = special
  

# SUBTASK 2
groups = np.zeros(len(df), dtype=int)

current_group = 0

for team in [0,1]:

    subset = df[df["Team"] == team]
    X = subset[["x","y"]]

    db = DBSCAN(eps=0.02, min_samples=10)
    labels = db.fit_predict(X)

    labels = labels + current_group

    groups[subset.index] = labels

    current_group = groups.max() + 1


#submission

submission1 = pd.DataFrame({
    "subtaskID": 1,
    "datapointID": df["ID"],
    "answer": subtask1
})

submission2 = pd.DataFrame({
    "subtaskID": 2,
    "datapointID": df["ID"],
    "answer": groups
})

submission = pd.concat([submission1, submission2])

submission.to_csv("submission_2_mar6th.csv", index=False)

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("data.csv")

# plt.figure(figsize=(8,6))

# # Team 0
# team0 = df[df["Team"] == 0]
# plt.scatter(team0["x"], team0["y"], label="0")

# # Team 1
# team1 = df[df["Team"] == 1]
# plt.scatter(team1["x"], team1["y"], label="1")

# plt.xlabel("x")
# plt.ylabel("y")

# plt.legend(title="Team")

# plt.show()