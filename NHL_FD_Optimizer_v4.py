#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DATA 670 - Capstone, UMGC 2022
# Written by Joseph Coleman
# Last updated March 13, 2022
# The purpose of this code is to define and construct a constraint satisfaction problem 
# in order to create a user-specified number of optimized NHL fanduel lineups.
# The script was built with the help of Nick's Niche passion project. 
# Nick was even nice enough to assess and correct my script after I contacted him via email. 


# In[2]:


pip install pulp


# In[3]:


import pandas as pd
import numpy as np
import re
from pulp import *

players = pd.read_csv(r"C:\\Users\\jcole\\Documents\\DATA670\\slate_player_lists\\slateplayers.csv")
players.sort_values(by=['Pos'])
players.rename(columns = {'Player.Name':'PlayerName'}, inplace = True)
player_pool = players[["Id", "PlayerName", "Pos", "Salary", "Team", "Proj.FP"]].groupby(["Id", "PlayerName", "Pos", "Salary", "Team", "Proj.FP"]).agg('count')

player_pool = player_pool.reset_index()

salaries = {}
points = {}
teams = {}
lineups_dict = {}

for pos in player_pool.Pos.unique():
    player_pool_pos = player_pool[player_pool.Pos == pos]
    salary = list(player_pool_pos[['Id', 'Salary']].set_index("Id").to_dict().values())[0]
    point = list(player_pool_pos[['Id', 'Proj.FP']].set_index("Id").to_dict().values())[0]
    team = list(player_pool_pos[['Id', 'Team']].set_index("Id").to_dict().values())[0]
    
    salaries[pos] = salary
    points[pos] = point
    teams[pos] = team 
    
pos_num_available = {
    "W": 2,
    "C": 2,
    "D": 2,
    "G": 1
}

team_num_available = {
    "ANH": 4,
    "ARI": 4,
    "BOS": 4,
    "BUF": 4,
    "CAR": 4,
    "CGY": 4,
    "CHI": 4,
    "CLS": 4,
    "COL": 4,
    "DAL": 4,
    "DET": 4,
    "EDM": 4,
    "FLA": 4,
    "LA": 4,
    "MIN": 4,
    "MON": 4,
    "NJ": 4,
    "NSH": 4,
    "NYI": 4,
    "NYR": 4,
    "OTT": 4,
    "PHI": 4,
    "PIT": 4,
    "SEA": 4,
    "SJ": 4,
    "STL": 4,
    "TB": 4,
    "TOR": 4,
    "VAN": 4,
    "VGK": 4,
    "WAS": 4,
    "WPG": 4
}

salary_cap = 55000

for lineup in range (1,17):
    _vars = {k: LpVariable.dict(k, v, cat="Binary") for k, v in points.items()}
    
    prob = LpProblem("Fantasy", LpMaximize)
    player_points = []
    player_salaries = []
    nplayers = []
    teams = []
    
    for k, v in _vars.items():
        player_salaries += lpSum([salaries[k][i] * _vars[k][i] for i in v])
        player_points += lpSum([points[k][i] * _vars[k][i] for i in v])
        nplayers += lpSum([_vars[k][i] for i in v])
        teams += lpSum([_vars[k][i] for i in v])
        if k == "G":
            prob += lpSum([_vars[k][i] for i in v]) == pos_num_available[k]
        else:
            prob += lpSum([_vars[k][i] for i in v]) >= pos_num_available[k]
    
    prob += lpSum(teams) <= team_num_available
    prob += lpSum(player_points)
    prob += lpSum(player_salaries) <= salary_cap
    prob += lpSum(nplayers) == 9 
    score = str(prob.objective)
    
    if not lineup == 1:
        prob += (lpSum(player_points) <= total_score - 0.001)
    
    prob.solve()
    
    constraints = [str(const) for const in prob.constraints.values()]
    lineupList = []
    
    for v in prob.variables():
        score = score.replace(v.name, str(v.varValue))
        if v.varValue != 0:
            lineupList.append(v.name)
            
    posList = [a[0] for a in lineupList]
    posUnique= list(set(posList))
    checkDict = {}
    for p in posUnique:
        if posList.count(p)> 2:
            checkDict.update({p: posList.count(p)})
    for i in checkDict.keys():
        n = 2
        for j in lineupList:
            print(j)
            if i == j[0] and n<checkDict[i]:
                lineupList.append(lineupList.pop(lineupList.index(j)))
                n+=1
                
    total_score = eval(score)        
    lineupList.append(total_score)
    print(lineup, total_score)
    lineups_dict.update({lineup: lineupList})

lineups_exp = pd.DataFrame(lineups_dict)
lineups_exp = lineups_exp.T
newcols = ['C1', 'C2', 'D1', 'D2', 'G', 'W1', 'W2', 'UTIL1', 'UTIL2', 'Total_Score']
lineups_exp.columns = newcols
lineups_exp = lineups_exp.reindex(columns = ['C1', 'C2', 'W1', 'W2', 'D1', 'D2', 'UTIL1', 'UTIL2', 'G', 'Total_Score'])
newcols = ['C', 'C', 'W', 'W', 'D', 'D', 'UTIL', 'UTIL', 'G', 'Total_Proj_Score']
lineups_exp.columns = newcols
display(lineups_exp)

lineupsexp = lineups_exp

lineupsexp.to_csv(r"C:\\Users\\jcole\\Documents\\DATA670\\slate_player_lists\\lineupsimport.csv")

