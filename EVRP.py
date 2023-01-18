import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import itertools
import gmaps
import googlemaps
import docplex.mp.model
import requests
from urllib.parse import urlencode
#Please download the respective libraries


API_KEY = 'ENTER_YOUR_API_HERE'
gmaps.configure(api_key=API_KEY)
googlemaps = googlemaps.Client(key=API_KEY)



#Requesting customers and depot coordinates via Google Maps Geocoding API

def extract_lat_lng(address, data_type = 'json'):
    endpoint = f"https://maps.googleapis.com/maps/api/geocode/{data_type}"
    params = {"address": address, "key": API_KEY}
    url_params = urlencode(params)

    url = f"{endpoint}?{url_params}"
    r = requests.get(url)
    if r.status_code not in range(200,299):
        return{}
    latlng = {}
    try:
         latlng = r.json()['results'][0]['geometry']['location']
    except:
        pass
    return latlng.get("lat"), latlng.get("lng")


#list of depot and customers address

list = [] 
#insert all addresses in quotation marks here, separated with commas
#the first address is the depot
#do note down the order of the addresses used for references 
#make sure address are valid (searchable via maps.google.com)


#Save respective latitude and longitude into DataFrame

df = pd.DataFrame({"latitude":[],"longitude":[]})

z=0
for i in list:
    latlng = extract_lat_lng(i)
    df.at[z,"latitude"] = latlng[0]
    df.at[z,"longitude"] = latlng[1]
    z=z+1
display (df)


#Save coordinates to local CSV file (to reduce repeating requests)

np.savetxt("coordinates.csv", df, delimiter=",")



#Request Distance between all customers locations using Google Maps Directions API and requested coordinates

def _distance_calculator(_df):
    _distance_result = np.zeros((len(_df),len(_df)))
    _df['latitude-longitude'] = '0'
    for i in range(len(_df)):
        _df['latitude-longitude'].iloc[i] = str(_df.latitude[i]) + ',' + str(_df.longitude[i])
    
    for i in range(len(_df)):
        for j in range(len(_df)):
            # calculate distance of all pairs
            _google_maps_api_result = googlemaps.directions(_df['latitude-longitude'].iloc[i],
                                                            _df['latitude-longitude'].iloc[j],
                                                            mode = 'driving')
            # append distance to result list
            _distance_result[i][j] = _google_maps_api_result[0]['legs'][0]['distance']['value']
    return _distance_result

distance = _distance_calculator(df)
display (distance)

#Save distances to local CSV File
np.savetxt("distances.csv", distance, delimiter=",")



#Request Elevation and Elevation Differences calculation using Google Maps Elevation API and requested coordinates

def extract_elevation(coordinates,data_type="json"):
    endpoint = f"https://maps.googleapis.com/maps/api/elevation/{data_type}"
    params = {"locations": coordinates,"key":API_KEY}
    url_params = urlencode(params)
    url = f"{endpoint}?{url_params}"
    r = requests.get(url)
    if r.status_code not in range(200,299):
        return{}
    elevation = {}
    try:
         elevation = r.json()['results'][0]
    except:
        pass
    return elevation.get("elevation")

elevation = []
for i in range(len(df)):
    elevation.append(extract_elevation(df.loc[i, "latitude"], df.loc[i, "longitude"]))


#Save Elevation Differences to local CSV File
np.savetxt("elevations.csv", elevation, delimiter=",")



#Calculate Elevation Differences between each locations

import numpy as np

def _elevation_dif_calculator(list):
    _elevation_result = np.zeros((len(list),len(list)))
    
    for i in range (len(list)):
        for j in range (len(list)):
            _elevation_result [i][j] = list[i] - list[j]
    return _elevation_result

elevationdiff = _elevation_dif_calculator(elevation)


#Save Elevation Differences to local CSV File
np.savetxt("elevationdifferences.csv",elevationdiff, delimiter = ",")



#Calculate Accumulative Fuel Consumption between each locations using Distances and Elevation Differences

import numpy as np
import math
distance = np.loadtxt("distances.csv",delimiter =",")
elevation = np.loadtxt("elevationdifferences.csv", delimiter =",")

v = 6.45
M = 30000

lamda = 3.08e-5
k = 0.2
N = 36.67
V = 6.9
g = 9.8
Cr = 0.01
Cd = 0.7
A = 8
phi = 1.2041
N1 = 0.45
N2 = 0.45
B1 = lamda*k*N*V
B2 = (lamda*g)/(1000*N1*N2)
B3 = (0.5*lamda*Cd*A*phi)/(1000*N1*N2)

def elevation_fuel_consumption(distance,elevation):
    elevation_fuel = np.zeros((len(distance),len(distance)))
                              
    for i in range (len(distance)):
        for j in range (len(distance)):
            if i==j:
                elevation_fuel[i][j]=0
            else:
                division = elevation[i][j]/distance[i][j]
                theta = np.arctan(elevation[i][j]/distance[i][j])
                u = math.sin(theta)+(math.cos(theta)*Cr)
                #if u < 0:
                #    u = 0
                
                elevation_fuel[i][j] = max( (B1*(distance[i][j]/v))+\
                                           (B2*u*M*(distance[i][j]))+\
                                           (B3*(distance[i][j])*v*v) , 0 )         
    
    return elevation_fuel



fuel= elevation_fuel_consumption(distance,elevation)
np.savetxt("accumulativefuel.csv",fuel,delimiter=",")



# Solving for the Route

# customer count ('0' is depot) 
customer_count = len(df)-1 

# the number of vehicle
vehicle_count = 1

# Assign fuel values between each node to the CSV defined above
fuel = np.loadtxt("accumulativefuel.csv", delimiter =",")


#Define Solver with Pulp
    
# definition of LpProblem
problem = pulp.LpProblem("EVRP", pulp.LpMinimize)
# definition of variables which are 0/1
x = [[[pulp.LpVariable("x%s_%s,%s"%(i,j,k), cat="Binary") if i != j else None for k in range(vehicle_count)]for j in range(customer_count)] for i in range(customer_count)]
t = pulp.LpVariable.dicts("t", (i for i in range(customer_count)), \
                             lowBound=1,upBound= customer_count, cat='Continuous')

# objective function
problem += pulp.lpSum(fuel[i][j] * x[i][j][k] if i != j else 0
                      for k in range(vehicle_count) 
                      for j in range(customer_count) 
                      for i in range (customer_count))

# constraints
# formulae (2)
for j in range(1, customer_count):
    problem += pulp.lpSum(x[i][j][k] if i != j else 0 
                          for i in range(customer_count) 
                          for k in range(vehicle_count)) == 1 


# formulae (3)
for k in range(vehicle_count):
    problem += pulp.lpSum(x[0][j][k] for j in range(1,customer_count)) == 1
    problem += pulp.lpSum(x[i][0][k] for i in range(1,customer_count)) == 1


# formulae (4)
for k in range(vehicle_count):
    for j in range(customer_count):
        problem += pulp.lpSum(x[i][j][k] if i != j else 0 
                              for i in range(customer_count)) -  pulp.lpSum(x[j][i][k] for i in range(customer_count)) == 0

# formulae (5)
for i in range (customer_count):
    for j in range(customer_count):
        if i!=j and (i!=0 and j!=0):
            problem += t[j]>=t[i]+1 - (2*customer_count)*(1-x[i][j][k])


# print vehicle_count which needed for solving problem
# print calculated minimum distance value
# change time limit to match with computer configuration, as desired
problem.solve(pulp.GUROBI_CMD(timeLimit=900,msg=1))
if problem.solve() == 1:
    print('Vehicle Requirements:', vehicle_count)
    print('Fuel Consumption:', pulp.value(problem.objective))

# print route 
import copy
active_arcs = []
for k in range(vehicle_count):
    for i in range(customer_count):
        for j in range(customer_count):
            if i != j and pulp.value(x[i][j][k]) == 1:
                active_arcs.append((i,j))
def get_plan(r0):
    r=copy.copy(r0)
    route = []
    while len(r) != 0:
        plan = [r[0]]
        del (r[0])
        l = 0
        while len(plan) > l:
            l = len(plan)
            for i, j in enumerate(r):
                if plan[-1][1] == j[0]:
                    plan.append(j)
                    del (r[i])
        route.append(plan)
    return(route)
route_plan = get_plan(active_arcs)
print(route_plan)



#Visualization 1

import matplotlib.pyplot as plt
#plt.figure(figsize=(ENTER_FIGSIZE))
#plt.axis([LON_1,LON_2,LAT_1,LAT_2])

for i in range(customer_count):    
    if i == 0:
        #plt.scatter(df.longitude[i], df.latitude[i], c='green', s=2000)
        #plt.text(df.longitude[i], df.latitude[i], "depot", fontsize=50)
        continue
    else:
        plt.scatter(df.longitude[i], df.latitude[i], c='orange', s=2000)
        plt.text(df.longitude[i], df.latitude[i], i, fontsize=50)

for k in range(vehicle_count):
    for i in range(customer_count):
        for j in range(customer_count):
            if i != j and pulp.value(x[i][j][k]) == 1:
                plt.plot([df.longitude[i], df.longitude[j]], [df.latitude[i], df.latitude[j]], c="black")

plt.show()



#Visualization 2

import matplotlib.pyplot as plt
route = 
plt.figure(figsize=(100,60))
plt.axis([-66.0559591, -66.0496207, 45.2713983, 45.2756326])

for i in range(customer_count):    
    if i == 0:
        #plt.scatter(df.longitude[i], df.latitude[i], c='green', s=2000)
        #plt.text(df.longitude[i], df.latitude[i], "depot", fontsize=50)
        continue
        
    else:
        plt.scatter(df.longitude[i], df.latitude[i], c='orange', s=2000)
        plt.text(df.longitude[i], df.latitude[i], i, fontsize=50)

for i in (currentlist):
    plt.plot([df.longitude[i], df.longitude[route_plan[route_plan.index(i)+1]]], [df.latitude[i], df.latitude[route_plan[route_plan.index(i)+1]]], c="black")

plt.show()
