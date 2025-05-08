from gurobipy import Model, GRB, quicksum
import numpy as np
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from collections import deque
from collections import defaultdict
# import seaborn as sns
# import pandas as pd
from tabulate import tabulate

# import os

# PICTURE_DIR = "Pictures"
# os.makedirs(PICTURE_DIR, exist_ok=True)






def calculate_r(V, e, c):
    # Initialize r as a 3D array
    num_nodes = len(V)
    num_safeguards = len(e)
    num_levels = len(c[0])  # Assuming all safeguards have the same number of levels
    r = np.zeros((num_nodes, num_safeguards, num_levels))

    # Calculate r_ijl for each node, safeguard, and level
    for i in range(num_nodes):
        for j in range(num_safeguards):
            for k in range(num_levels):
                r[i][j][k] = V[i] ** (e[j] * c[j][k])
    return r





def binary_reachability(adj_list):
    """Returns a 2D array where P[i][j] = 1 if there is a path from i to j, else 0."""
    num_nodes = len(adj_list)
    P = np.zeros((num_nodes, num_nodes))

    def dfs(i, j):
        if P[i][j] == 1:
            return
        P[i][j] = 1
        for child in adj_list[j]:
            dfs(i, child)

    for i in range(num_nodes):
        dfs(i, i)

    for i in range(num_nodes):
        P[i][i] = 0

    return P





def normalized_path_count(adj_list):
    """Returns a 2D array where P[i][j] is the fraction of paths from i to j."""
    num_nodes = len(adj_list)
    P = np.zeros((num_nodes, num_nodes))

    def count_paths(i, j, memo):
        if i == j:
            return 1
        if i in memo and j in memo[i]:
            return memo[i][j]
        
        # Initialize i in memo if not exists
        memo.setdefault(i, {})

        paths = sum(count_paths(child, j, memo) for child in adj_list[i])
        memo[i][j] = paths
        return paths

    for i in range(num_nodes):
        memo = {}
        total_paths = sum(count_paths(i, k, memo) for k in range(num_nodes))
        total_paths -= 1
        if total_paths > 0:
            for j in range(num_nodes):
                if j == i:
                    continue
                P[i][j] = count_paths(i, j, memo) / total_paths

    return P





def inverse_shortest_path(adj_list):
    """Returns a 2D array where P[i][j] = 1/d(i, j), with d(i, j) as the shortest path length."""
    num_nodes = len(adj_list)
    P = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        queue = deque([(i, 0)])
        visited = set()

        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node != i:
                P[i][node] = 1 / (dist + 1)

            for child in adj_list[node]:
                queue.append((child, dist + 1))

    return P





def decay_factor_over_distance(adj_list, lambda_=0.5):
    """Returns a 2D array where P[i][j] = e^(-lambda * d(i, j))."""
    num_nodes = len(adj_list)
    P = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        queue = deque([(i, 0)])
        visited = set()

        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node != i:
                P[i][node] = np.exp(-lambda_ * dist)

            for child in adj_list[node]:
                queue.append((child, dist + 1))

    return P





def weighted_path_contribution(adj_list):
    """Returns a 2D array where P[i][j] is the sum of 1/d(i,j) over all paths from i to j."""
    num_nodes = len(adj_list)
    P = np.zeros((num_nodes, num_nodes))

    def dfs(i, j, depth):
        if i == j:
            return 1 / max(1, depth)
        return sum(dfs(child, j, depth + 1) for child in adj_list[i])

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                P[i][j] = dfs(i, j, 0)

    return P





def heat_diffusion_model(adj_list):
    """Returns a 2D array where P[i][j] follows a heat diffusion model: e^(-d(i,j))."""
    return decay_factor_over_distance(adj_list, lambda_=1.0)





def random_walk_influence(adj_list, lambda_=0.85):
    """Returns a 2D array where P[i][j] represents influence spread through a random walk."""
    num_nodes = len(adj_list)
    P = np.zeros((num_nodes, num_nodes))

    def walk(i, j, prob):
        if i == j:
            return prob
        if not adj_list[i]:
            return 0
        spread = prob * lambda_ / len(adj_list[i])
        return sum(walk(child, j, spread) for child in adj_list[i])

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                P[i][j] = walk(i, j, 1)

    return P





def eigenvector_centrality_influence(adj_list):
    """Returns a 2D array where P[i][j] is based on eigenvector centrality of nodes."""
    num_nodes = len(adj_list)
    A = np.zeros((num_nodes, num_nodes))

    for i, children in enumerate(adj_list):
        for child in children:
            A[i][child] = 1

    eigenvalues, eigenvectors = np.linalg.eig(A)
    centrality = np.abs(eigenvectors[:, np.argmax(eigenvalues.real)])

    P = np.outer(centrality, centrality) / np.max(np.outer(centrality, centrality))
    return P





def cybsec_l(c, d, p, q, V, r, budget, g):
    num_nodes = len(d)
    num_safeguards = len(q)
    num_levels = len(c[0])
    implemented_safeguards = np.full((num_nodes, num_safeguards, num_levels), False, dtype=bool)

    # Initialize Gurobi model
    model = Model("Minimize_Cost")
    
    # Define decision variables
    u = model.addVars(num_nodes, num_safeguards, num_levels, vtype=GRB.BINARY, name="u")
    v = model.addVars(num_nodes, num_safeguards, num_levels, lb=0, name="v")
    W = model.addVars(num_nodes, lb=0, name="W")
    Z = model.addVars(num_nodes, lb=0, name="Z")
    
    # Constraints
    for i in range(num_nodes):
        for j in range(num_safeguards):
            model.addConstr(sum(u[i, j, l] for l in range(num_levels)) == 1)
    
    for i in range(num_nodes):
        model.addConstr(sum(v[i, 0, l] for l in range(num_levels)) == V[i])
    
    for i in range(num_nodes):
        for j in range(num_safeguards):
            for l in range(num_levels):
                model.addConstr(v[i, j, l] <= u[i, j, l])
    
    for i in range(num_nodes):
        for j in range(num_safeguards - 1):
            model.addConstr(sum(r[i][j][l] * v[i, j, l] for l in range(num_levels)) == sum(v[i, j + 1, l] for l in range(num_levels)))
    
    for i in range(num_nodes):
        model.addConstr(W[i] == sum(r[i][num_safeguards - 1][l] * v[i, num_safeguards - 1, l] for l in range(num_levels)))
    
    # Budget constraint
    model.addConstr(sum(c[j][l] * u[i, j, l] for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels)) <= budget)
    
    # Propagation constraint
    for i in range(num_nodes):
        propagated_vulnerability = sum(g[h, i] * W[h] for h in range(num_nodes) if g[h, i] > 0)
        model.addConstr(Z[i] >= propagated_vulnerability + W[i])
    
    # Objective function
    model.setObjective(
        sum(c[j][l] * u[i, j, l] for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels)) +
        sum(p[i] * d[i] * Z[i] for i in range(num_nodes)), GRB.MINIMIZE
    )
    
    # Solve the problem
    model.optimize()

    print("Status:", model.Status)
    print("Final Decision Variables:")
    for i in range(num_nodes):
        for j in range(num_safeguards):
            for l in range(1, num_levels):  # Skip l=0 since it's always 1 due to constraint
                if u[i, j, l].X > 0:
                    print(f"u[{i}][{j}][{l}] = {u[i, j, l].X}")
                    implemented_safeguards[i,j,l] = True
    
    final_safeguards = np.argwhere(implemented_safeguards)
    
    # Print results
    if model.status == GRB.OPTIMAL:
        print("Optimal Solution Found")
        cybersecurity_investment = sum(c[j][l] * u[i, j, l].x for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels))
        cybersecurity_gained = sum(p[i] * d[i] * V[i] * (1 - np.prod([sum(r[i][j][l_prime] * u[i, j, l_prime].x for l_prime in range(num_levels)) for j in range(num_safeguards) for l in range(num_levels)])) for i in range(num_nodes))
        cybersecurity_value = cybersecurity_gained - cybersecurity_investment
        cybersecurity_ratio = cybersecurity_gained / cybersecurity_investment if cybersecurity_investment > 0 else 0
        minimized_value = model.objVal
        print("Cybersecurity Investment:", cybersecurity_investment / 1000)
        print("Cybersecurity Value:", cybersecurity_value / 1000)
        print("Cybersecurity Ratio:", cybersecurity_ratio)
        print("Minimized Value of E:", minimized_value / 1000)
        print("Time taken for optimization (CPU seconds):", model.Runtime)
        print("Final Safeguards:")
        for i, j, l in final_safeguards:
            print((i + 1, j + 1, l))

        return cybersecurity_investment, cybersecurity_value, cybersecurity_ratio, minimized_value, final_safeguards, model.Runtime
    else:
        print("No optimal solution found.")
        return None, None, None, None, None, None





def cybsec_bw(c, q, V, r, budget, lambda_val, g):
    num_nodes = len(V)
    num_safeguards = len(q)
    num_levels = len(c[0])
    
    model = gp.Model("Minimize_Cost")
    
    # Decision variables
    u = model.addVars(num_nodes, num_safeguards, num_levels, vtype=GRB.BINARY, name="u")
    v = model.addVars(num_nodes, num_safeguards, num_levels, vtype=GRB.CONTINUOUS, lb=0, name="v")
    W = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, lb=0, name="W")
    W_min = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="W_min")
    W_max = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="W_max")
    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    Z = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, lb=0, name="Z")
    
    # Constraints
    for i in range(num_nodes):
        for j in range(num_safeguards):
            model.addConstr(gp.quicksum(u[i, j, l] for l in range(num_levels)) == 1)
    
    for i in range(num_nodes):
        model.addConstr(gp.quicksum(v[i, 0, l] for l in range(num_levels)) == V[i])
    
    for i in range(num_nodes):
        for j in range(num_safeguards):
            for l in range(num_levels):
                model.addConstr(v[i, j, l] <= u[i, j, l])
    
    for i in range(num_nodes):
        for j in range(num_safeguards - 1):
            model.addConstr(gp.quicksum(r[i][j][l] * v[i, j, l] for l in range(num_levels)) ==
                            gp.quicksum(v[i, j + 1, l] for l in range(num_levels)))
    
    for i in range(num_nodes):
        model.addConstr(W[i] == gp.quicksum(r[i][num_safeguards - 1][l] * v[i, num_safeguards - 1, l] for l in range(num_levels)))
    
    for i in range(num_nodes):
        model.addConstr(Z[i] == gp.quicksum(g[h][i] * W[h] for h in range(num_nodes)) + W[i])

    for i in range(num_nodes):
        model.addConstr(W_max >= Z[i])
        model.addConstr(W_min >= Z[i] - (1 - x[i]))
    
    model.addConstr(gp.quicksum(x[i] for i in range(num_nodes)) == 1)
    
    model.addConstr(gp.quicksum(c[j][l] * u[i, j, l] for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels)) <= budget)
    
    model.setObjective((1 - lambda_val) * W_min + lambda_val * W_max, GRB.MINIMIZE)
    
    start_time = time.process_time()
    model.optimize()
    end_time = time.process_time()
    
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
    else:
        print("No optimal solution.")
    
    cybersecurity_investment = sum(c[j][l] * u[i, j, l].x for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels))
    minimized_value = model.objVal
    
    return cybersecurity_investment, W_min.x, W_max.x, minimized_value, end_time - start_time





def scybsecl_pmax(c, d, p, q, V, r, budget, g):
    num_nodes = len(V)
    num_safeguards = len(c)
    num_levels = len(c[0])

    # Initialize model
    model = Model("Minimize_Pmax")
    model.setParam('Threads', 12)
    
    # Define decision variables
    u = model.addVars(num_nodes, num_safeguards, num_levels, vtype=GRB.BINARY, name="u")
    v = model.addVars(num_nodes, num_safeguards, num_levels, lb=0, name="v")
    W = model.addVars(num_nodes, lb=0, name="W")
    Pmax = model.addVar(lb=0, name="Pmax")

    # Define constraints
    for i in range(num_nodes):
        for j in range(num_safeguards):
            model.addConstr(quicksum(u[i, j, l] for l in range(num_levels)) == 1)

    for i in range(num_nodes):
        model.addConstr(quicksum(v[i, 0, l] for l in range(num_levels)) == V[i])

    for i in range(num_nodes):
        for j in range(num_safeguards):
            for l in range(num_levels):
                model.addConstr(v[i, j, l] <= u[i, j, l])

    for i in range(num_nodes):
        for j in range(num_safeguards - 1):
            model.addConstr(quicksum(r[i][j][l] * v[i, j, l] for l in range(num_levels)) == quicksum(v[i, j + 1, l] for l in range(num_levels)))

    for i in range(num_nodes):
        model.addConstr(W[i] == quicksum(r[i][num_safeguards - 1][l] * v[i, num_safeguards - 1, l] for l in range(num_levels)))

    model.addConstr(quicksum(c[j][l] * u[i, j, l] for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels)) <= budget)

    for i in range(num_nodes):
        model.addConstr(p[i] * W[i] + quicksum(g[h][i] * p[h] * W[h] for h in range(num_nodes)) <= Pmax)

    # Set objective
    model.setObjective(Pmax, GRB.MINIMIZE)

    # Start timer
    start_time = time.process_time()
    model.optimize()
    end_time = time.process_time()

    # Print the status
    if model.status == GRB.OPTIMAL:
        print("Optimal Solution Found")
    else:
        print("No Optimal Solution Found")

    cybersecurity_investment = sum(c[j][l] * u[i, j, l].X for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels))
    cybersecurity_investment_per_node = [(sum(c[j][l] * u[i, j, l].X for j in range(num_safeguards) for l in range(num_levels)))/1000 for i in range(num_nodes)]
    breach_prob_per_node = [p[i] * W[i].X + sum(g[h][i] * p[h] * W[h].X for h in range(num_nodes)) for i in range(num_nodes)]
    exact_breach_prob_per_node = [ 1 - (1 - p[i] * W[i].X) * np.prod([(1 - g[h][i] * W[h].X) for h in range(num_nodes)]) for i in range(num_nodes)]
    minimized_value = model.ObjVal
    final_W_values =  [W[i].X for i in range(num_nodes)]

    print("Cybersecurity Investment:", cybersecurity_investment / 1000)
    print("Objective function:", minimized_value / 1000)
    print("Pmax:", Pmax.X)
    # print("Time taken for optimization (CPU seconds):", end_time - start_time)

    return cybersecurity_investment, minimized_value, Pmax.X, cybersecurity_investment_per_node, breach_prob_per_node, exact_breach_prob_per_node, final_W_values





def scybsecl_lmax(c, d, p, q, V, r, budget, g):
    num_nodes = len(V)
    num_safeguards = len(c)
    num_levels = len(c[0])

    # Initialize model
    model = Model("Minimize_Lmax")
    model.setParam('Threads', 12)
    
    # Define decision variables
    u = model.addVars(num_nodes, num_safeguards, num_levels, vtype=GRB.BINARY, name="u")
    v = model.addVars(num_nodes, num_safeguards, num_levels, lb=0, name="v")
    W = model.addVars(num_nodes, lb=0, name="W")
    Lmax = model.addVar(lb=0, name="Lmax")

    # Define constraints
    for i in range(num_nodes):
        for j in range(num_safeguards):
            model.addConstr(quicksum(u[i, j, l] for l in range(num_levels)) == 1)

    for i in range(num_nodes):
        model.addConstr(quicksum(v[i, 0, l] for l in range(num_levels)) == V[i])

    for i in range(num_nodes):
        for j in range(num_safeguards):
            for l in range(num_levels):
                model.addConstr(v[i, j, l] <= u[i, j, l])

    for i in range(num_nodes):
        for j in range(num_safeguards - 1):
            model.addConstr(quicksum(r[i][j][l] * v[i, j, l] for l in range(num_levels)) == quicksum(v[i, j + 1, l] for l in range(num_levels)))

    for i in range(num_nodes):
        model.addConstr(W[i] == quicksum(r[i][num_safeguards - 1][l] * v[i, num_safeguards - 1, l] for l in range(num_levels)))

    model.addConstr(quicksum(c[j][l] * u[i, j, l] for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels)) <= budget)

    for i in range(num_nodes):
        model.addConstr(p[i] * W[i] + quicksum(g[h][i] * p[h] * W[h] for h in range(num_nodes)) <= Lmax * (1/d[i]))

    # Set objective
    model.setObjective(Lmax, GRB.MINIMIZE)

    # Start timer
    start_time = time.process_time()
    model.optimize()
    end_time = time.process_time()

    # Print the status
    if model.status == GRB.OPTIMAL:
        print("Optimal Solution Found")
    else:
        print("No Optimal Solution Found")

    cybersecurity_investment = sum(c[j][l] * u[i, j, l].X for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels))
    cybersecurity_investment_per_node = [(sum(c[j][l] * u[i, j, l].X for j in range(num_safeguards) for l in range(num_levels)))/1000 for i in range(num_nodes)]
    breach_prob_per_node_1 = [p[i] * W[i].X + sum(g[h][i] * p[h] * W[h].X for h in range(num_nodes)) for i in range(num_nodes)]
    breach_prob_per_node = [d[i] * value for value in breach_prob_per_node_1]
    exact_breach_prob_per_node_1 = [ 1 - (1 - p[i] * W[i].X) * np.prod([(1 - g[h][i] * W[h].X) for h in range(num_nodes)]) for i in range(num_nodes)]
    exact_breach_prob_per_node = [d[i] * value for value in exact_breach_prob_per_node_1]
    minimized_value = model.ObjVal
    final_W_values =  [W[i].X for i in range(num_nodes)]

    print("Cybersecurity Investment:", cybersecurity_investment / 1000)
    print("Objective function:", minimized_value / 1000)
    print("Lmax:", Lmax.X)
    # print("Time taken for optimization (CPU seconds):", end_time - start_time)

    return cybersecurity_investment, minimized_value, Lmax.X, cybersecurity_investment_per_node, breach_prob_per_node, exact_breach_prob_per_node, final_W_values





def scybsecl_qmin(c, d, p, q, V, r, budget, g):
    num_nodes = len(V)
    num_safeguards = len(c)
    num_levels = len(c[0])

    # Initialize model
    model = Model("Maximize_Qmin")
    model.setParam('Threads', 12)
    
    # Define decision variables
    u = model.addVars(num_nodes, num_safeguards, num_levels, vtype=GRB.BINARY, name="u")
    v = model.addVars(num_nodes, num_safeguards, num_levels, lb=0, name="v")
    W = model.addVars(num_nodes, lb=0, name="W")
    Qmin = model.addVar(lb=0, name="Qmin")

    # Define constraints
    for i in range(num_nodes):
        for j in range(num_safeguards):
            model.addConstr(quicksum(u[i, j, l] for l in range(num_levels)) == 1)

    for i in range(num_nodes):
        model.addConstr(quicksum(v[i, 0, l] for l in range(num_levels)) == V[i])

    for i in range(num_nodes):
        for j in range(num_safeguards):
            for l in range(num_levels):
                model.addConstr(v[i, j, l] <= u[i, j, l])

    for i in range(num_nodes):
        for j in range(num_safeguards - 1):
            model.addConstr(quicksum(r[i][j][l] * v[i, j, l] for l in range(num_levels)) == quicksum(v[i, j + 1, l] for l in range(num_levels)))

    for i in range(num_nodes):
        model.addConstr(W[i] == quicksum(r[i][num_safeguards - 1][l] * v[i, num_safeguards - 1, l] for l in range(num_levels)))

    model.addConstr(quicksum(c[j][l] * u[i, j, l] for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels)) <= budget)

    for i in range(num_nodes):
        model.addConstr(p[i] * W[i] + quicksum(g[h][i] * p[h] * W[h] for h in range(num_nodes)) <= 1 - Qmin)

    # Set objective
    model.setObjective(Qmin, GRB.MAXIMIZE)

    # Start timer
    # start_time = time.process_time()
    model.optimize()
    # end_time = time.process_time()

    # Print the status
    if model.status == GRB.OPTIMAL:
        print("Optimal Solution Found")
    else:
        print("No Optimal Solution Found")

    cybersecurity_investment = sum(c[j][l] * u[i, j, l].X for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels))
    cybersecurity_investment_per_node = [(sum(c[j][l] * u[i, j, l].X for j in range(num_safeguards) for l in range(num_levels)))/1000 for i in range(num_nodes)]
    breach_prob_per_node_1 = [p[i] * W[i].X + sum(g[h][i] * p[h] * W[h].X for h in range(num_nodes)) for i in range(num_nodes)]
    breach_prob_per_node = [1 - value for value in breach_prob_per_node_1]
    exact_breach_prob_per_node = [(1 - p[i] * W[i].X) * np.prod([(1 - g[h][i] * W[h].X) for h in range(num_nodes)]) for i in range(num_nodes)]
    # exact_breach_prob_per_node = [d[i] * value for value in exact_breach_prob_per_node_1]
    maximized_value = model.ObjVal
    final_W_values =  [W[i].X for i in range(num_nodes)]

    print("Cybersecurity Investment:", cybersecurity_investment / 1000)
    print("Objective function:", maximized_value / 1000)
    print("Qmin:", Qmin.X)
    # print("Time taken for optimization (CPU seconds):", end_time - start_time)

    return cybersecurity_investment, maximized_value, Qmin.X, cybersecurity_investment_per_node, breach_prob_per_node, exact_breach_prob_per_node, final_W_values





def scybsecl_smin(c, d, p, q, V, r, budget, g):
    num_nodes = len(V)
    num_safeguards = len(c)
    num_levels = len(c[0])

    # Initialize model
    model = Model("Maximize_Smin")
    model.setParam('Threads', 12)
    
    # Define decision variables
    u = model.addVars(num_nodes, num_safeguards, num_levels, vtype=GRB.BINARY, name="u")
    v = model.addVars(num_nodes, num_safeguards, num_levels, lb=0, name="v")
    W = model.addVars(num_nodes, lb=0, name="W")
    Smin = model.addVar(lb=0, name="Smin")

    # Define constraints
    for i in range(num_nodes):
        for j in range(num_safeguards):
            model.addConstr(quicksum(u[i, j, l] for l in range(num_levels)) == 1)

    for i in range(num_nodes):
        model.addConstr(quicksum(v[i, 0, l] for l in range(num_levels)) == V[i])

    for i in range(num_nodes):
        for j in range(num_safeguards):
            for l in range(num_levels):
                model.addConstr(v[i, j, l] <= u[i, j, l])

    for i in range(num_nodes):
        for j in range(num_safeguards - 1):
            model.addConstr(quicksum(r[i][j][l] * v[i, j, l] for l in range(num_levels)) == quicksum(v[i, j + 1, l] for l in range(num_levels)))

    for i in range(num_nodes):
        model.addConstr(W[i] == quicksum(r[i][num_safeguards - 1][l] * v[i, num_safeguards - 1, l] for l in range(num_levels)))

    model.addConstr(quicksum(c[j][l] * u[i, j, l] for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels)) <= budget)

    for i in range(num_nodes):
        model.addConstr(p[i] * W[i] + quicksum(g[h][i] * p[h] * W[h] for h in range(num_nodes)) <= 1 - Smin * (1/d[i]))

    # Set objective
    model.setObjective(Smin, GRB.MAXIMIZE)

    # Start timer
    # start_time = time.process_time()
    model.optimize()
    # end_time = time.process_time()

    # Print the status
    if model.status == GRB.OPTIMAL:
        print("Optimal Solution Found")
    else:
        print("No Optimal Solution Found")

    cybersecurity_investment = sum(c[j][l] * u[i, j, l].X for i in range(num_nodes) for j in range(num_safeguards) for l in range(num_levels))
    cybersecurity_investment_per_node = [(sum(c[j][l] * u[i, j, l].X for j in range(num_safeguards) for l in range(num_levels)))/1000 for i in range(num_nodes)]
    breach_prob_per_node_1 = [p[i] * W[i].X + sum(g[h][i] * p[h] * W[h].X for h in range(num_nodes)) for i in range(num_nodes)]
    breach_prob_per_node = [d[i] * (1 - value) for value in breach_prob_per_node_1]
    exact_breach_prob_per_node_1 = [(1 - p[i] * W[i].X) * np.prod([(1 - g[h][i] * W[h].X) for h in range(num_nodes)]) for i in range(num_nodes)]
    exact_breach_prob_per_node = [d[i] * value for value in exact_breach_prob_per_node_1]
    maximized_value = model.ObjVal
    final_W_values =  [W[i].X for i in range(num_nodes)]

    print("Cybersecurity Investment:", cybersecurity_investment / 1000)
    print("Objective function:", maximized_value / 1000)
    print("Smin:", Smin.X)
    # print("Time taken for optimization (CPU seconds):", end_time - start_time)

    return cybersecurity_investment, maximized_value, Smin.X, cybersecurity_investment_per_node, breach_prob_per_node, exact_breach_prob_per_node, final_W_values





def cyberport_slp(c, d, V, q, P, r, g):
    num_controls = len(c)
    num_levels = len(c[0])
    num_components = len(V)
    num_scenarios = len(P)

    model = gp.Model("Cyberport_SLP_Minimize_Cost")
    model.setParam('OutputFlag', 0)  # Silence solver
    model.setParam('Threads', 12)

    u = model.addVars(num_controls, num_levels, vtype=GRB.BINARY, name="u")
    v = model.addVars(num_controls, num_components, num_levels, vtype=GRB.CONTINUOUS, lb=0, name="v")
    W = model.addVars(num_components, vtype=GRB.CONTINUOUS, lb=0, name="W")
    Z = model.addVars(num_components, vtype=GRB.CONTINUOUS, lb=0, name="Z")

    model.setObjective(
        gp.quicksum(c[j][l] * u[j, l] for j in range(num_controls) for l in range(num_levels)) +
        gp.quicksum(P[s] * d[k] * Z[k] for s in range(num_scenarios) for k in range(num_components)),
        GRB.MINIMIZE
    )

    for j in range(num_controls):
        model.addConstr(gp.quicksum(u[j, l] for l in range(num_levels)) == 1)

    for k in range(num_components):
        model.addConstr(gp.quicksum(v[0, k, l] for l in range(num_levels)) == V[k])

    for j in range(num_controls - 1):
        for k in range(num_components):
            model.addConstr(
                gp.quicksum(r[j][k][l] * v[j, k, l] for l in range(num_levels)) ==
                gp.quicksum(v[j + 1, k, l] for l in range(num_levels))
            )

    for k in range(num_components):
        model.addConstr(
            gp.quicksum(r[num_controls - 1][k][l] * v[num_controls - 1, k, l] for l in range(num_levels)) == W[k]
        )

    for j in range(num_controls):
        for k in range(num_components):
            for l in range(num_levels):
                model.addConstr(v[j, k, l] <= u[j, l])

    for i in range(num_components):
        propagated = gp.quicksum(g[h][i] * W[h] for h in range(num_components) if g[h][i] > 0)
        model.addConstr(Z[i] == propagated + W[i])

    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None, None, None, None, None

    implemented_controls = []
    C = 0
    for j in range(num_controls):
        for l in range(num_levels):
            if u[j, l].X > 0.5:
                if l > 0:
                    implemented_controls.append(j + 1)
                    C += c[j][l]

    D = sum(P[s] * d[k] * W[k].X for s in range(num_scenarios) for k in range(num_components))
    E = model.ObjVal
    return C, D, E, 0, implemented_controls




# === Scenario and Probability Calculations ===
def calculate_scenario_probabilities(I, pi):
    num_scenarios = 2 ** len(I)
    scenario_probabilities = {}
    for s in range(num_scenarios):
        bits = [int(b) for b in f"{s:0{len(I)}b}"]
        Ps = np.prod([pi[i] if bits[i] else (1 - pi[i]) for i in range(len(I))])
        scenario_probabilities[s] = Ps
    return scenario_probabilities





def cyberport_ubp(c, d, V, q, h, pk, g):
    num_controls = len(c)
    num_components = len(V)

    print("h")
    print(h)
    print("c")
    print(c)

    u = np.zeros(num_controls)
    for j in range(num_controls):
        if h[j] > c[j][1]:
            u[j] = 1

    # Extract implemented controls and calculate cybersecurity metrics
    implemented_controls = []
    C = 0  # Cybersecurity Investment
    D = 0
    for j in range(num_controls):
        if u[j] == 1:
            implemented_controls.append((j+1))

    for j in range(num_controls):
        if u[j] == 1:
            C += c[j][1]

    for j in range(num_controls):
        if (u[j]==0):
            for k in range(num_components):
                mult = 1.0
                for n in range(num_components):
                    if g[n][k] > 0:
                        mult *= 1 + g[n][k]
                D += pk[k] * d[k] * mult
        else:
            for k in range(num_components):
                mult = 1.0
                for n in range(num_components):
                    if g[n][k] > 0:
                        mult *= 1 + g[n][k]
                D += pk[k] * d[k] * q[j] * mult

    # Calculate total objective function (E)
    E = C + D

    # Display the results
    print()
    # print("Status:", pulp.LpStatus[prob.status])
    print("Cybersecurity Investment (C):", C/1000)
    print("Expected Cost of Losses (D):", D/1000)
    print("Objective Function (E):", E/1000)
    # print("Cybersecurity Value of Control Portfolio (H):", H)
    print("Implemented Controls:", implemented_controls)
    print()


    return C, D, E, implemented_controls





def calculate_h(P, d, r, V, c):
    num_controls = len(c)
    num_components = len(V)
    h = [0] * num_controls
    for j in range(num_controls):
        h[j] = sum(P[s] * d[k] * (1 - r[j][k][1]) * V[k] for s in P for k in range(num_components))
        h[j] *= 1e7
    return h





def run_program():
    c_light_temp = [10, 20, 10, 35, 20, 10, 50, 45, 10, 30, 15, 40, 10, 60, 62, 58, 20, 40, 26, 10]
    c_light = [value * 1000 for value in c_light_temp]
    c_medium = [5 * cost for cost in c_light]
    c_strong = [10 * cost for cost in c_light]
    c = [[0, c_light[i], c_medium[i], c_strong[i]] for i in range(len(c_light))]
    e_temp = [6.09209, 1.89873, 9.21892, 9.57156, 1.05726, 7.14106, 5.51532, 2.63135, 3.49604, 4.07247, 6.65212, 5.75807, 9.42022, 3.63525, 0.0308876, 7.55598, 4.50103, 1.70122, 7.87748, 8.37808]
    e = [value * 1e-5 for value in e_temp]
    p = [0.35, 0.40, 0.35, 0.25, 0.40, 0.25, 0.55, 0.55, 0.75, 0.75]
    d_temp = [450, 1500, 550, 300, 1200, 350, 2500, 2500, 10000, 10000]
    d = [value * 1000 for value in d_temp]
    V = [0.6713, 0.7705, 0.6691, 0.5067, 0.7799, 0.5282, 0.8976, 0.8821, 0.9772, 0.9939]
    # Define the adjacency matrix for g (replace the values with your specified probabilities)
    children = [[3],
                [3,4,5],
                [5],
                [6],
                [6,7],
                [7],
                [8,9],
                [8,9],
                [],
                []]
    h_parents = [[],
        [],
        [],
        [0,1],
        [1],
        [1,2],
        [3,4],
        [4,5],
        [6,7],
        [6,7]]

    r = calculate_r(V, e, c)
    # q = 1/len(c_light_temp)**2
    q = 1
    
    g_binary_temp = binary_reachability(children).tolist()
    g_path_count_temp = normalized_path_count(children).tolist()
    g_inverse_temp = inverse_shortest_path(children).tolist()
    g_decay_temp = decay_factor_over_distance(children, lambda_=0.5).tolist()
    g_weighted_path_temp = weighted_path_contribution(children).tolist()
    g_heat_temp = heat_diffusion_model(children).tolist()
    g_random_walk_temp = random_walk_influence(children, lambda_=0.85).tolist()
    g_eigenvector_temp = eigenvector_centrality_influence(children).tolist()

    g_binary = [[q * value for value in row]  for row in g_binary_temp]
    g_path_count = [[q * value for value in row]  for row in g_path_count_temp]
    g_inverse = [[q * value for value in row]  for row in g_inverse_temp]
    g_decay = [[q * value for value in row]  for row in g_decay_temp]
    g_weighted_path = [[q * value for value in row]  for row in g_weighted_path_temp]
    g_heat = [[q * value for value in row]  for row in g_heat_temp]
    g_random_walk = [[q * value for value in row]  for row in g_random_walk_temp]
    g_eigenvector = [[q * value for value in row]  for row in g_eigenvector_temp]

    g_unconnected = [[0 for _ in range(I)] for _ in range(I)]
    # print(g_unconnected)

    g_options = [g_unconnected, g_binary, g_path_count, g_inverse, g_decay, g_weighted_path, g_heat, g_random_walk, g_eigenvector]
    g_option_names = ["Unconnected", "Binary Reachability", "Normalized Path Count", "Inverse Shortest Path", "Decay Factor Over Distance", "Weighted Path Contribution", "Heat Diffusion", "Random Walk", "Eigenvector Centrality"]

    # budgets_temp = [500, 1000, 2500, 5000, 10000, 15000, float('inf')]
    budgets_temp = [500, float('inf')]
    budgets = [value*1000 if value != float('inf') else value for value in budgets_temp]

    # for cybsec_bw
    lambda_vals = [0, 0.25, 0.5, 0.75, 1]

    # Initialize storage dictionaries
    results = {
        "cybsec_l": {
            "investment": [],
            "value": [],
            "ratio": [],
            "minimized_value": [],
            "final_safeguards": [],
            "cpu_time": [],
        },
        "cybsec_bw": {},  # 3D Dictionary (budget -> lambda_val -> values)
        "scybsecl_pmax": {
            "investment": [],
            "minimized_value": [],
            "pmax": [],
            "investment_per_node": [],
            "breach_prob_per_node": [],
            "exact_breach_prob_per_node": [],
            "final_W_values": [],
        },
        "scybsecl_lmax": {
            "investment": [],
            "minimized_value": [],
            "lmax": [],
            "investment_per_node": [],
            "breach_prob_per_node": [],
            "exact_breach_prob_per_node": [],
            "final_W_values": [],
        },
        "scybsecl_qmin": {
            "investment": [],
            "minimized_value": [],
            "qmin": [],
            "investment_per_node": [],
            "breach_prob_per_node": [],
            "exact_breach_prob_per_node": [],
            "final_W_values": [],
        },
        "scybsecl_smin": {
            "investment": [],
            "minimized_value": [],
            "smin": [],
            "investment_per_node": [],
            "breach_prob_per_node": [],
            "exact_breach_prob_per_node": [],
            "final_W_values": [],
        },
        "cyberport_slp": {
            "C": [],
            "D": [],
            "E": [],
            "H": [],
            "implemented_controls": [],
        },
        "cyberport_ubp": {
            "C": [],
            "D": [],
            "E": [],
            "implemented_controls": [],
        },
    }


    for budget in budgets:
        for i in range(len(g_options)):
            g = g_options[i]
            g_name = g_option_names[i]

            print("---------IMPORTANT----------")
            print(f"Budget: {budget}")
            print(f"Graph Algorithm: {g_name}")
            print(f"Optimizer: Cybsec_L")
            print("----------------------------")

            # cybsec_l results
            cybersecurity_investment_1, cybersecurity_value_1, cybersecurity_ratio_1, minimized_value_1, final_safeguards_1, cpu_time_1 = cybsec_l(c, d, p, e, V, r, budget, g)
            
            results["cybsec_l"]["investment"].append(cybersecurity_investment_1)
            results["cybsec_l"]["value"].append(cybersecurity_value_1)
            results["cybsec_l"]["ratio"].append(cybersecurity_ratio_1)
            results["cybsec_l"]["minimized_value"].append(minimized_value_1)
            results["cybsec_l"]["final_safeguards"].append(final_safeguards_1)
            results["cybsec_l"]["cpu_time"].append(cpu_time_1)
            
            # cybsec_bw results

            # Initialize 3D dictionary for cybsec_bw
            if budget not in results["cybsec_bw"]:
                results["cybsec_bw"][budget] = {}

            for lambda_val in lambda_vals:
                print("---------IMPORTANT----------")
                print(f"Budget: {budget}")
                print(f"Graph Algorithm: {g_name}")
                print(f"Optimizer: Cybsec_BW")
                print(f"Lambda: {lambda_val}")
                print("----------------------------")
                cybersecurity_investment_2, W_min_2, W_max_2, minimized_value_2, cpu_time_2 = cybsec_bw(c, e, V, r, budget, lambda_val, g)

                if lambda_val not in results["cybsec_bw"][budget]:
                    results["cybsec_bw"][budget][lambda_val] = {
                        "investment": [],
                        "W_min": [],
                        "W_max": [],
                        "minimized_value": [],
                        "cpu_time": [],
                    }

                results["cybsec_bw"][budget][lambda_val]["investment"].append(cybersecurity_investment_2)
                results["cybsec_bw"][budget][lambda_val]["W_min"].append(W_min_2)
                results["cybsec_bw"][budget][lambda_val]["W_max"].append(W_max_2)
                results["cybsec_bw"][budget][lambda_val]["minimized_value"].append(minimized_value_2)
                results["cybsec_bw"][budget][lambda_val]["cpu_time"].append(cpu_time_2)

            print("---------IMPORTANT----------")
            print(f"Budget: {budget}")
            print(f"Graph Algorithm: {g_name}")
            print(f"Optimizer: SCybsecL_Pmax")
            print("----------------------------")

            # scybsecl_pmax results
            cybersecurity_investment_3, minimized_value_3, Pmax_3, cybersecurity_investment_per_node_3, breach_prob_per_node_3, exact_breach_prob_per_node_3, final_W_values_3 = scybsecl_pmax(c, d, p, e, V, r, budget, g)

            results["scybsecl_pmax"]["investment"].append(cybersecurity_investment_3)
            results["scybsecl_pmax"]["minimized_value"].append(minimized_value_3)
            results["scybsecl_pmax"]["pmax"].append(Pmax_3)
            results["scybsecl_pmax"]["investment_per_node"].append(cybersecurity_investment_per_node_3)
            results["scybsecl_pmax"]["breach_prob_per_node"].append(breach_prob_per_node_3)
            results["scybsecl_pmax"]["exact_breach_prob_per_node"].append(exact_breach_prob_per_node_3)
            results["scybsecl_pmax"]["final_W_values"].append(final_W_values_3)

            print("---------IMPORTANT----------")
            print(f"Budget: {budget}")
            print(f"Graph Algorithm: {g_name}")
            print(f"Optimizer: SCybsecL_Lmax")
            print("----------------------------")

            # scybsecl_lmax results
            cybersecurity_investment_4, minimized_value_4, Lmax_4, cybersecurity_investment_per_node_4, breach_prob_per_node_4, exact_breach_prob_per_node_4, final_W_values_4 = scybsecl_lmax(c, d, p, e, V, r, budget, g)

            results["scybsecl_lmax"]["investment"].append(cybersecurity_investment_4)
            results["scybsecl_lmax"]["minimized_value"].append(minimized_value_4)
            results["scybsecl_lmax"]["lmax"].append(Lmax_4)
            results["scybsecl_lmax"]["investment_per_node"].append(cybersecurity_investment_per_node_4)
            results["scybsecl_lmax"]["breach_prob_per_node"].append(breach_prob_per_node_4)
            results["scybsecl_lmax"]["exact_breach_prob_per_node"].append(exact_breach_prob_per_node_4)
            results["scybsecl_lmax"]["final_W_values"].append(final_W_values_4)

            print("---------IMPORTANT----------")
            print(f"Budget: {budget}")
            print(f"Graph Algorithm: {g_name}")
            print(f"Optimizer: SCybsecL_Qmin")
            print("----------------------------")

            # scybsecl_qmin results
            cybersecurity_investment_5, minimized_value_5, Qmin_5, cybersecurity_investment_per_node_5, breach_prob_per_node_5, exact_breach_prob_per_node_5, final_W_values_5 = scybsecl_qmin(c, d, p, e, V, r, budget, g)

            results["scybsecl_qmin"]["investment"].append(cybersecurity_investment_5)
            results["scybsecl_qmin"]["minimized_value"].append(minimized_value_5)
            results["scybsecl_qmin"]["qmin"].append(Qmin_5)
            results["scybsecl_qmin"]["investment_per_node"].append(cybersecurity_investment_per_node_5)
            results["scybsecl_qmin"]["breach_prob_per_node"].append(breach_prob_per_node_5)
            results["scybsecl_qmin"]["exact_breach_prob_per_node"].append(exact_breach_prob_per_node_5)
            results["scybsecl_qmin"]["final_W_values"].append(final_W_values_5)

            print("---------IMPORTANT----------")
            print(f"Budget: {budget}")
            print(f"Graph Algorithm: {g_name}")
            print(f"Optimizer: SCybsecL_Smin")
            print("----------------------------")

            # scybsecl_smin results
            cybersecurity_investment_6, minimized_value_6, Smin_6, cybersecurity_investment_per_node_6, breach_prob_per_node_6, exact_breach_prob_per_node_6, final_W_values_6 = scybsecl_smin(c, d, p, e, V, r, budget, g)

            results["scybsecl_smin"]["investment"].append(cybersecurity_investment_6)
            results["scybsecl_smin"]["minimized_value"].append(minimized_value_6)
            results["scybsecl_smin"]["smin"].append(Smin_6)
            results["scybsecl_smin"]["investment_per_node"].append(cybersecurity_investment_per_node_6)
            results["scybsecl_smin"]["breach_prob_per_node"].append(breach_prob_per_node_6)
            results["scybsecl_smin"]["exact_breach_prob_per_node"].append(exact_breach_prob_per_node_6)
            results["scybsecl_smin"]["final_W_values"].append(final_W_values_6)

            I = list(range(10))
            J = list(range(20))
            K = list(range(10))

            Vk = [0.771269, 0.760485, 0.669096, 0.50657, 0.57994, 0.682282, 0.697596, 0.992109, 0.977199, 0.993963]
            cj1 = [10, 20, 10, 35, 20, 10, 50, 45, 10, 30, 15, 40, 10, 60, 62, 58, 20, 40, 26, 10]
            pi = [0.35, 0.25, 0.15, 0.25, 0.20, 0.25, 0.50, 0.35, 0.40, 0.003]
            pk = [0.55, 0.003, 0.76, 0.51, 0.70, 0.88, 0.40, 0.90, 0.55, 0.64]
            dk = [1000 * value for value in [24, 122, 350, 5, 250, 20, 20, 25, 30, 10000]]
            q = [val * 1e-5 for val in [6.09209, 1.89873, 9.21892, 9.57156, 1.05726, 7.14106, 5.51532, 2.63135, 3.49604, 4.07247,
                                        6.65212, 5.75807, 9.42022, 3.63525, 0.0308876, 7.55598, 4.50103, 1.70122, 7.87748, 8.37808]]
            P = calculate_scenario_probabilities(I, pi)
            q1 = 1

            implementation_levels = ["light", "medium", "strong"]

            for level in implementation_levels:
                c = []
                multiplier = 1000 if level == "light" else 5*1000 if level == "medium" else 10*1000
                c = [[0, cj1[i] * multiplier] for i in range(len(cj1))]

                if budget != float('inf'):
                    c = [[0, cost if cost <= budget else 1000000000] for [_, cost] in c]

                r = np.zeros((len(c), len(Vk), len(c[0])))
                for j in range(len(c)):
                    for k in range(len(Vk)):
                        for l in range(len(c[j])):
                            r[j][k][l] = Vk[k] ** (q[j] * c[j][l])

                C_7, D_7, E_7, H_7, implemented_controls_7 = cyberport_slp(c, dk, Vk, q, P, r, g)
            
            for level in implementation_levels:
                # Adjust cost based on implementation level
                factor = 1000 if level == "light" else 5000 if level == "medium" else 10000
                c = [[0, cj1[i] * factor] for i in range(len(cj1))]

                # Apply budget constraint
                if budget != float('inf'):
                    for i in range(len(c)):
                        if c[i][1] > budget:
                            c[i][1] = 0

                # Build r matrix based on g_option
                r = np.zeros((len(c), len(Vk), 2))
                for j in range(len(c)):
                    for k in range(len(Vk)):
                        r[j][k][1] = Vk[k] ** (q[j] * c[j][1]) if isinstance(g, list) else 0

                h = calculate_h(P, dk, r, Vk, c)
                C_8, D_8, E_8, controls = cyberport_ubp(c, dk, Vk, q, h, pk, g)






if __name__=="__main__":
    run_program()



