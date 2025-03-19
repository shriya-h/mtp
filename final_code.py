from gurobipy import Model, GRB, quicksum
import numpy as np
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from collections import deque





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





def cyberport_slp(c, d, V, q, P, r, g):
    num_controls = len(c)
    num_levels = len(c[0])
    num_components = len(V)
    num_scenarios = len(P)

    # Initialize Gurobi model
    model = gp.Model("Cyberport_SLP_Minimize_Cost")
    
    # Define decision variables
    u = model.addVars(num_controls, num_levels, vtype=GRB.BINARY, name="u")
    v = model.addVars(num_controls, num_components, num_levels, vtype=GRB.CONTINUOUS, lb=0, name="v")
    W = model.addVars(num_components, vtype=GRB.CONTINUOUS, lb=0, name="W")
    Z = model.addVars(num_components, vtype=GRB.CONTINUOUS, lb=0, name="Z")

    # Define the objective function
    model.setObjective(
        gp.quicksum(c[j][l] * u[j, l] for j in range(num_controls) for l in range(num_levels)) +
        gp.quicksum(gp.quicksum(P[s] * d[k] * Z[k] for k in range(num_components)) for s in range(num_scenarios)), 
        GRB.MINIMIZE
    )

    # Control Selection Constraint: Exactly one level l must be selected for each control j
    for j in range(num_controls):
        model.addConstr(gp.quicksum(u[j, l] for l in range(num_levels)) == 1, f"Control_Selection_{j}")

    # Initial Vulnerability Constraint
    for k in range(num_components):
        model.addConstr(gp.quicksum(v[0, k, l] for l in range(num_levels)) == V[k], f"Initial_Vulnerability_{k}")

    # Intermediate Vulnerability Constraint
    for j in range(num_controls - 1):
        for k in range(num_components):
            model.addConstr(
                gp.quicksum(r[j][k][l] * v[j, k, l] for l in range(num_levels)) ==
                gp.quicksum(v[j + 1, k, l] for l in range(num_levels)),
                f"Intermediate_Vulnerability_{j}_{k}"
            )

    # Final Vulnerability Constraint
    for k in range(num_components):
        model.addConstr(
            gp.quicksum(r[num_controls - 1][k][l] * v[num_controls - 1, k, l] for l in range(num_levels)) == W[k],
            f"Final_Vulnerability_{k}"
        )

    # Implemented Controls Constraint
    for j in range(num_controls):
        for k in range(num_components):
            for l in range(num_levels):
                model.addConstr(v[j, k, l] <= u[j, l], f"Implemented_Controls_{j}_{k}_{l}")

    # Adjust vulnerability values with propagation probabilities
    for i in range(num_components):
        propagated_vulnerability = gp.quicksum(g[h, i] * W[h] for h in range(num_components) if g[h, i] > 0)
        model.addConstr(Z[i] == propagated_vulnerability + W[i], f"Propagation_Constraint_{i}")

    # Solve the problem
    model.optimize()

    # **Check if the model was solved successfully before accessing variable values**
    if model.status != GRB.OPTIMAL:
        print("Optimization was unsuccessful. Status code:", model.status)
        return None, None, None, None, None

    # Extract implemented controls and calculate cybersecurity metrics
    implemented_controls = []
    C = 0  # Cybersecurity Investment
    H = 0  # Cybersecurity value of control portfolio

    for j in range(num_controls):
        for l in range(num_levels):
            if u[j, l].X > 0.5:  # Extract variable value with .X safely
                if l > 0 and not implemented_controls:
                    implemented_controls.append(j + 1)
                elif l > 0 and implemented_controls[-1] != (j + 1):
                    implemented_controls.append(j + 1)
                C += c[j][l]

    # Calculate expected cost of losses (D)
    D = sum(sum(P[s] * d[k] * W[k].X for k in range(num_components)) for s in range(num_scenarios))

    # Calculate total objective function (E)
    E = model.ObjVal
    C_alt = E - D

    # Display the results
    print("\nStatus:", model.Status)
    print("Cybersecurity Investment (C):", C_alt / 1000)
    print("Expected Cost of Losses (D):", D / 1000)
    print("Objective Function (E):", E / 1000)
    print("Implemented Controls:", implemented_controls)
    print()

    return C_alt, D, E, H, implemented_controls





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
    g_temp = random_walk_influence(children)
    q = 1/len(c_light_temp)**2
    g = [[q * value for value in row] for row in g_temp]





if __name__=="__main__":
    run_program()



