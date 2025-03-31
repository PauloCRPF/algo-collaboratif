import tkinter as tk
from tkinter import ttk
import numpy as np
import random
from ant_colony_tsp import Environment
import matplotlib.pyplot as plt
import itertools
from skopt import gp_minimize
from skopt.space import Real, Integer
from concurrent.futures import ProcessPoolExecutor


class TSPInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP")
        
        self.canvas = tk.Canvas(self.root, width=1250, height=575, bg="white")
        self.canvas.pack()
        
        self.route_label = tk.Label(self.root, text="Best Route: ", font=("Poppins", 12))
        self.route_label.pack()
        
        self.points = []
        self.point_objects = []
        self.point_labels = []
        
        self.canvas.bind("<Button-1>", self.add_or_remove_point)
        
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        
        self.undo_button = tk.Button(button_frame, text="Undo", command=self.undo_last_point)
        self.undo_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(button_frame, text="Clear All", command=self.clear_all_points)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.random_entry = tk.Entry(button_frame, width=5)
        self.random_entry.pack(side=tk.LEFT, padx=5)
        self.random_button = tk.Button(button_frame, text="Add Random", command=self.add_random_points)
        self.random_button.pack(side=tk.LEFT, padx=5)
        
        self.solve_button = tk.Button(button_frame, text="Solve TSP", command=self.solve_tsp)
        self.solve_button.pack(side=tk.LEFT, padx=5)

        self.graph_button = tk.Button(button_frame, text="Show Distance Graph", command=self.show_distance_graph)
        self.graph_button.pack(side=tk.LEFT, padx=5)
        
        self.evol_button = tk.Button(button_frame, text="Optimize (Evolution)", command=self.run_evolutionary_optimization)
        self.evol_button.pack(side=tk.LEFT, padx=5)

        self.grid_button = tk.Button(button_frame, text="Optimize (Grid Search)", command=self.run_grid_search_optimization)
        self.grid_button.pack(side=tk.LEFT, padx=5)

        self.bayes_button = tk.Button(button_frame, text="Optimize (Bayesian)", command=self.run_bayesian_optimization)
        self.bayes_button.pack(side=tk.LEFT, padx=5)

        
        param_frame = tk.Frame(self.root)
        param_frame.pack(pady=10)
        
        # 📌 Parameters in one line
        param_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        param_frame.pack()

        tk.Label(param_frame, text="Alpha:").pack(side=tk.LEFT, padx=5)
        self.alpha_entry = tk.Entry(param_frame, width=5)
        self.alpha_entry.pack(side=tk.LEFT, padx=5)
        self.alpha_entry.insert(0, "1.0")

        tk.Label(param_frame, text="Beta:").pack(side=tk.LEFT, padx=5)
        self.beta_entry = tk.Entry(param_frame, width=5)
        self.beta_entry.pack(side=tk.LEFT, padx=5)
        self.beta_entry.insert(0, "2.0")

        tk.Label(param_frame, text="Gamma:").pack(side=tk.LEFT, padx=5)
        self.gamma_entry = tk.Entry(param_frame, width=5)
        self.gamma_entry.pack(side=tk.LEFT, padx=5)
        self.gamma_entry.insert(0, "0.1")
        
        tk.Label(param_frame, text="Iter:").pack(side=tk.LEFT, padx=5)
        self.iter_entry = tk.Entry(param_frame, width=5)
        self.iter_entry.pack(side=tk.LEFT, padx=5)
        self.iter_entry.insert(0, "100")
        
    def run_evolutionary_optimization(self):
        from evolution import run_evolution  # Supondo que exista essa função
        best_alpha, best_beta, best_gamma = run_evolution(self.points)
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, str(best_alpha))
        self.beta_entry.delete(0, tk.END)
        self.beta_entry.insert(0, str(best_beta))
        self.gamma_entry.delete(0, tk.END)
        self.gamma_entry.insert(0, str(best_gamma))
        print(f"Best found by evolution: alpha={best_alpha}, beta={best_beta}, gamma={best_gamma}")
    
    def run_grid_search_optimization(self):
        alphas = [0.5, 1, 2]
        betas = [2, 3, 5]
        gammas = [0.1, 0.3, 0.5]
        best_distance = float('inf')
        best_params = None

        num_cities = len(self.points)
        positions = {i: self.points[i] for i in range(num_cities)}

        print("🔍 Running Grid Search...")

        for a, b, g in itertools.product(alphas, betas, gammas):
            env = Environment(num_cities=num_cities, alpha=a, beta=b, gamma=g, num_ants=50, positions=positions)
            _, dist, _ = env.simulate(num_iterations=200, track_progress=True)
                        
            if dist < best_distance:
                best_distance = dist
                best_params = (a, b, g)

        # Update UI
        a, b, g = best_params
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, str(a))
        self.beta_entry.delete(0, tk.END)
        self.beta_entry.insert(0, str(b))
        self.gamma_entry.delete(0, tk.END)
        self.gamma_entry.insert(0, str(g))

        print(f"✅ Best Grid Search Result: alpha={a}, beta={b}, gamma={g} → distance={best_distance:.2f}")

    
    def run_bayesian_optimization(self):
        print('start bayesian optimization')
        def objective(params):
            a, b, g = params
            env = Environment(
                num_cities=len(self.points),
                alpha=a, beta=b, gamma=g,
                num_ants=50,
                positions={i: self.points[i] for i in range(len(self.points))}
            )
            _, dist= env.simulate(num_iterations=200)
            return dist

        res = gp_minimize(
            func=objective,
            dimensions=[Real(0.1, 5.0), Integer(1, 10), Real(0.01, 0.9)],
            n_calls=20,
            random_state=42
        )

        best_alpha, best_beta, best_gamma = res.x
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, str(best_alpha))
        self.beta_entry.delete(0, tk.END)
        self.beta_entry.insert(0, str(best_beta))
        self.gamma_entry.delete(0, tk.END)
        self.gamma_entry.insert(0, str(best_gamma))

        print(f"Best found by Bayesian: {res.x} with score {res.fun}")


    def generate_random_positions(self, num_cities, x_range=(10, 1220), y_range=(10, 550)):
        return [(np.random.uniform(*x_range), np.random.uniform(*y_range)) for _ in range(num_cities)]
    
    def add_random_points(self):
        try:
            num_points = int(self.random_entry.get())
            new_positions = self.generate_random_positions(num_points)
            for x, y in new_positions:
                self.add_point(x, y)
        except ValueError:
            print("Invalid number of points")
    
    def add_point(self, x, y):
        index = len(self.points)
        point = self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
        label = self.canvas.create_text(x+10, y-10, text=str(index), font=("Arial", 10), fill="blue")
        self.points.append((x, y))
        self.point_objects.append(point)
        self.point_labels.append(label)
        print(f"Point added: ({x}, {y})")
    
    def add_or_remove_point(self, event):
        x, y = event.x, event.y
        for i in range(len(self.points) - 1, -1, -1):
            px, py = self.points[i]
            if abs(px - x) <= 5 and abs(py - y) <= 5:
                self.canvas.delete(self.point_objects[i])
                self.canvas.delete(self.point_labels[i])
                del self.points[i]
                del self.point_objects[i]
                del self.point_labels[i]
                print(f"Point removed: ({px}, {py})")
                return
        self.add_point(x, y)
    
    def undo_last_point(self):
        if self.points:
            last_point = self.points.pop()
            last_obj = self.point_objects.pop()
            last_label = self.point_labels.pop()
            self.canvas.delete(last_obj)
            self.canvas.delete(last_label)
            print(f"Undo last point: {last_point}")
    
    def clear_all_points(self):
        for obj in self.point_objects:
            self.canvas.delete(obj)
        for label in self.point_labels:
            self.canvas.delete(label)
        self.canvas.delete("all")  # Remove everything from the canvas
        self.points.clear()
        self.point_objects.clear()
        self.point_labels.clear()
        self.route_label.config(text="Best Route: ")
        print("All points cleared.")
    
    def solve_tsp(self):
        if len(self.points) < 2:
            print("Not enough points to solve TSP")
            return
        
        num_cities = len(self.points)
        distances = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    distances[i][j] = np.linalg.norm(
                        np.array(self.points[i]) - np.array(self.points[j])
                    )
        
        self.env = Environment(num_cities=num_cities, alpha=float(self.alpha_entry.get()), beta=float(self.beta_entry.get()), gamma=float(self.gamma_entry.get()), num_ants=100, positions={i: self.points[i] for i in range(num_cities)})
        best_path, best_distance, self.distance_progress = self.env.simulate(num_iterations=int(self.iter_entry.get()), track_progress=True)
        
        self.canvas.delete("all")
        
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                x1, y1 = self.points[i]
                x2, y2 = self.points[j]
                self.canvas.create_line(x1, y1, x2, y2, fill="gray", dash=(2, 2))
        
        for index, (x, y) in enumerate(self.points):
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
            self.canvas.create_text(x+10, y-10, text=str(index), font=("Arial", 10), fill="blue")
        
        for i in range(len(best_path) - 1):
            x1, y1 = self.points[best_path[i]]
            x2, y2 = self.points[best_path[i + 1]]
            self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
        
        route_text = " -> ".join(str(i) for i in best_path)
        self.route_label.config(text=f"Best Route: {route_text}, Distance: {best_distance:.2f}")
        
        print(f"Best path: {best_path}, Distance: {best_distance}")
    
    
    def update_parameters(self, event=None):
        """Automatically updates TSP solution when parameters change."""
        self.solve_tsp()
    
    def show_distance_graph(self):
        if not hasattr(self, 'distance_progress') or not self.distance_progress:
            print(self.distance_progress)
            print("No data available. Run Solve TSP first.")
            return
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.distance_progress)), self.distance_progress, marker='o', linestyle='-')
        plt.xlabel("Iterations")
        plt.ylabel("Best Distance Found")
        plt.title("Distance Progression Over Iterations")
        plt.grid()
        plt.show()
        
    
if __name__ == "__main__":
    root = tk.Tk()
    app = TSPInterface(root)
    root.mainloop()


