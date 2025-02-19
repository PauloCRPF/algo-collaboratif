import tkinter as tk
import numpy as np
import random
from ant_colony_tsp import Environment

class TSPInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Interface - Ajouter des points")
        
        self.canvas = tk.Canvas(self.root, width=1000, height=650, bg="white")
        self.canvas.pack()
        
        self.route_label = tk.Label(self.root, text="Best Route: ", font=("Arial", 12))
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
        
        self.solve_button = tk.Button(button_frame, text="Solve TSP", command=self.solve_tsp)
        self.solve_button.pack(side=tk.LEFT, padx=5)
        
        self.random_entry = tk.Entry(button_frame, width=5)
        self.random_entry.pack(side=tk.LEFT, padx=5)
        self.random_button = tk.Button(button_frame, text="Add Random", command=self.add_random_points)
        self.random_button.pack(side=tk.LEFT, padx=5)
    
    def generate_random_positions(self, num_cities, x_range=(10, 990), y_range=(10, 640)):
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
        
        env = Environment(num_cities=num_cities, alpha=1, beta=2, gamma=0.1, num_ants=100, positions={i: self.points[i] for i in range(num_cities)})
        best_path, best_distance = env.simulate(num_iterations=500)
        
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

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPInterface(root)
    root.mainloop()


