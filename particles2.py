import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

dt = 0.0001 
box_size = 25.0
T = 100    
N = 50    
total_steps = 100000
save_interval = 10
epsilon = 1 
sigma = 1
d_min = 3 * sigma

class Particle:
    def __init__(self):
        self.x = np.random.rand() * box_size
        self.y = np.random.rand() * box_size
        self.vx = np.random.normal(0, np.sqrt(T))
        self.vy = np.random.normal(0, np.sqrt(T))
        self.ax = 0.0
        self.ay = 0.0
        self.oldax = 0.0
        self.olday = 0.0

    def move(self):
        self.x = (self.x + self.vx * dt + self.ax * dt ** 2 / 2) % box_size
        self.y = (self.y + self.vy * dt + self.ay * dt ** 2 / 2) % box_size

    def accelerate(self):
        self.vx += (self.ax + self.oldax) * dt / 2
        self.vy += (self.ay + self.olday) * dt / 2

    def save_a(self): 
        self.oldax = self.ax
        self.olday = self.ay
        self.ay = 0
        self.ax = 0

#def interaction(particle1, particle2):
#    d = np.sqrt((particle1.x - particle2.x) ** 2 + (particle1.y - particle2.y) ** 2)
#    if d < d_min and d != 0:
#        particle1.ax += (-24 * epsilon / sigma ** 2) * (2 * (sigma / d) ** 14 - (sigma / d) ** 8) * (particle1.x - particle2.x)
#        particle2.ax += (-24 * epsilon / sigma ** 2) * (2 * (sigma / d) ** 14 - (sigma / d) ** 8) * (particle2.x - particle1.x)
#        particle1.ay += (-24 * epsilon / sigma ** 2) * (2 * (sigma / d) ** 14 - (sigma / d) ** 8) * (particle1.y - particle2.y)
#        particle2.ay += (-24 * epsilon / sigma ** 2) * (2 * (sigma / d) ** 14 - (sigma / d) ** 8) * (particle2.y - particle1.y)

def interaction(p1, p2):
    for dx in [-box_size, 0, box_size]:
        for dy in [-box_size, 0, box_size]:
            d = np.sqrt((p1.x - p2.x - dx) ** 2 + (p1.y - p2.y - dy) ** 2)            
            if d < d_min and d != 0:
                force_mult = (-24 * epsilon / sigma ** 2) * (2 * (sigma / d) ** 6 - 1) * (sigma / d) ** 8
                p1.ax += force_mult * (p1.x - p2.x - dx)
                p1.ay += force_mult * (p1.y - p2.y - dy)

def energy(ps):
    kinetic = 0.5 * sum(p.vx**2 + p.vy**2 for p in ps)
    potential = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            for dx in [-box_size, 0, box_size]:
                for dy in [-box_size, 0, box_size]:
                    d = np.sqrt((ps[i].x - ps[j].x + dx)**2 + (ps[i].y - ps[j].y + dy)**2)
                    potential += 4 * epsilon * ((sigma / d)**12 - (sigma / d)**6)
    return kinetic + potential

particles = [Particle() for _ in range(N)]

avg_vx = sum(p.vx for p in particles) / N
avg_vy = sum(p.vy for p in particles) / N
for p in particles:
    p.vx -= avg_vx
    p.vy -= avg_vy

all_positions = []
all_speeds = []
for step in range(total_steps):
    
    for p in particles:
        p.save_a()

    for i in range(N):
        for j in range(N):
            interaction(particles[i], particles[j])
        
    for p in particles:
        p.accelerate()
        p.move()

    if step % save_interval == 0:
        all_positions.append(np.array([[p.x, p.y] for p in particles]))


    if step % 1000 == 0:
        print('Пройдено ', step, ' шагов')
        print(energy(particles))

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.grid(True, linestyle='--', alpha=0.5)  # Добавим сетку для наглядности
scatter = ax.scatter([], [], s=20, c='red', edgecolor='black')  # Яркие крупные точки
title = ax.set_title('Движение частиц')

def init():
    scatter.set_offsets(np.empty((0, 2)))
    return scatter,

def update(frame):
    scatter.set_offsets(all_positions[frame])
    return scatter,

ani = FuncAnimation(
    fig,
    update,
    frames=len(all_positions),
    init_func=init,
    interval=100,  
    blit=True,
    repeat=True
)

plt.gcf()._ani = ani

plt.tight_layout()
plt.show()
