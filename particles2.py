import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from pathlib import Path

dt = 0.0005     
epsilon = 1
sigma = 1
k_b = 1

box_size = 10 * sigma 
d_min = 9 * sigma
#T = 0.45 * epsilon / k_b #0.2-0.7
#N = 100 

ani_speed = 10
energy_interval = 10
save_interval = 10 
term_times = 2000 
total_steps = 40000
pressure_interval = 50


def last_mean(arr, n):
    summ = 0
    if len(arr) > (n + 1):
        for i in range(n):
            summ += arr[len(arr) - 1 - n]
        return (summ / n)
    else:
        return np.mean(arr)

def main(temp, num):
    T = temp
    N = num
    n_rows = int(np.ceil(np.sqrt(N)))
    spacing = box_size / n_rows
    particles = []
    natural_list = []
    xcount = 0.0

    
    class Particle:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.vx = np.random.normal(0, np.sqrt(T))
            self.vy = np.random.normal(0, np.sqrt(T))
            self.ax = 0.0
            self.ay = 0.0
            self.oldax = 0.0
            self.olday = 0.0

        def is_x_border(self):
            count = 0
            if (self.x + self.vx * dt + self.ax * dt ** 2 / 2) // box_size != 0:
                count += np.abs(self.vx)
            if (self.y + self.vy * dt + self.ay * dt ** 2 / 2) // box_size != 0:
                count += np.abs(self.vy)
            return count

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
                d = ((p1.x - p2.x - dx) ** 2 + (p1.y - p2.y - dy) ** 2)            
                if d < d_min and d != 0:
                    force_mult = (24) * (2 * (1 / d) ** 3 - 1) * (1 / d) ** 4
                    p1.ax += force_mult * (p1.x - p2.x - dx)
                    p1.ay += force_mult * (p1.y - p2.y - dy)
                    
                    p2.ax -= force_mult * (p1.x - p2.x - dx)
                    p2.ay -= force_mult * (p1.y - p2.y - dy)

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
    
    
    def pressure_ver(ps):
        V = 100
        virial_term = 0.0
        for p in ps:
            virial_term += p.x * p.ax + p.y * p.ay
        virial_term /= (2 * V) 
        total_pressure = T * N / V + virial_term
    
        return total_pressure

    def kin_en(ps):
        return 0.5 * sum(p.vx**2 + p.vy**2 for p in ps)

    def termalize(ps, t):
        kinetic = sum(p.vx**2 + p.vy**2 for p in ps) / (2 * len(ps))
        scale = np.sqrt(t/kinetic)
        for p in ps:
            p.vx *= scale
            p.vy *= scale

    def interaction_wrapper(args):
        i, j, particles = args
        p1 = particles[i]
        p2 = particles[j]
        interaction(p1, p2)
        return (i, j, p1.ax, p1.ay, p2.ax, p2.ay)

    def compute_interactions_multiprocessing(particles):
        N = len(particles)
        args = [(i, j, particles) for i in range(N) for j in range(i+1, N)]
        
        with Pool(8) as pool:
            results = pool.map(interaction_wrapper, args)
        
        # Reset accelerations
        for p in particles:
            p.ax = 0
            p.ay = 0
        
        # Apply results from multiprocessing
        for res in results:
            i, j, ax1, ay1, ax2, ay2 = res
            particles[i].ax += ax1
            particles[i].ay += ay1
            particles[j].ax += ax2
            particles[j].ay += ay2

    def compute_interactions(ps):
        for i in range(N):
            for j in range(i + 1, N):
                interaction(ps[i], ps[j])

    def compute_xborder(ps):
        total = 0;
        for p in ps:
            total += p.is_x_border()
        return total

    for i in range(N):
        x = (i % n_rows + 0.5) * spacing
        y = (i // n_rows + 0.5) * spacing
        p = Particle()
        p.x = x
        p.y = y
        particles.append(p)

    avg_vx = sum(p.vx for p in particles) / N
    avg_vy = sum(p.vy for p in particles) / N
    for p in particles:
        p.vx -= avg_vx
        p.vy -= avg_vy

    all_positions = []
    en_list = []
    en_term = []
    kin_en_list = []
    times = 0
    term = True
    pressure_ver_list = []

    en_term.append(energy(particles))

    for step in range(total_steps):
    
        for p in particles:
            p.save_a()

        compute_interactions(particles)
#        compute_interactions_multiprocessing(particles)
        xcount += compute_xborder(particles)

        for p in particles:
            p.accelerate()
            p.move()

        if step % save_interval == 0:
            all_positions.append(np.array([[p.x, p.y] for p in particles]))

        if step % 100 == 0 and step != 0:
            print(np.abs(last_mean(en_list, 50) / last_mean(en_list, 5) - 1))
            print('Пройдено ', step, ' шагов')

        if step % pressure_interval == 0 and step != 0:
            pressure_ver_list.append(pressure_ver(particles))

        if step % energy_interval == 0:
            en_list.append(energy(particles))
            kin_en_list.append(kin_en(particles))
#            print('Энергия:', en_list[len(en_list) - 1])

        if (np.abs(last_mean(en_list, 50) / last_mean(en_list, 5) - 1) > 0.1):
            term = False

        if term or times < term_times:
            termalize(particles, T)
#            print('термализация!')
            times += 1
            if times % 100 == 0:
                print(times, 'термализация')

#fig, ax = plt.subplots(figsize=(8, 8))
#ax.set_xlim(0, box_size)
#ax.set_ylim(0, box_size)
#ax.grid(True, linestyle='--', alpha=0.5)  # Добавим сетку для наглядности
#scatter = ax.scatter([], [], s=20, c='red', edgecolor='black')  # Яркие крупные точки
#title = ax.set_title('Движение частиц')

#for i in range(len(en_list)):
#    natural_list.append(i)

#fig1, ax1 = plt.subplots()
#scatter1 = ax1.scatter(natural_list, en_list)
#scatter2 = ax1.scatter(natural_list, kin_en_list, c = 'red')
    pressure = xcount / (4 * dt * total_steps)
    pressure_file = Path(f"/home/kutuka/Documents/part2/N={N}|T={T}/pressure.txt")
    pressure_file.parent.mkdir(exist_ok=True, parents=True)

    ver_pressure_file = Path(f"/home/kutuka/Documents/part2/N={N}|T={T}/ver_pressure.txt")
    ver_pressure_file.parent.mkdir(exist_ok=True, parents=True)   
    
    energy_file = Path(f"/home/kutuka/Documents/part2/N={N}|T={T}/energy.txt")
    energy_file.parent.mkdir(exist_ok=True, parents=True)

    pressure_file.write_text(str(pressure))

    str_ver_pressure = "\n".join(map(str, pressure_ver_list))
    ver_pressure_file.write_text(str_ver_pressure)

    str_energy = "\n".join(map(str, en_list))
    energy_file.write_text(str_energy)

#    print('Pressure:',xcount / (4 * dt * total_steps))

    #def init():
     #   scatter.set_offsets(np.empty((0, 2)))
      #  return scatter,

#def update(frame):
 #   scatter.set_offsets(all_positions[frame])
  #  return scatter,

#ani = FuncAnimation(
 #   fig,
  #  update,
   # frames=len(all_positions),
#    init_func=init,
 #   interval=ani_speed,  
  #  blit=True,
   # repeat=True
#)

#plt.gcf()._ani = ani

#plt.tight_layout()
#plt.show()
n = int(input())
for t in range (1, 21):
    main(0.2 + t * 0.005 + n * 0.1, 40)



