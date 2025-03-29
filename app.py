import pygame
import numpy as np
import random
import time
import json
import multiprocessing as mp
from numba import jit, prange

# Constantes
MAX_RADIUS = 200
MIN_CLUSTER_SIZE = 50
PREDEFINED_COLORS = {
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'orange': (255, 165, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'lavender': (230, 230, 250),
    'teal': (0, 128, 128),
    'white': (255, 255, 255),
    'gray': (128, 128, 128)
}

@jit(nopython=True, parallel=True)
def compute_forces(atoms, rules_array, radii2_array, num_colors, time_scale, viscosity, wall_repel, width, height, camera_x, camera_y):
    """Computar todas las fuerzas usando Numba para acelerar el cálculo"""
    n = len(atoms)
    fx_array = np.zeros(n)
    fy_array = np.zeros(n)
    
    # Calcular fuerzas
    for i in prange(n):
        a = atoms[i] # Atomo actual
        fx = 0.0 # Fuerza en x
        fy = 0.0 # Fuerza en y
        idx = int(a[4] * num_colors) # Índice de color
        r2 = radii2_array[int(a[4])] # Radio al cuadrado
        
        # Coordenadas del átomo
        ax = a[0]
        ay = a[1]
        
        for j in range(n):
            if i == j:
                continue
                
            b = atoms[j]
            g = rules_array[idx + int(b[4])]
            
            # Calcular todas las posibles interacciones (mundo toroidal)
            for dx_offset in [-width, 0, width]:
                for dy_offset in [-height, 0, height]:
                    # Posición de b considerando desplazamiento toroidal
                    bx = b[0] + dx_offset
                    by = b[1] + dy_offset
                    
                    dx = ax - bx
                    dy = ay - by
                    
                    if dx != 0 or dy != 0:
                        d = dx * dx + dy * dy
                        if d < r2:
                            F = g / np.sqrt(d)
                            fx += F * dx
                            fy += F * dy
        
        fx_array[i] = fx
        fy_array[i] = fy
    
    # Actualizar velocidades y posiciones
    total_v = 0.0
    for i in range(n):
        a = atoms[i]
        
        # Actualizar velocidad con viscosidad
        vmix = (1.0 - viscosity)
        a[2] = a[2] * vmix + fx_array[i] * time_scale * viscosity
        a[3] = a[3] * vmix + fy_array[i] * time_scale * viscosity
        
        # Registrar actividad típica
        total_v += abs(a[2])
        total_v += abs(a[3])
        
        # Actualizar posiciones
        a[0] += a[2]
        a[1] += a[3]
        
        # Teletransportar partículas (efecto toroidal/infinito)
        if a[0] < 0:
            a[0] += width
        elif a[0] >= width:
            a[0] -= width
        
        if a[1] < 0:
            a[1] += height
        elif a[1] >= height:
            a[1] -= height
    
    return total_v, atoms


class ParticleLife:
    def __init__(self):
        # Inicializar pygame
        pygame.init()
        self.width = int(1080*0.7)
        self.height = int(1920*0.7)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Particle Life - Mundo Infinito")
        self.font = pygame.font.SysFont("Arial", 18)
        
        # Configuración
        self.settings = {
            "seed": 91651088029,
            "fps": 0,
            "atoms": {
                "count": 1000,  # Por color
                "radius": 1,
            },
            "drawings": {
                "background_color": (0, 0, 0),
            },
            "num_colors": 4,
            "time_scale": 0.5,
            "viscosity": 0.7,
            "wall_repel": 0,
            "explore": False,
            "explore_period": 100,
        }
        
        # Configuración de cámara
        self.camera_x = self.width / 2
        self.camera_y = self.height / 2
        self.camera_speed = 5
        self.camera_moving = False
        self.zoom_level = 1.0  # Nivel de zoom
        self.zoom_speed = 0.05  # Velocidad de zoom
        
        # Guardar partículas originales
        self.original_atoms = set()
        
        # Crear colores, reglas y átomos
        self.colors = []
        self.rules = {}
        self.rules_array = np.array([])
        self.radii = {}
        self.radii2_array = np.array([])
        self.atoms = []
        
        # Establecer el número de colores
        self.set_number_of_colors()
        
        # Inicializar las reglas y los átomos
        self.random_rules()
        self.random_atoms(self.settings["atoms"]["count"], True)
        
        # Variables para exploración
        self.exploration_timer = 0
        
        # Para cálculo de FPS
        self.last_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.total_v = 0.0
        
        # Bandera para salir
        self.running = True
        
        self.menu_visible = True  # Nuevo atributo para controlar la visibilidad del menú
    
    def set_number_of_colors(self):
        self.colors = []
        color_names = list(PREDEFINED_COLORS.keys())
        # Excluir white y gray que se usarán para las partículas duplicadas
        color_names = [c for c in color_names if c not in ['white', 'gray']]
        for i in range(min(self.settings["num_colors"], len(color_names))):
            self.colors.append(color_names[i])
    
    def random_rules(self):
        # Generar una nueva semilla aleatoria cada vez
        if not self.settings["explore"]:  # Si no estamos en modo exploración
            self.settings["seed"] = int(time.time() * 1000000) % (2**32 - 1)
        
        seed = self.settings["seed"]
        if not np.isfinite(seed):
            self.settings["seed"] = 0xcafecafe
        
        print(f"Seed={self.settings['seed']}")
        numpy_seed = int(self.settings["seed"] % (2**32 - 1))
        random.seed(self.settings["seed"])
        np.random.seed(numpy_seed)
        
        self.rules = {}
        self.radii = {}
        
        for i in self.colors:
            self.rules[i] = {}
            for j in self.colors:
                self.rules[i][j] = np.random.random() * 2 - 1
            self.radii[i] = 80
        
        print(json.dumps(self.rules))
        self.flatten_rules()
    
    def symmetric_rules(self):
        for i in self.colors:
            for j in self.colors:
                if j < i:
                    v = 0.5 * (self.rules[i][j] + self.rules[j][i])
                    self.rules[i][j] = self.rules[j][i] = v
        
        print(json.dumps(self.rules))
        self.flatten_rules()
    
    def flatten_rules(self):
        # Convertir reglas a formato NumPy para mayor velocidad
        rules_list = []
        radii2_list = []
        
        for c1 in self.colors:
            for c2 in self.colors:
                rules_list.append(self.rules[c1][c2])
            radii2_list.append(self.radii[c1] * self.radii[c1] * 2)
        
        self.rules_array = np.array(rules_list, dtype=np.float32)
        self.radii2_array = np.array(radii2_list, dtype=np.float32)
    
    def random_x(self):
        return np.random.random() * (self.width - 100) + 50
    
    def random_y(self):
        return np.random.random() * (self.height - 100) + 50
    
    def create_atom(self, number, color_idx):
        for i in range(number):
            atom_id = len(self.atoms)
            self.atoms.append([self.random_x(), self.random_y(), 0, 0, color_idx])
            # Guardar IDs de átomos originales
            self.original_atoms.add(atom_id)
    
    def random_atoms(self, number_of_atoms_per_color, clear_previous):
        if clear_previous:
            self.atoms = []
            self.original_atoms = set()
        
        for c in range(len(self.colors)):
            self.create_atom(number_of_atoms_per_color, c)
    
    def start_random(self):
        self.random_rules()
        self.random_atoms(self.settings["atoms"]["count"], True)

    def apply_rules(self):
        # Usar la función optimizada con Numba
        self.total_v, updated_atoms = compute_forces(
            np.array(self.atoms, dtype=np.float32),
            self.rules_array,
            self.radii2_array,
            self.settings["num_colors"],
            self.settings["time_scale"],
            self.settings["viscosity"],
            self.settings["wall_repel"],
            self.width,
            self.height,
            self.camera_x,
            self.camera_y
        )
        
        # Actualizar átomos
        self.atoms = updated_atoms.tolist()
        self.total_v /= len(self.atoms)
    
    def explore_parameters(self):
        """Explorar parámetros aleatorios"""
        if self.exploration_timer <= 0:
            c1 = self.colors[int(np.random.random() * self.settings["num_colors"])]
            
            if np.random.random() >= 0.2:  # 80% de las veces, cambiamos la fuerza
                c2 = self.colors[int(np.random.random() * self.settings["num_colors"])]
                new_strength = np.random.random() * 2 - 1  # Entre -1 y 1
                self.rules[c1][c2] = new_strength
            else:  # de lo contrario, el radio
                self.radii[c1] = 1 + int(np.random.random() * MAX_RADIUS)
            
            self.flatten_rules()
            self.exploration_timer = self.settings["explore_period"]
        
        self.exploration_timer -= 1
    
    def world_to_screen(self, x, y):
        """Convierte coordenadas del mundo a coordenadas de pantalla, considerando zoom y cámara"""
        screen_x = (x - self.camera_x) * self.zoom_level + self.width / 2
        screen_y = (y - self.camera_y) * self.zoom_level + self.height / 2
        return screen_x, screen_y
    
    def update_params(self):
        """Actualizar parámetros como FPS y time_scale"""
        # Registrar FPS una vez por segundo
        self.frame_count += 1
        cur_time = time.time()
        if cur_time - self.last_fps_update >= 1.0:
            self.settings["fps"] = self.frame_count
            self.frame_count = 0
            self.last_fps_update = cur_time
        
        if self.settings["explore"]:
            self.explore_parameters()
            
        # Manejar zoom de cámara
        keys = pygame.key.get_pressed()
        
        # Limitar el zoom a un rango razonable
        if keys[pygame.K_z] and self.zoom_level < 5.0:  # Zoom in, máximo 5x
            self.zoom_level += self.zoom_speed
        if keys[pygame.K_x] and self.zoom_level > 0.2:  # Zoom out, mínimo 0.2x
            self.zoom_level -= self.zoom_speed
        
        # Ajustar la velocidad de la cámara según el nivel de zoom
        camera_speed_adjusted = self.camera_speed / self.zoom_level
        
        # Manejar movimiento de cámara
        if keys[pygame.K_LEFT]:
            self.camera_x -= camera_speed_adjusted
            self.camera_moving = True
        if keys[pygame.K_RIGHT]:
            self.camera_x += camera_speed_adjusted
            self.camera_moving = True
        if keys[pygame.K_UP]:
            self.camera_y -= camera_speed_adjusted
            self.camera_moving = True
        if keys[pygame.K_DOWN]:
            self.camera_y += camera_speed_adjusted
            self.camera_moving = True
    
    def toggle_setting(self, setting_name):
        """Función auxiliar para alternar configuraciones booleanas"""
        if setting_name in ["explore"]:
            self.settings[setting_name] = not self.settings[setting_name]
    
    def handle_events(self):
        """Manejar eventos de teclado"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Manejar teclas
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_h:  # Nueva opción para ocultar/mostrar el menú
                    self.menu_visible = not self.menu_visible
                elif event.key == pygame.K_r:
                    self.random_rules()
                    self.random_atoms(self.settings["atoms"]["count"], True)
                elif event.key == pygame.K_o:
                    self.random_atoms(self.settings["atoms"]["count"], True)
                elif event.key == pygame.K_s:
                    self.symmetric_rules()
                    self.random_atoms(self.settings["atoms"]["count"], True)
                elif event.key == pygame.K_e:
                    self.toggle_setting("explore")
                elif event.key == pygame.K_u:
                    self.settings["num_colors"] = min(5, self.settings["num_colors"] + 1)  # Limitado a 5 colores para dejar white y gray
                    self.set_number_of_colors()
                    self.start_random()
                elif event.key == pygame.K_d:
                    self.settings["num_colors"] = max(1, self.settings["num_colors"] - 1)
                    self.set_number_of_colors()
                    self.start_random()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.settings["time_scale"] *= 1.1
                    print(f"Time Scale: {self.settings['time_scale']:.2f}")
                elif event.key == pygame.K_MINUS:
                    self.settings["time_scale"] /= 1.1
                elif event.key == pygame.K_m:  # Nueva opción para editar la matriz
                    self.edit_rules_matrix()
                elif event.key == pygame.K_c:  # Resetear posición de cámara y zoom
                    self.camera_x = self.width / 2
                    self.camera_y = self.height / 2
                    self.zoom_level = 1.0
    
    def edit_rules_matrix(self):
        """Editar la matriz de reglas manualmente"""
        print("Ingrese los nuevos valores para la matriz de reglas (formato: color1 color2 valor):")
        for i in self.colors:
            for j in self.colors:
                value = input(f"Valor para {i} a {j}: ")
                try:
                    self.rules[i][j] = float(value)
                except ValueError:
                    print("Valor no válido, se mantendrá el valor anterior.")
        
        self.flatten_rules()  # Asegurarse de que la matriz se actualice
        print("Matriz de reglas actualizada.")
    
    def draw_info(self):
        """Mostrar información en pantalla"""
        if self.menu_visible:  # Solo dibujar si el menú es visible
            # Mostrar información de FPS y configuraciones
            info_texts = [
                f"FPS: {self.settings['fps']}",
                f"Seed: {self.settings['seed']}",
                f"Time Scale: {self.settings['time_scale']:.2f}",
                f"Atoms: {len(self.atoms)}",
                f"Colors: {self.settings['num_colors']}",
                f"Camera: ({self.camera_x:.2f}, {self.camera_y:.2f})",
                f"Zoom: {self.zoom_level:.2f}x"
            ]
            
            y_pos = 10
            for text in info_texts:
                text_surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, y_pos))
                y_pos += 20
            
            # Mostrar controles
            controls = [
                "R: New Random Rules",
                "O: Reset Atoms",
                "S: Symmetric Rules",
                "E: Toggle Explore",
                "U/D: Change Colors",
                "+/-: Adjust Time Scale",
                "M: Edit Rules Matrix",
                "←/→/↑/↓: Move Camera",
                "Z/X: Zoom In/Out",
                "C: Reset Camera & Zoom"
            ]
            
            y_pos = 160
            for control in controls:
                control_text = self.font.render(control, True, (200, 200, 200))
                self.screen.blit(control_text, (10, y_pos))
                y_pos += 20
    
    def run(self):
        """Bucle principal del programa"""
        clock = pygame.time.Clock()
        
        while self.running:
            # Manejar eventos
            self.handle_events()
            
            # Actualizar el fondo
            self.screen.fill(self.settings["drawings"]["background_color"])
            
            # Aplicar reglas
            self.apply_rules()
            
            # Dibujar átomos y sus réplicas en los bordes (efecto mundo infinito)
            for i, a in enumerate(self.atoms):
                # Determinar color: original o duplicado
                if i in self.original_atoms:
                    color_rgb = PREDEFINED_COLORS[self.colors[int(a[4])]]
                else:
                    color_rgb = PREDEFINED_COLORS['gray']
                
                # Calcular tamaño base para las partículas
                base_radius = self.settings["atoms"]["radius"]
                radius = max(1, int(base_radius))  # Asegurar que siempre es al menos 1 píxel
                
                # Calcular las coordenadas centrales de la partícula
                center_x, center_y = self.world_to_screen(a[0], a[1])
                
                # Dibujar partícula principal (solo si está en pantalla)
                if 0 <= center_x <= self.width and 0 <= center_y <= self.height:
                    pygame.draw.rect(self.screen, color_rgb, 
                                   (int(center_x - radius), int(center_y - radius), 
                                    2 * radius, 2 * radius))
                
                # Dibujar duplicados en las 8 direcciones para simular mundo infinito
                for dx in [-self.width, 0, self.width]:
                    for dy in [-self.height, 0, self.height]:
                        if dx == 0 and dy == 0:
                            continue  # Saltamos la posición principal que ya dibujamos
                        
                        # Color para las réplicas
                        if i in self.original_atoms:
                            replica_color = PREDEFINED_COLORS['white']
                        else:
                            replica_color = PREDEFINED_COLORS['gray']
                        
                        # Calcular posición de la réplica
                        replica_x, replica_y = self.world_to_screen(a[0] + dx, a[1] + dy)
                        
                        # Solo dibujar si está dentro de la pantalla
                        if (0 <= replica_x <= self.width and 0 <= replica_y <= self.height):
                            pygame.draw.rect(self.screen, replica_color, 
                                           (int(replica_x - radius), int(replica_y - radius), 
                                            2 * radius, 2 * radius))
            
            # Actualizar parámetros
            self.update_params()
            
            # Mostrar información
            self.draw_info()
            
            # Actualizar pantalla
            pygame.display.flip()
            
            # Limitar FPS
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    # Configurar Numba para usar todos los núcleos disponibles
    import os
    os.environ["NUMBA_NUM_THREADS"] = str(mp.cpu_count())
    
    # Iniciar simulación
    sim = ParticleLife()
    sim.run()