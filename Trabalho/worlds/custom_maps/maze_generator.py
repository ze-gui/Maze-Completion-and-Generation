import random
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import csv

def add_dead_end_branches(mz, length):
    H = (mz.shape[0] - 1) // 2
    W = (mz.shape[1] - 1) // 2
    dead_ends = []

    # encontrar becos
    for y in range(H):
        for x in range(W):
            if mz[2*y+1, 2*x+1] == 1:
                nbrs = 0
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    if mz[2*y+1+dy, 2*x+1+dx] == 1:
                        nbrs += 1
                if nbrs == 1:
                    dead_ends.append((x, y))

    # número de becos a ramificar
    num_to_branch = int(len(dead_ends) * (length / 10.0))
    for x, y in random.sample(dead_ends, num_to_branch):
        walls = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            wy, wx = 2*y+1+dy, 2*x+1+dx
            if 0 <= wy < mz.shape[0] and 0 <= wx < mz.shape[1] and mz[wy, wx] == 0:
                walls.append((wy, wx))
        if walls:
            wy, wx = random.choice(walls)
            mz[wy, wx] = 1


def make_maze(cw, ch, dead_end_diff=5):
    H, W = ch, cw
    mz = np.zeros((2 * H + 1, 2 * W + 1), dtype=np.uint8)

    # criar as celulas centrais
    for y in range(H):
        for x in range(W):
            mz[2*y+1, 2*x+1] = 1

    # construir as paredes
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    visited = set()
    def carve(x, y):
        visited.add((x, y))
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                mz[2*y+1+dy, 2*x+1+dx] = 1
                carve(nx, ny)
    carve(0, 0)

    # modificar paredes para modificiar becos
    add_dead_end_branches(mz, dead_end_diff)

    # criar entrada e saida
    mz[-1, 1] = 1

    return mz


if __name__ == "__main__":
    diff      = int(input("Enter maze size (cells per side): "))
    png_size  = int(input("Enter PNG size  (px): "))
    dead_end  = int(input("Enter dead‐end length (1–10): "))

    # contruir o labirinto
    maze_low = make_maze(diff, diff, dead_end)

    # determinar tamanho das paredes e blocos de celulas para preencher o PNG exatamente
    tot_walls_x = diff + 1
    tot_walls_y = diff + 1
    base_cw = (png_size  - tot_walls_x) // diff
    rem_x   = (png_size  - tot_walls_x) %  diff
    base_ch = (png_size - tot_walls_y) // diff
    rem_y   = (png_size - tot_walls_y) %  diff

    # construir as larguras das colunas
    cols = []
    for i in range(2*diff + 1):
        if i % 2 == 0:
            cols.append(1)
        else:
            idx = (i - 1)//2
            cols.append(base_cw + (1 if idx < rem_x else 0))

    # construir as alturas das linhas
    rows = []
    for i in range(2*diff + 1):
        if i % 2 == 0:
            rows.append(1)
        else:
            idx = (i - 1)//2
            rows.append(base_ch + (1 if idx < rem_y else 0))

    # criar a imagem do labirinto
    img = np.repeat(np.repeat(maze_low, rows, axis=0), cols, axis=1)
    entrance_width = cols[1]
    W_actual = img.shape[1]
    start = (W_actual - entrance_width)//2
    img[0, start:start + entrance_width] = 1

    # guardar a imagem como PNG
    png_name = f"maze_{diff}_{dead_end}.png"
    plt.imsave(png_name, img, cmap="gray", vmin=0, vmax=1)

    # fazer o YAML de configuração
    yaml_name = f"maze_{diff}_{dead_end}_config.yaml"
    with open(yaml_name, "w") as f:
        f.write(f"image: {png_name}\n")
        f.write("resolution: 0.005\n")
        f.write("origin: [-0.200000, -0.200000, 0.000000]\n")
        f.write("occupied_thresh: 0.65\n")
    full_png  = os.path.abspath(png_name)
    full_yaml = os.path.abspath(yaml_name)
    h, w = img.shape
    print(f"Maze PNG saved to:    {full_png} ({w}×{h} px)")
    print(f"Config YAML saved to: {full_yaml}")

    # criação do arquivo de coordenadas e do Webots
    map_name: str = f'maze_{diff}_{dead_end}'

    with open(yaml_name, 'r') as stream:
        yaml_data = yaml.safe_load(stream)
    image_filename: str = yaml_data['image']
    resolution: float = yaml_data['resolution']
    origin: [float, float, float] = yaml_data['origin']
    occupied_thresh: float = yaml_data['occupied_thresh']
    max_pixel_value_for_wall: int = int(255 * occupied_thresh)
    wall_pixels_coords: [(int, int)] = []

    img: Image = Image.open(image_filename).convert('L')

    np_img: np.array = np.array(img)
    height: int = len(np_img)
    width: int = len(np_img[0])
    for row in range(len(np_img)):
        for col in range(len(np_img[row])):
            if np_img[row][col] <= max_pixel_value_for_wall:
                wall_pixels_coords.append((origin[0] + resolution * col, origin[1] + resolution * (height - row)))

    print('num walls = ', len(wall_pixels_coords))

    f = open(map_name + '_points.csv', 'w', newline='')
    writer = csv.writer(f)
    for x in range(width):
        writer.writerow((origin[0] + resolution * x, origin[1]))
        writer.writerow((origin[0] + resolution * x, origin[1] + resolution * (height - 1)))
    for y in range(1, width - 1):
        writer.writerow((origin[0], origin[1] + resolution * y))
        writer.writerow((origin[0] + resolution * (width - 1), origin[1] + resolution * y))
    for coord in wall_pixels_coords:
        writer.writerow(coord)
    f.close()

    # salvar o mapa de Webots
    base_map_webots_filepath: str = 'base_map.wbt'
    f = open(base_map_webots_filepath, 'r')
    webots_str: str = f.read()
    f.close()

    map_webots_filepath: str = map_name + '.wbt'
    f = open(map_webots_filepath, 'w')
    f.write(webots_str)

    f.write('RectangleArena {')
    f.write('  translation ' + str(origin[0] + resolution * width / 2) + ' ' + str(
        origin[1] + resolution * height / 2) + ' 0.0')
    f.write('  floorSize ' + str(resolution * width) + ' ' + str(resolution * height))
    f.write('  floorTileSize 0.25 0.25')
    f.write('  floorAppearance Parquetry {')
    f.write('    type "light strip"')
    f.write('  }')
    f.write('  wallHeight 0.05')
    f.write('}')

    index: int = 0
    for coord in wall_pixels_coords:
        f.write('Solid {')
        f.write('    translation ' + str(coord[0]) + ' ' + str(coord[1]) + ' 0.025')
        f.write('    children [')
        f.write('        Shape {')
        f.write('            geometry Box {')
        f.write('                size ' + str(resolution) + ' ' + str(resolution) + ' 0.05')
        f.write('            }')
        f.write('        }')
        f.write('    ]')
        f.write('    name "solid' + str(index) + '"')
        f.write('}')
        index += 1
    f.close()
