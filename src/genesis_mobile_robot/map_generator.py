import random


def build_empty_grid(x_min, x_max, y_min, y_max, res):
    nx = int((x_max - x_min) / res)
    ny = int((y_max - y_min) / res)
    grid = [[0 for _ in range(nx)] for _ in range(ny)]
    return grid


def add_random_blocks(
    grid,
    x_min,
    y_min,
    res,
    num_blocks=4,
    x_max=5.0,
    y_max=5.0,
    block_w_range=(0.5, 1.5),
    block_h_range=(0.5, 1.5),
):
    """
    Coloca alguns retângulos sólidos no grid.
    """
    h = len(grid)
    w = len(grid[0])

    def world_rect_to_cells(x0, y0, x1, y1):
        cells = []
        for i in range(h):
            for j in range(w):
                cx = j * res + x_min + res * 0.5
                cy = i * res + y_min + res * 0.5
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    cells.append((i, j))
        return cells

    for _ in range(num_blocks):
        bw = random.uniform(*block_w_range)
        bh = random.uniform(*block_h_range)

        cx = random.uniform(x_min + 1.0, x_max - 1.0)
        cy = random.uniform(y_min + 1.0, y_max - 1.0)
        x0 = cx - bw / 2
        x1 = cx + bw / 2
        y0 = cy - bh / 2
        y1 = cy + bh / 2
        cells = world_rect_to_cells(x0, y0, x1, y1)
        for i, j in cells:
            grid[i][j] = 1

    return grid
