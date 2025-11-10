import heapq


def astar(grid, start, goal):
    h = len(grid)
    w = len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        ci, cj = current
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] == 0:
                tentative = g_score[current] + 1
                if (ni, nj) not in g_score or tentative < g_score[(ni, nj)]:
                    g_score[(ni, nj)] = tentative
                    f = tentative + heuristic((ni, nj), goal)
                    heapq.heappush(open_set, (f, (ni, nj)))
                    came_from[(ni, nj)] = current
    return []


def world_to_grid(x, y, x_min, y_min, res):
    i = int((y - y_min) / res)
    j = int((x - x_min) / res)
    return i, j


def grid_to_world(i, j, x_min, y_min, res):
    x = j * res + x_min + res * 0.5
    y = i * res + y_min + res * 0.5
    return x, y
