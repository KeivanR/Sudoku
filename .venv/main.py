import numpy as np
import grids


def update_grid():
    idim, jdim = np.where(np.sum(cube, 2) == 1)
    grid[idim, jdim] = np.argmax(cube[idim, jdim], 1) + 1


def erase_from_val():
    idim, jdim = np.where(np.sum(cube, 2) == 1)
    for i, j in zip(idim, jdim):
        val = np.argmax(cube[i, j])
        cube[:, j, val] = 0  # row
        cube[i, :, val] = 0  # column
        cube[3 * (i // 3):3 * (i // 3 + 1), 3 * (j // 3):3 * (j // 3 + 1), val] = 0  # square
        cube[i, j, val] = 1
    update_grid()

def erase_from_row():
    idim, jdim = np.where(np.sum(cube, 0) == 1)
    for i, j in zip(idim, jdim):
        val = np.argmax(cube[:, i, j])
        cube[val, :, j] = 0  # row
        cube[val, i, :] = 0  # column
        cube[val, 3 * (i // 3):3 * (i // 3 + 1), 3 * (j // 3):3 * (j // 3 + 1)] = 0  # square
        cube[val, i, j] = 1
    update_grid()

def erase_from_col():
    idim, jdim = np.where(np.sum(cube, 1) == 1)
    for i, j in zip(idim, jdim):
        val = np.argmax(cube[i,:, j])
        cube[:, val, j] = 0  # row
        cube[i, val, :] = 0  # column
        cube[3 * (i // 3):3 * (i // 3 + 1), val, 3 * (j // 3):3 * (j // 3 + 1)] = 0  # square
        cube[i, val, j] = 1
    update_grid()

def erase_from_square():
    idim, jdim = np.where(np.sum(cube, 2) == 1)
    for i, j in zip(idim, jdim):
        val = np.argmax(cube[i, j])
        cube[:, j, val] = 0  # row
        cube[i, :, val] = 0  # column
        cube[3 * (i // 3):3 * (i // 3 + 1), 3 * (j // 3):3 * (j // 3 + 1), val] = 0  # square
        cube[i, j, val] = 1
    update_grid()


grid = grids.hard_grid
cube = np.ones((9, 9, 9))
rows, cols = np.where(grid > 0)
cube[rows, cols] = 0
cube[rows, cols, grid[rows, cols] - 1] = 1

print(grid)
for i in range(30):
    erase_from_val()
    #erase_from_row()
    erase_from_col()
    print(grid)
