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
        cube[3 * (val // 3):3 * (val // 3 + 1), 3 * (i // 3):3 * (i // 3 + 1), j] = 0  # square
        cube[val, i, j] = 1
    update_grid()


def erase_from_col():
    idim, jdim = np.where(np.sum(cube, 1) == 1)
    for i, j in zip(idim, jdim):
        val = np.argmax(cube[i, :, j])
        cube[:, val, j] = 0  # row
        cube[i, val, :] = 0  # column
        cube[3 * (i // 3):3 * (i // 3 + 1), 3 * (val // 3):3 * (val // 3 + 1), j] = 0  # square
        cube[i, val, j] = 1
    update_grid()


def squares_sums(array):
    sums = np.zeros((3, 3, 9))
    for i in range(3):
        for j in range(3):
            sums[i, j] = np.sum(array[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)], (0, 1))
    return sums


def erase_from_square():
    idim, jdim, kdim = np.where(squares_sums(cube) == 1)
    for i, j, k in zip(idim, jdim, kdim):
        ival, jval = np.unravel_index(cube[3 * i:3 * (i + 1), 3 * j:3 * (j + 1), k].argmax(), (3,3))
        ival += 3 * i
        jval += 3 * j
        cube[:, jval, k] = 0  # row
        cube[ival, :, k] = 0  # column
        cube[ival, jval, k] = 1
    update_grid()


grid = grids.expert_grid
cube = np.ones((9, 9, 9))
rows, cols = np.where(grid > 0)
cube[rows, cols] = 0
cube[rows, cols, grid[rows, cols] - 1] = 1

print(grid)
for i in range(100):
    erase_from_val()
    erase_from_row()
    erase_from_col()
    erase_from_square()
    print(f'Step {i}')
    print(grid)


