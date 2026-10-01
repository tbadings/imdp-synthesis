"""State values on a tiled grid, so that the DP reads a successor cell's value by its grid position.

The dynamic program needs the values of all grid cells in a successor box (it takes a nondeterministic
minimum over them). Instead of mapping every cell to a state ID first, the values are kept on the grid
itself during the DP: the bounding box of the partition is cut into tiles of up to TILE cells per
dimension, and only the tiles that contain states are stored, after one all-zero tile that every cell
outside the partition reads (the absorbing state, value 0). A directory over the tiles of the bounding box
gives each stored tile's position. Reading a cell thus costs one directory read (the directory is small,
and the cells of a box share few tiles) and one read of the values, where neighbouring cells are close
in memory. Memory scales with the number of states, not with the size of the grid.

A cell's position is separable across dimensions, so the positions of all cells of a box are outer sums
of per-dimension vectors (no meshgrid).
"""

import logging

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# Tile edge (in grid cells) per dimension
TILE = 4


def _c_strides(shape):
    '''Strides (in elements) of a C-ordered array of the given shape.'''
    return np.concatenate([np.cumprod(np.asarray(shape[1:])[::-1])[::-1], [1]]).astype(np.int64)


class TiledGrid:
    """
    Positions of grid cells in the tiled value array of a partition.

    Attributes:
        size (int): Length of the tiled value array
        state_pos (np.ndarray): Position of every partition state's value, shape [num_states], int32
        directory (jnp.ndarray): Position (in tiles) of every tile of the bounding box, 0 for empty tiles;
            passed to `positions` as an argument, so it is not embedded into compiled functions
    """

    def __init__(self, partition, wrap):
        """
        :param partition: Partition (provides region_idx_inv, the grid index of every state, and number_per_dim)
        :param wrap: Per-dimension flags: grid indices of wrapped dimensions are taken modulo number_per_dim
        """
        coords = np.asarray(partition.region_idx_inv, dtype=np.int64)
        num_per_dim = np.asarray(partition.number_per_dim, dtype=np.int64)
        wrap = np.asarray(wrap, dtype=bool)

        # Bounding box of the states; wrapped dimensions span the whole grid (indices are folded into it)
        lo = np.where(wrap, 0, coords.min(axis=0))
        ext = np.where(wrap, num_per_dim, coords.max(axis=0) - lo + 1)
        tile = np.minimum(TILE, ext)
        num_tiles = -(-ext // tile)
        assert np.prod(num_tiles) < 2 ** 31, f'Tile directory of {np.prod(num_tiles)} entries exceeds int32 keys'

        # Each tile holding states gets a slot (from 1; slot 0 is the all-zero tile)
        rel = coords - lo
        used, slot = np.unique((rel // tile) @ _c_strides(num_tiles), return_inverse=True)
        directory = np.zeros(int(np.prod(num_tiles)), dtype=np.int32)
        directory[used] = np.arange(1, len(used) + 1, dtype=np.int32)

        self.tile_cells = int(np.prod(tile))
        self.size = (len(used) + 1) * self.tile_cells
        assert self.size < 2 ** 31, f'Tiled value array of {self.size} entries exceeds int32 positions'
        self.state_pos = ((slot.reshape(-1) + 1) * self.tile_cells + (rel % tile) @ _c_strides(tile)).astype(np.int32)
        self.directory = jnp.asarray(directory)

        # Static (Python) data of the position arithmetic
        self.wrap = tuple(bool(w) for w in wrap)
        self.num_per_dim = tuple(int(n) for n in num_per_dim)
        self.lo = tuple(int(x) for x in lo)
        self.ext = tuple(int(x) for x in ext)
        self.tile = tuple(int(x) for x in tile)
        self.tile_strides = tuple(int(x) for x in _c_strides(num_tiles))
        self.inner_strides = tuple(int(x) for x in _c_strides(tile))

        logger.info(f'- Values on a tiled grid: {len(used)} tiles of {self.tile} cells '
                    f'({len(coords) / (len(used) * self.tile_cells):.0%} filled), '
                    f'{4 * (self.size + directory.size) / 2 ** 20:.0f} MB')

    def positions(self, cols, directory):
        """
        Positions in the tiled value array of all cells of a box.

        :param cols: Per dimension the box's grid indices (list of D int arrays, unclipped: out-of-grid
            indices are allowed)
        :param directory: The `directory` attribute
        :return: int32 array of shape [len(cols[0]), ..., len(cols[D-1])]; position 0 (value 0) for cells
            outside the grid or outside the partition's bounding box
        """
        D = len(cols)
        tile_key, inner, inside = 0, 0, True
        for d, col in enumerate(cols):
            if self.wrap[d]:
                col = col % self.num_per_dim[d]
            rel = col - self.lo[d]
            ok = (rel >= 0) & (rel < self.ext[d])
            rel = jnp.clip(rel, 0, self.ext[d] - 1)

            shape = [1] * D
            shape[d] = -1
            tile_key = tile_key + (rel // self.tile[d] * self.tile_strides[d]).reshape(shape)
            inner = inner + (rel % self.tile[d] * self.inner_strides[d]).reshape(shape)
            inside = inside & ok.reshape(shape)
        return jnp.where(inside, directory[tile_key] * self.tile_cells + inner, 0)
