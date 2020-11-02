import numpy as np

__all__ = ['TilesManager']


class _MeshGenerator:
    class MGException(Exception):
        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, region: list, cell_size: list, overlap: list = None):
        region = np.array(region)
        if np.linalg.norm(region[1] - region[0]) == 0:
            raise self.MGException("Region size is zero!")

        if region[0][0] >= region[1][0] or region[0][1] >= region[1][1]:
            raise self.MGException("Bad region coordinates")

        if len(cell_size) < 2 or cell_size[0] <= 0 or cell_size[1] <= 0:
            raise self.MGException("Bad cell size")

        self.__region = region
        self.__cell_size = cell_size
        if overlap is not None:
            self.__overlap = 0.5 * np.array([overlap[0], overlap[1]])
        else:
            cells_cnt = np.array(np.abs(self.__region[1] - self.__region[0]) / self.__cell_size)
            if np.array_equal(cells_cnt, [0, 0]):
                self.__overlap = np.array([0, 0])
            else:
                self.__overlap = 2 * (np.ceil(cells_cnt) * cell_size - np.abs(self.__region[1] - self.__region[0])) / np.round(
                    cells_cnt)

    def generate_cells(self):
        result = []

        def walk_cells(callback: callable):
            y_start = self.__region[0][1]
            x_start = self.__region[0][0]

            y = y_start

            step_cnt = np.array(np.ceil(np.abs(self.__region[1] - self.__region[0]) / self.__cell_size), dtype=np.uint64)

            for i in range(step_cnt[1]):
                x = x_start

                for j in range(step_cnt[0]):
                    callback([np.array([x, y]),
                              np.array([x + self.__cell_size[0], y + self.__cell_size[1]])], i, j)
                    x += self.__cell_size[0]

                y += self.__cell_size[1]

        def on_cell(coords, i, j):
            offset = self.__overlap * np.array([j, i], dtype=np.float32)
            coords[0] = coords[0] - offset
            coords[1] = coords[1] - offset
            result.append(np.array(coords, dtype=np.uint32))

        walk_cells(on_cell)
        return result


class TilesManager:
    def __init__(self):
        self._tiles = []
        self._size = None

    def generate_tiles(self, size: [], tiles_size: [], overlap: list = None) -> 'TilesManager':
        self._size = size
        self._tiles = _MeshGenerator(size, tiles_size, overlap).generate_cells()
        return self

    def get_tiles(self) -> []:
        return self._tiles

    def cut_image_by_tiles(self, image: np.ndarray) -> []:
        res = []
        for tile in self._tiles:
            res.append(TilesManager._cut_image_by_tile(image, tile))
        return res

    def merge_images_by_tiles(self, images: [np.ndarray]) -> np.ndarray:
        res = np.array(self._size, dtype=images[0].dtype)
        for i, tile in enumerate(self._tiles):
            TilesManager._insert_tile_to_image(res, images[i], tile)
        return res

    @staticmethod
    def _cut_image_by_tile(image: np.ndarray, tile: []) -> np.ndarray:
        return image[tile[0][1]: tile[1][1], tile[0][0]: tile[1][0], :]

    @staticmethod
    def _insert_tile_to_image(image: np.ndarray, image_part: np.ndarray, tile: []) -> None:
        image[tile[0][1]: tile[1][1], tile[0][0]: tile[1][0], :] = image_part
