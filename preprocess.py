import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree

class Preprocess(object):

    def __init__(self):
        pass

    def simple_preprocess_normalize_all_columns(self, df):
        ans_df = pd.DataFrame()
        scenes = np.unique(df.scene_id.values)
        for scene_id in tqdm(scenes):  # начинаем ходить по каждой сцене
            df_scene_id = df[df.scene_id == scene_id]  # берем все данные из этой сцены
            eps = 0.001
            df_scene_id["x_normailized"] = (df_scene_id.x - df_scene_id.x.min()) / (
                        df_scene_id.x.max() - df_scene_id.x.min() + eps)
            df_scene_id["y_normailized"] = (df_scene_id.y - df_scene_id.y.min()) / (
                        df_scene_id.y.max() - df_scene_id.y.min() + eps)
            df_scene_id["z_normailized"] = (df_scene_id.z - df_scene_id.z.min()) / (
                        df_scene_id.z.max() - df_scene_id.z.min() + eps)
            df_scene_id["ring_normailized"] = (df_scene_id.ring - df_scene_id.ring.min()) / (
                    df_scene_id.ring.max() - df_scene_id.ring.min() + eps)
            df_scene_id["intensity_normailized"] = (df_scene_id.intensity - df_scene_id.intensity.min()) / (
                    df_scene_id.intensity.max() - df_scene_id.intensity.min() + eps)

            ans_df = ans_df.append(df_scene_id)
        return ans_df

    def kdtree_preprocess(self, df):
        ans_df = pd.DataFrame()
        scenes = np.unique(df.scene_id.values)
        for scene_id in tqdm(scenes):   # начинаем ходить по каждой сцене
            df_scene_id = df[df.scene_id == scene_id]  # берем все данные из этой сцены
            eps = 0.001
            df_scene_id["x_normailized"] = (df_scene_id.x - df_scene_id.x.min()) / (
                        df_scene_id.x.max() - df_scene_id.x.min() + eps)
            df_scene_id["y_normailized"] = (df_scene_id.y - df_scene_id.y.min()) / (
                        df_scene_id.y.max() - df_scene_id.y.min() + eps)
            df_scene_id["z_normailized"] = (df_scene_id.z - df_scene_id.z.min()) / (
                        df_scene_id.z.max() - df_scene_id.z.min() + eps)

            # делаем min_max нормализацию всех координат, чтоб у нас были однородные данные в датасете

            tree = KDTree(df_scene_id[["x_normailized", "y_normailized", "z_normailized"]], leaf_size=5)

            # заводим KDTree чтоб искать для каждой точки сцены ее соседей и считать фичи на основе ее ближайших соседей

            dist_mean = []  # среднее расстояние от точки до всех ее соседей
            dist_std = []  # стандартное отклонение от точки до всех ее соседей
            dist_for_first = []  # расстоние до самой ближней точки (из найденных соседей)
            dist_for_last = []  # расстоние до самой дальней точки точки (из найденных соседей)

            # тоже самое для ring и intensity
            ring_mean = []
            ring_std = []
            ring_for_first = []
            ring_for_last = []

            int_mean = []
            int_std = []
            int_for_first = []
            int_for_last = []

            for i, row in df_scene_id.iterrows():  # ходим по точкам сцена и ищем соседей для этих точек
                dist, ind = tree.query([row[["x_normailized", "y_normailized", "z_normailized"]]],
                                       k=7)  # берем 7 ее соседей

                temp_df_indx = df_scene_id.loc[df_scene_id.index.intersection(ind[0][1:] + df_scene_id.index[0])]

                dist_mean.append(np.mean(dist[0][1:]))
                dist_std.append(np.std(dist[0][1:]))
                dist_for_first.append(dist[0][1])
                dist_for_last.append(dist[0][6])

                ring_mean.append(temp_df_indx.ring.mean())
                ring_std.append(temp_df_indx.ring.std())
                ring_for_first.append(temp_df_indx.ring.values[0])
                ring_for_last.append(temp_df_indx.ring.values[5])

                int_mean.append(temp_df_indx.intensity.mean())
                int_std.append(temp_df_indx.intensity.std())
                int_for_first.append(temp_df_indx.intensity.values[0])
                int_for_last.append(temp_df_indx.intensity.values[5])

            df_scene_id["dist_mean"] = np.array(dist_mean)
            df_scene_id["dist_std"] = np.array(dist_std)
            df_scene_id["dist_for_first"] = np.array(dist_for_first)
            df_scene_id["dist_for_last"] = np.array(dist_for_last)

            df_scene_id["ring_mean"] = np.array(ring_mean)
            df_scene_id["ring_std"] = np.array(ring_std)
            df_scene_id["ring_for_first"] = np.array(ring_for_first)
            df_scene_id["ring_for_last"] = np.array(ring_for_last)

            df_scene_id["int_mean"] = np.array(int_mean)
            df_scene_id["int_std"] = np.array(int_std)
            df_scene_id["int_for_first"] = np.array(int_for_first)
            df_scene_id["int_for_last"] = np.array(int_for_last)

            ans_df = ans_df.append(df_scene_id)  # сохраняю все фичи и генерю новый датасет
        return ans_df

    def kdtree_preprocess_fast(self, df):
        ans_df = pd.DataFrame()
        scenes = np.unique(df.scene_id.values)
        for scene_id in tqdm(scenes):   # начинаем ходить по каждой сцене
            df_scene_id = df[df.scene_id == scene_id]  # берем все данные из этой сцены
            eps = 0.001
            df_scene_id["x_normailized"] = (df_scene_id.x - df_scene_id.x.min()) / (
                        df_scene_id.x.max() - df_scene_id.x.min() + eps)
            df_scene_id["y_normailized"] = (df_scene_id.y - df_scene_id.y.min()) / (
                        df_scene_id.y.max() - df_scene_id.y.min() + eps)
            df_scene_id["z_normailized"] = (df_scene_id.z - df_scene_id.z.min()) / (
                        df_scene_id.z.max() - df_scene_id.z.min() + eps)

            # делаем min_max нормализацию всех координат, чтоб у нас были однородные данные в датасете

            tree = KDTree(df_scene_id[["x_normailized", "y_normailized", "z_normailized"]], leaf_size=5)

            # заводим KDTree чтоб искать для каждой точки сцены ее соседей и считать фичи на основе ее ближайших соседей

            dist_mean = []  # среднее расстояние от точки до всех ее соседей
            dist_for_last = []  # расстоние до самой дальней точки точки (из найденных соседей)

            int_mean = []


            for i, row in df_scene_id.iterrows():  # ходим по точкам сцена и ищем соседей для этих точек
                dist, ind = tree.query([row[["x_normailized", "y_normailized", "z_normailized"]]],
                                       k=7)  # берем 7 ее соседей

                temp_df_indx = df_scene_id.loc[df_scene_id.index.intersection(ind[0][1:] + df_scene_id.index[0])]

                dist_mean.append(np.mean(dist[0][1:]))
                dist_for_last.append(dist[0][6])

                int_mean.append(temp_df_indx.intensity.mean())

            df_scene_id["dist_mean"] = np.array(dist_mean)
            df_scene_id["dist_for_last"] = np.array(dist_for_last)

            df_scene_id["int_mean"] = np.array(int_mean)

            ans_df = ans_df.append(df_scene_id)  # сохраняю все фичи и генерю новый датасет
        return ans_df