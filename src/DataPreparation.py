import os
import psutil
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, log):
        self.x = np.array([])
        self.y = np.array([])
        self.__log = log
        self.__img_bytes = 784

    def get_dataset(
        self,
        data_dir: str = "data",
        raw_dir: str = "data/raw",
        class_size: int = 5_000,
        x_path: str = "data/x.npy",
        y_path: str = "data/y.npy",
    ):
        self.__check_ram(class_size=class_size)

        if os.path.exists(x_path) and os.path.exists(y_path):
            self.__log.write_log(
                f"File {x_path} and {y_path} exist and will be use, check the version of it",
                "WARNING",
            )
            return np.load(x_path), np.load(y_path)

        self.__log.write_log(
            f"The data is not in data folder '{data_dir}', loading data. It will take a while",
            "WARNING",
        )

        pbar = tqdm(os.listdir(raw_dir))
        for np_file in pbar:
            batch = np.load(f"data/raw/{np_file}").reshape(-1, 28, 28)
            name = np_file.replace(".npy", "").split("_")[-1]
            if class_size != None:
                np.random.shuffle(batch)
                nums_batch = len(batch) // class_size
                batch = np.array_split(batch, nums_batch)[0]

            pbar.set_description(f"Loading {name}")

            if self.x.shape[0] == 0:
                self.x = batch
                self.y = [name] * batch.shape[0]
                continue

            self.x = np.append(self.x, batch).reshape(-1, 28, 28)
            self.y = np.append(self.y, [name] * batch.shape[0])
            del batch

        np.save(x_path, self.x)
        np.save(y_path, self.y)
        self.__log.write_log(f"Saving data in data folder ({data_dir})", "SUCCESS")
        del self.x, self.y
        return np.load(x_path), np.load(y_path)

    def __check_ram(self, class_size: int = None):
        if class_size == None:
            class_size = 140_000

        total_memory = (((345 * class_size * self.__img_bytes) / 1024) / 1024) / 1024
        total_ram = psutil.virtual_memory().total / 1000000000
        self.__log.write_log(
            f"Total RAM {round(total_ram, 2)} GB",
            "INFO",
        )

        self.__log.write_log(
            f"Total memory ued {round(total_memory, 2)} GB",
            "INFO",
        )
        diff = total_ram - total_memory

        if diff < 0:
            self.__log.write_log(
                f"Not enough RAM to load the data, need {round(total_memory, 2)} GB, have {round(total_ram, 2)} GB",
                "CRITICAL",
            )
            exit()
        elif diff > 0 and diff < 1:
            self.__log.write_log(
                "There is little RAM so the program could take a long time and could crash",
                "WARNING",
            )
        else:
            self.__log.write_log(
                f"There is enough ram. You have {round(diff, 2)} GB of ram available",
                "INFO",
            )

    def get_train_test_split(self, x, y, test_size: float = 0.2):
        self.__log.write_log(
            f"Splitting data into train and test sets, test size is {math.ceil(x.shape[0]*test_size)}",
            "INFO",
        )
        return train_test_split(x, y, test_size=test_size, random_state=21)

    def __zoom_at(self, img, x, y, zoom):
        w, h = img.size
        zoom2 = zoom * 2
        img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))
        return img.resize((w, h), Image.LANCZOS)

    def data_augmentation(
        self, x, y, n_augmentation: int = 1_000, x_dir: str = "data/x_augmented.npy", y_dir: str = "data/Y_augmented.npy"
    ):
        if os.path.exists(x_dir) and os.path.exists(y_dir):
            self.__log.write_log(
                f"File {x_dir} and {y_dir} exist and will be use, check the version of it",
                "WARNING",
            )
            return np.load(x_dir), np.load(y_dir)

        x_augmented = []
        y_augmented = []
        for img_array, y in tqdm(zip(x, y), leave=False, total=x.shape[0]):
            x_augmented.append(img_array)
            y_augmented.append(y)
            for _ in range(n_augmentation):
                img = Image.fromarray(img_array)
                img = img.rotate(np.random.randint(-90, 90))
                img = img.resize((28, 28))
                img = self.__zoom_at(
                    img, np.random.randint(0, 28), np.random.randint(0, 28), 0.6
                )
                x_augmented.append(np.array(img))
                y_augmented.append(y)

        x_augmented = np.array(x_augmented)
        y_augmented = np.array(y_augmented)
        
        np.save(x_dir, x_augmented)
        np.save(y_dir, y_augmented)
        
        return x_augmented, y_augmented
