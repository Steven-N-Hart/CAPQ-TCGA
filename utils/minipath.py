import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import logging
from PIL import Image
import pydicom
import os
from io import BytesIO
from google.cloud import storage
from utils import download_public_file
import pandas as pd


def read_dicom(dcm_input):
    """
    Load a DICOM file from a local path or Google Cloud Storage.

    :param dcm_input: Local file path or GCS path.
    :return: pydicom FileDataset object.
    """
    if isinstance(dcm_input, pd.Series):
        dcm_input = dcm_input.values[0]
    if isinstance(dcm_input, str):
        if dcm_input.startswith('gs://'):
            # Read DICOM from GCS
            return read_dicom_from_gcs(dcm_input)
        else:
            # Read local DICOM file
            return pydicom.dcmread(dcm_input)
    raise f"Could not complete with {dcm_input}"

def read_dicom_from_gcs(gcs_path):
    """
    Read a DICOM file from Google Cloud Storage.

    :param gcs_path: GCS path to the DICOM file.
    :return: pydicom FileDataset object.
    """
    # Split the GCS path to get bucket and file name
    path_parts = gcs_path.replace('gs://', '').split('/')
    bucket_name = path_parts[0]
    file_name = '/'.join(path_parts[1:])

    # Initialize a client and get the bucket
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
    except :
        blob = download_public_file(bucket_name, file_name, gcs_path, local=False)
    # Download the file as a bytes object
    dicom_bytes = blob.download_as_bytes()

    # Use pydicom to read the DICOM file from bytes
    dicom_file = pydicom.dcmread(BytesIO(dicom_bytes))

    return dicom_file


class Minipath:
    def __init__(self, img: Image.Image):
        """
        Initialize Minipath with an image and calculate its entropy.

        :param img: PIL Image object in RGB mode.
        """
        self.img = img

    @staticmethod
    def entropy_from_histogram(histogram: list[int]) -> float:
        """
        Compute entropy from a histogram.

        :param histogram: List of pixel counts.
        :return: Entropy value.
        """
        hist_length = sum(histogram)
        probability = [float(h) / hist_length for h in histogram]
        return -sum([p * np.log2(p) for p in probability if p != 0])

    def extract_entropy_feature_vector(self, img_patch: Image.Image) -> np.ndarray:
        """
        Extract entropy-based feature vector from an image patch.

        :param img_patch: PIL Image object in RGB mode.
        :return: Numpy array containing entropy values for each RGB channel.
        """
        if img_patch.mode != 'RGB':
            raise ValueError('Image patch should be RGB')

        # Extract histograms for each channel
        histogram_r = img_patch.histogram()[0:256]
        histogram_g = img_patch.histogram()[256:512]
        histogram_b = img_patch.histogram()[512:768]

        # Compute entropy for each channel
        entropy_r = self.entropy_from_histogram(histogram_r)
        entropy_g = self.entropy_from_histogram(histogram_g)
        entropy_b = self.entropy_from_histogram(histogram_b)

        # Form the feature vector using entropy values
        feature_vector = np.array([entropy_r, entropy_g, entropy_b])

        return feature_vector

    @staticmethod
    def determine_optimal_components(scaled_features: np.ndarray, explained_variance: float = 0.8) -> int:
        """
        Determine the optimal number of PCA components.

        :param scaled_features: Scaled feature matrix.
        :param explained_variance: Threshold for cumulative explained variance.
        :return: Optimal number of components.
        """
        pca = PCA()
        pca.fit(scaled_features)
        cum_exp_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cum_exp_variance > explained_variance) + 1  # +1 because index starts from 0
        logging.debug(f'n_components: {n_components}, cum_exp_variance: {cum_exp_variance}')
        return n_components

    @staticmethod
    def calculate_euclidean(pca_features: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate Euclidean distances between PCA features and centroids.

        :param pca_features: Numpy array of PCA features.
        :param centroids: Numpy array of centroid coordinates.
        :return: Indices of the closest samples and ordered sample indices.
        """
        distances = np.linalg.norm(pca_features - centroids[:, np.newaxis], axis=2)
        closest_samples_idx = np.argmin(distances, axis=1)
        ordered_samples_idx = np.argsort(np.sum(distances, axis=0))
        return closest_samples_idx, ordered_samples_idx

    @staticmethod
    def determine_optimal_clusters(features: np.ndarray, max_k: int = 50) -> int:
        """
        Determine the optimal number of clusters using the elbow method.

        :param features: Feature matrix.
        :param max_k: Maximum number of clusters to test.
        :return: Optimal number of clusters.
        """
        num_unique_data_points = len(set(tuple(row) for row in features))
        wcss = []

        for i in range(1, min(max_k, num_unique_data_points)):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(features)
            wcss.append(kmeans.inertia_)

        distances = []
        for i in range(len(wcss)):
            x = i + 1  # +1 because the range of ks starts from 1
            y = wcss[i]
            a = (wcss[-1] - wcss[0]) / len(wcss)  # Slope of the line
            b = wcss[0]  # Intercept
            distance = abs(a * x - y + b) / np.sqrt(a ** 2 + 1)  # Distance formula from a point to a line
            distances.append(distance)

        elbow_k = distances.index(max(distances)) + 1

        return elbow_k

    def rank_patches_for_diversity(
            self, img_size: int = 256, patch_size: int = 8, explained_variance: float = 0.9, max_k: int = 50
    ) -> dict:
        """
        Rank patches for diversity using clustering and PCA.

        :param img_size: Size to which the image is resized.
        :param patch_size: Size of each patch.
        :param explained_variance: Threshold for cumulative explained variance in PCA.
        :param max_k: Maximum number of clusters to test.
        :return: Dictionary containing clustering and PCA results.
        """
        # Rescale the image
        img_resized = self.img.resize((img_size, img_size))
        scale_factor_x = self.img.width / img_size
        scale_factor_y = self.img.height / img_size

        # Extract patches from the resized image and their coordinates
        resized_patches = [
            (img_resized.crop((x, y, x + patch_size, y + patch_size)), (x, y))
            for x in range(0, img_size, patch_size)
            for y in range(0, img_size, patch_size)
        ]

        # Extract patches from the original image and their coordinates
        original_patches = [
            (self.img.crop((int(x * scale_factor_x), int(y * scale_factor_y), int((x + patch_size) * scale_factor_x),
                            int((y + patch_size) * scale_factor_y))),
             (int(x * scale_factor_x), int(y * scale_factor_y)))
            for x in range(0, img_size, patch_size)
            for y in range(0, img_size, patch_size)
        ]

        # Extract entropy-based features from patches
        features = np.array([self.extract_entropy_feature_vector(patch[0]) for patch in resized_patches])

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Determine optimal number of PCA components
        n_components = self.determine_optimal_components(scaled_features, explained_variance=explained_variance)

        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)

        # Determine optimal number of clusters
        n_clusters = self.determine_optimal_clusters(pca_features, max_k=max_k)

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(pca_features)
        closest_samples_idx, ordered_samples_idx = self.calculate_euclidean(pca_features, kmeans.cluster_centers_)

        patches_with_labels = list(zip(original_patches, cluster_labels))
        patches_with_labels.sort(key=lambda x: x[1])

        silhouette_avg = silhouette_score(pca_features, cluster_labels)

        results = {
            'patches_with_labels': patches_with_labels,
            'n_clusters': n_clusters,
            'n_components': n_components,
            'closest_samples_idx': closest_samples_idx,
            'ordered_samples_idx': ordered_samples_idx,
            'silhouette_avg': silhouette_avg,
        }

        return results


class MagPairs:
    def __init__(self, low_mag_dcm, img_to_use_at_low_mag=None, bq_results_df=None):
        self.low_mag_dcm = self._load_dcm(low_mag_dcm)
        self.high_mag_dcm = self._load_dcm(self.get_local_dcm_pair(low_mag_dcm, bq_results_df))
        self.low_mag_img = get_single_dcm_img(low_mag_dcm)
        self.pixel_spacing_at_low_mag = self.get_pixel_spacing(low_mag_dcm)
        self.pixel_spacing_at_high_mag = self.get_pixel_spacing(self.high_mag_dcm)
        self.scaling_factor = int(self.pixel_spacing_at_low_mag / self.pixel_spacing_at_high_mag)
        self.fd = self.get_frame_dict(self.high_mag_dcm)
        self.minmax_list = self.get_minmax(img_to_use_at_low_mag)
        self.high_mag_frame_list = [
            self.find_intersecting_frames(self.fd, m['x_min'], m['x_max'], m['y_min'], m['y_max']) for m in
            self.minmax_list]
        self.high_mag_frames = self.frame_extraction(self.high_mag_dcm, self.high_mag_frame_list)
        self.clean_high_mag_frames = [frame for frame in self.high_mag_frames if self.is_foreground(frame['img_array'])]


    @staticmethod
    def is_foreground(tile) -> bool:
        """
        Function to determine if a tile shows mainly tissue (foreground) or background.
        Returns True if tile shows <= 50% background and False otherwise.
        """
        if isinstance(tile, np.ndarray):
            tile = Image.fromarray(tile)
        grey = tile.convert(mode='L')
        thresholded = grey.point(lambda x: 0 if x < 220 else 1, mode='F')
        avg_bkg = np.average(np.array(thresholded))
        return avg_bkg <= 0.5

    @staticmethod
    def frame_extraction(dcm, high_mag_frame_list):
        img_array_list = list()
        pixel_array = dcm.pixel_array
        for high_mag_frame in high_mag_frame_list:
            for j in high_mag_frame:
                frame_id = j['frame']
                j['img_array'] = pixel_array[frame_id]
                img_array_list.append(j)
        return img_array_list

    @staticmethod
    def get_local_name(gcs_url, data_dir):
        blob = '/'.join(gcs_url.values[0].split('/')[3:])
        return os.path.join(data_dir, blob)

    @staticmethod
    def find_intersecting_frames(fd, x_min, x_max, y_min, y_max):
        """
        Find all frames that intersect with the given coordinates.

        Parameters:
        - fd: List of dictionaries with frame data containing 'row_min', 'row_max', 'col_min', 'col_max', and 'frame'.
        - x_min, x_max: x-coordinate range to check for intersection.
        - y_min, y_max: y-coordinate range to check for intersection.

        Returns:
        - List of dictionaries that intersect with the given coordinates.
        """
        intersecting_frames = []

        for frame_data in fd:
            row_min = frame_data['row_min']
            row_max = frame_data['row_max']
            col_min = frame_data['col_min']
            col_max = frame_data['col_max']

            # Check for intersection in both x and y ranges
            if (x_min <= col_max and x_max >= col_min) and (y_min <= row_max and y_max >= row_min):
                intersecting_frames.append(frame_data)

        return intersecting_frames

    def get_minmax(self, img_to_use_at_low_mag):
        minmax_list = []
        for i in img_to_use_at_low_mag:
            x_range, y_range = i[0][0].size
            raw_ranges = i[0][1]
            x_min = raw_ranges[0] * self.scaling_factor
            x_max = (raw_ranges[0] + x_range) * self.scaling_factor
            y_min = raw_ranges[1] * self.scaling_factor
            y_max = (raw_ranges[1] + y_range) * self.scaling_factor
            logging.debug(
                f'x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}, '
                f'x_range: {x_range}, y_range: {y_range}, raw_ranges:{raw_ranges} ')
            minmax_list.append({'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'x_range': x_range})
        return minmax_list

    def get_local_dcm_pair(self, dcm, bq_results_df):
        gcs_url_pair = bq_results_df['gcs_url'][
            (bq_results_df['SeriesInstanceUID'] == dcm.SeriesInstanceUID) & (bq_results_df['row_num_desc'] == 1)]
        #local_pair_name = self.get_local_name(gcs_url_pair, data_dir)
        return read_dicom(gcs_url_pair)

    @staticmethod
    def _load_dcm(dcm_input):
        if isinstance(dcm_input, str):
            dcm = pydicom.dcmread(dcm_input)
        elif isinstance(dcm_input, pydicom.dataset.FileDataset):
            dcm = dcm_input
        else:
            raise ValueError("Input must be a DICOM file path string or a pydicom DICOM object")
        return dcm

    @staticmethod
    def get_pixel_spacing(dcm):
        return float(dcm.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0])

    @staticmethod
    def get_frame_dict(dcm_input):
        dcm, total_pixel_matrix_columns, total_pixel_matrix_rows, columns, rows, grid_rows, grid_cols = parse_dcm_info(
            dcm_input)
        frame_list = list()
        frame_index = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                frame_list.append({'row_min': row * rows, 'row_max': row * rows + rows, 'col_min': col * columns,
                                   'col_max': col * columns + columns, 'frame': frame_index})
                frame_index += 1

        return frame_list


def get_single_dcm_img(dcm_input) -> np.ndarray:
    dcm, total_pixel_matrix_columns, total_pixel_matrix_rows, columns, rows, grid_rows, grid_cols = parse_dcm_info(dcm_input)

    frames = dcm.pixel_array

    if len(frames.shape) != 4:
        raise ValueError("Expected frames to have shape (num_frames, height, width, channels)")

    num_frames, frame_height, frame_width, channels = frames.shape

    if grid_rows * grid_cols != num_frames:
        raise ValueError(f"Expected {grid_rows * grid_cols} frames, but got {num_frames}")

    # Create an empty array to hold the grid
    grid_array = np.zeros((grid_rows * frame_height, grid_cols * frame_width, channels), dtype=np.uint8)

    # Populate the grid array using nested loops
    frame_index = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if frame_index < num_frames:
                start_row = row * frame_height
                end_row = (row + 1) * frame_height
                start_col = col * frame_width
                end_col = (col + 1) * frame_width
                grid_array[start_row:end_row, start_col:end_col, :] = frames[frame_index]
                frame_index += 1

    return grid_array



def dcm_checker(dcm_input):
    if isinstance(dcm_input, str):
        dcm = pydicom.dcmread(dcm_input)
    elif isinstance(dcm_input, pydicom.dataset.FileDataset):
        dcm = dcm_input
    else:
        raise ValueError("Input must be a DICOM file path string or a pydicom DICOM object")
    return dcm


def parse_dcm_info(dcm_input):
    dcm = dcm_checker(dcm_input)
    # Extract necessary metadata
    total_pixel_matrix_columns = dcm.TotalPixelMatrixColumns
    total_pixel_matrix_rows = dcm.TotalPixelMatrixRows
    columns = dcm.Columns
    rows = dcm.Rows

    # Calculate grid size
    grid_cols = int(np.ceil(total_pixel_matrix_columns / columns))
    grid_rows = int(np.ceil(total_pixel_matrix_rows / rows))
    return dcm, total_pixel_matrix_columns, total_pixel_matrix_rows, columns, rows, grid_rows, grid_cols
