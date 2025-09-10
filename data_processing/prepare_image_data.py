"""
处理肺部三个方向的AIP投影图像(暂时只用了axial path，没有做融合)
axial_path:轴状面投影图像路径，指向轴状面的平均强度投影图像(.png格式)
coronal_path:冠状面投影图像路径，指向冠状面的平均强度投影图像(.png格式)
sagittal_path:矢状面投影图像路径，指向矢状面的平均强度投影图像(.png格式)

Creates train.pkl / test.pkl with exactly the required dict schema in readme file
Files are saved under processed_data/ (not ./data/)
processed_data/images/ for PNGs
processed_data/disease_info.pkl for label meta.
"""

import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm


class ImageDataProcessor:
    def __init__(self, projection_csv_path, data_csv_path):
        """
        Args:
            projection_csv_path: Path to lung_intensity_projection.csv
            data_csv_path: Path to lung_data_merged.csv
        """
        self.projection_csv_path = projection_csv_path
        self.data_csv_path = data_csv_path

        print("Loading CSV files...")
        self.df_projection = pd.read_csv(projection_csv_path)
        self.df_data = pd.read_csv(data_csv_path)

        print(f"Projection data: {len(self.df_projection)} records")
        print(f"Label data: {len(self.df_data)} records")

        self.disease_mapping = None
        self.disease_list = None
        self.num_classes = 0

    def analyze_data(self):
        """Analyze dataset and build disease mapping."""
        print("\nAnalyzing dataset...")

        # Get image IDs from both CSVs
        projection_ids = set(self.df_projection.iloc[:, 0])  # Column A
        data_ids = set(self.df_data.iloc[:, 0])  # Column A
        common_ids = projection_ids & data_ids

        print(f"Projection CSV IDs: {len(projection_ids)}")
        print(f"Data CSV IDs: {len(data_ids)}")
        print(f"Common IDs: {len(common_ids)}")
        print(f"Match rate: {len(common_ids) / len(projection_ids) * 100:.1f}%")

        # Analyze disease distribution
        disease_counts = self.df_data.iloc[:, 4].value_counts()  # Column E
        self.disease_list = list(disease_counts.index)
        self.num_classes = len(self.disease_list)
        self.disease_mapping = {disease: idx for idx, disease in enumerate(self.disease_list)}

        print(f"\nFound {self.num_classes} disease classes:")
        for disease, count in disease_counts.items():
            print(f"  {disease}: {count} samples")

        # Check image file existence (sample check)
        print(f"\nChecking image files (first 10)...")
        sample_ids = list(common_ids)[:10]
        existing = 0

        for image_id in sample_ids:
            proj_row = self.df_projection[self.df_projection.iloc[:, 0] == image_id].iloc[0]
            axial_path = proj_row.iloc[3]  # Column D

            if os.path.exists(axial_path):
                existing += 1
            else:
                print(f"  Missing: {axial_path}")

        print(f"  {existing}/{len(sample_ids)} files found")

        return common_ids

    def process_labels(self, valid_ids):
        """Process disease labels into one-hot vectors."""
        print("\nProcessing labels...")

        valid_data = self.df_data[self.df_data.iloc[:, 0].isin(valid_ids)].copy()
        print(f"Processing {len(valid_data)} samples")

        labels_dict = {}

        for _, row in tqdm(valid_data.iterrows(), total=len(valid_data), desc="Labels"):
            image_id = row.iloc[0]  # Column A
            disease = row.iloc[4]  # Column E

            if pd.isna(disease):
                continue

            if disease in self.disease_mapping:
                label_vector = np.zeros(self.num_classes, dtype=np.float32)
                label_vector[self.disease_mapping[disease]] = 1.0
                labels_dict[image_id] = label_vector

        print(f"Successfully processed {len(labels_dict)} labels")

        # Print label distribution
        label_counts = np.zeros(self.num_classes)
        for label_vector in labels_dict.values():
            label_counts += label_vector

        print("Label distribution:")
        for i, disease in enumerate(self.disease_list):
            print(f"  {disease}: {int(label_counts[i])} samples")

        return labels_dict

    def process_images(self, valid_ids):
        """Process axial images."""
        print("\nProcessing images...")

        valid_projection = self.df_projection[self.df_projection.iloc[:, 0].isin(valid_ids)]

        image_data = {}
        failed_ids = []

        for _, row in tqdm(valid_projection.iterrows(), total=len(valid_projection), desc="Images"):
            image_id = row.iloc[0]  # Column A
            axial_path = row.iloc[3]  # Column D

            try:
                if not os.path.exists(axial_path):
                    raise FileNotFoundError(f"Image not found: {axial_path}")

                # Load image
                image = Image.open(axial_path)
                image_data[image_id] = image

            except Exception as e:
                print(f"Failed to process {image_id}: {e}")
                failed_ids.append(image_id)
                continue

        print(f"Successfully processed {len(image_data)} images")
        print(f"Failed: {len(failed_ids)} images")

        return image_data, failed_ids

    def save_data(self, image_data, labels_dict, output_dir, test_size=0.2):
        """Save processed data in IRENE format."""
        print("\nSaving processed data...")

        os.makedirs(output_dir, exist_ok=True)

        # Get samples with both image and label
        common_ids = set(image_data.keys()) & set(labels_dict.keys())
        print(f"Final dataset: {len(common_ids)} samples")

        if len(common_ids) == 0:
            print("No valid samples found")
            return

        # Train/test split
        train_ids, test_ids = train_test_split(
            list(common_ids),
            test_size=test_size,
            random_state=42
        )

        print(f"Train set: {len(train_ids)} samples")
        print(f"Test set: {len(test_ids)} samples")

        # Save images
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        for image_id in tqdm(common_ids, desc="Saving images"):
            image_path = os.path.join(images_dir, f"{image_id}.png")
            image_data[image_id].save(image_path)

        # Save train/test data
        self._save_split(train_ids, labels_dict, output_dir, 'train')
        self._save_split(test_ids, labels_dict, output_dir, 'test')

        # Save disease info
        disease_info = {
            'disease_mapping': self.disease_mapping,
            'disease_list': self.disease_list,
            'num_classes': self.num_classes
        }

        with open(os.path.join(output_dir, 'disease_info.pkl'), 'wb') as f:
            pickle.dump(disease_info, f)

        print("Data processing completed!")
        print(f"Output directory: {output_dir}")
        print(f"Images: {images_dir}")
        print(f"Train data: train.pkl")
        print(f"Test data: test.pkl")
        print(f"Disease info: disease_info.pkl")

    def _save_split(self, ids, labels_dict, output_dir, split_name):
        """Save train/test split data."""
        split_data = {}

        for image_id in ids:
            split_data[image_id] = {
                'label': labels_dict[image_id],
                'pdesc': np.zeros((40, 768), dtype=np.float32),  # Placeholder for text
                'bics': np.array([0.0, 0.0], dtype=np.float32),  # Placeholder for demographics
                'bts': np.zeros(92, dtype=np.float32),  # Placeholder for lab data
            }

        output_path = os.path.join(output_dir, f'{split_name}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(split_data, f)

        print(f"Saved {split_name} split: {len(split_data)} samples")


def main():
    parser = argparse.ArgumentParser(description="Process IRENE image data")
    parser.add_argument('--csv_projection', type=str,
                        default='./data/lung_intensity_projection.csv',
                        help='Path to projection CSV file')
    parser.add_argument('--csv_data', type=str,
                        default='./data/lung_data_merged.csv',
                        help='Path to data CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='./processed_data',
                        help='Output directory')

    args = parser.parse_args()

    print("IRENE Image Data Processing")
    print(f"Projection CSV: {args.csv_projection}")
    print(f"Data CSV: {args.csv_data}")
    print(f"Output directory: {args.output_dir}")

    # Initialize processor
    processor = ImageDataProcessor(args.csv_projection, args.csv_data)

    # Process data
    valid_ids = processor.analyze_data()
    labels_dict = processor.process_labels(valid_ids)
    image_data, failed_ids = processor.process_images(valid_ids)
    processor.save_data(image_data, labels_dict, args.output_dir)

    print(f"\nProcessing complete: {processor.num_classes} classes, {len(image_data)} images")


if __name__ == "__main__":
    main()