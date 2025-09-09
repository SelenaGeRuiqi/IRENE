"""
Data Validator for IRENE processed image data.
"""

import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def validate_data(data_dir):
    """Validate processed data integrity."""
    print("Validating processed data...")
    print("=" * 50)

    # Check required files
    required_files = {
        'train.pkl': os.path.join(data_dir, 'train.pkl'),
        'test.pkl': os.path.join(data_dir, 'test.pkl'),
        'disease_info.pkl': os.path.join(data_dir, 'disease_info.pkl'),
        'images/': os.path.join(data_dir, 'images')
    }

    print("1. File existence check:")
    for name, path in required_files.items():
        exists = os.path.exists(path)
        print(f"   {name}: {'✓' if exists else '✗'}")
        if not exists:
            print(f"ERROR: Missing {path}")
            return False

    # Load disease info
    with open(required_files['disease_info.pkl'], 'rb') as f:
        disease_info = pickle.load(f)

    print(f"\n2. Disease information:")
    print(f"   Classes: {disease_info['num_classes']}")
    print(f"   Diseases: {disease_info['disease_list']}")

    # Load train/test data
    with open(required_files['train.pkl'], 'rb') as f:
        train_data = pickle.load(f)

    with open(required_files['test.pkl'], 'rb') as f:
        test_data = pickle.load(f)

    print(f"\n3. Dataset splits:")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")

    # Check data consistency
    all_ids = set(train_data.keys()) | set(test_data.keys())
    image_files = os.listdir(required_files['images/'])
    image_ids = {f.replace('.png', '') for f in image_files if f.endswith('.png')}

    print(f"\n4. Data consistency:")
    print(f"   Total unique IDs: {len(all_ids)}")
    print(f"   Image files: {len(image_ids)}")
    print(f"   ID-Image match: {'✓' if all_ids == image_ids else '✗'}")

    if all_ids != image_ids:
        missing_images = all_ids - image_ids
        extra_images = image_ids - all_ids
        if missing_images:
            print(f"   Missing images: {list(missing_images)[:5]}...")
        if extra_images:
            print(f"   Extra images: {list(extra_images)[:5]}...")

    # Check label distribution
    print(f"\n5. Label distribution:")
    all_labels = []
    for data in [train_data, test_data]:
        for sample in data.values():
            all_labels.append(sample['label'])

    all_labels = np.array(all_labels)
    label_counts = np.sum(all_labels, axis=0)

    for i, disease in enumerate(disease_info['disease_list']):
        count = int(label_counts[i])
        print(f"   {disease}: {count} samples")
        if count < 5:
            print(f"     WARNING: Very few samples for {disease}")

    # Check sample data format
    print(f"\n6. Data format check:")
    sample_id = list(train_data.keys())[0]
    sample = train_data[sample_id]

    print(f"   Sample ID: {sample_id}")
    print(f"   Label shape: {sample['label'].shape}")
    print(f"   Label sum: {sample['label'].sum()}")
    print(f"   Placeholder shapes: pdesc{sample['pdesc'].shape}, bics{sample['bics'].shape}, bts{sample['bts'].shape}")

    # Check image integrity
    print(f"\n7. Image integrity check (first 3):")
    for i, img_file in enumerate(image_files[:3]):
        img_path = os.path.join(required_files['images/'], img_file)
        try:
            img = Image.open(img_path)
            print(f"   {img_file}: {img.size}, {img.mode}")
        except Exception as e:
            print(f"   {img_file}: ERROR - {e}")

    print("\n✓ Validation complete")
    return True, disease_info


def visualize_samples(data_dir, num_samples=6):
    """Generate sample visualization."""
    print(f"\nGenerating sample visualization...")

    train_path = os.path.join(data_dir, 'train.pkl')
    disease_info_path = os.path.join(data_dir, 'disease_info.pkl')
    images_dir = os.path.join(data_dir, 'images')

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(disease_info_path, 'rb') as f:
        disease_info = pickle.load(f)

    disease_list = disease_info['disease_list']

    # Select diverse samples
    sample_ids = []
    used_diseases = set()

    for image_id, data in train_data.items():
        if len(sample_ids) >= num_samples:
            break

        label = data['label']
        disease_idx = np.argmax(label)
        disease_name = disease_list[disease_idx]

        if disease_name not in used_diseases:
            sample_ids.append(image_id)
            used_diseases.add(disease_name)

    # Fill remaining slots
    remaining_ids = [id for id in train_data.keys() if id not in sample_ids]
    sample_ids.extend(remaining_ids[:num_samples - len(sample_ids)])

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, image_id in enumerate(sample_ids[:num_samples]):
        img_path = os.path.join(images_dir, f"{image_id}.png")

        if os.path.exists(img_path):
            img = Image.open(img_path)
            label = train_data[image_id]['label']
            disease_idx = np.argmax(label)
            disease_name = disease_list[disease_idx]

            axes[i].imshow(img)
            axes[i].set_title(f"{image_id}\n{disease_name}", fontsize=10)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Missing\n{image_id}",
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(sample_ids), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    output_path = os.path.join(data_dir, 'sample_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Visualization saved: {output_path}")


def print_training_info(disease_info):
    """Print information needed for training."""
    print(f"\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Number of classes (--CLS): {disease_info['num_classes']}")
    print(f"Disease classes: {disease_info['disease_list']}")
    print(f"Data directory (--DATA_DIR): ./processed_data/images")
    print(f"Training mode (--MODE): image")

    print(f"\nSample training command:")
    print(f"python irene.py \\")
    print(f"    --CLS {disease_info['num_classes']} \\")
    print(f"    --BSZ 16 \\")
    print(f"    --DATA_DIR ./processed_data/images \\")
    print(f"    --SET_TYPE train \\")
    print(f"    --MODE image")


def main():
    data_dir = "./processed_data"

    success, disease_info = validate_data(data_dir)
    if success:
        visualize_samples(data_dir)
        print_training_info(disease_info)


if __name__ == "__main__":
    main()