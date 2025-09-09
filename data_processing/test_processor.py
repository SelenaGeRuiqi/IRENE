"""
Test processor for data pipeline validation.
"""

import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.prepare_image_data import ImageDataProcessor


def test_csv_loading():
    """Test CSV file loading and structure."""
    print("Testing CSV file loading...")

    csv_projection = './data/lung_intensity_projection.csv'
    csv_data = './data/lung_data_merged.csv'

    try:
        df_proj = pd.read_csv(csv_projection)
        df_data = pd.read_csv(csv_data)

        print(f"✓ Projection CSV: {df_proj.shape}")
        print(f"  Columns: {list(df_proj.columns)}")
        print(f"  Sample IDs: {df_proj.iloc[:3, 0].tolist()}")
        print(f"  Sample paths: {df_proj.iloc[:3, 3].tolist()}")

        print(f"✓ Data CSV: {df_data.shape}")
        print(f"  Columns: {list(df_data.columns)}")
        print(f"  Sample IDs: {df_data.iloc[:3, 0].tolist()}")
        print(f"  Sample diseases: {df_data.iloc[:3, 4].tolist()}")

        # Disease analysis
        diseases = df_data.iloc[:, 4].value_counts()
        print(f"\n✓ Found {len(diseases)} disease types:")
        for disease, count in diseases.head(10).items():
            print(f"  {disease}: {count}")

        return True

    except Exception as e:
        print(f"✗ CSV loading failed: {e}")
        return False


def test_image_access():
    """Test image file accessibility."""
    print("\nTesting image file access...")

    csv_projection = './data/lung_intensity_projection.csv'

    try:
        df_proj = pd.read_csv(csv_projection)

        # Check first 5 axial images
        test_count = min(5, len(df_proj))
        accessible = 0

        for i in range(test_count):
            image_id = df_proj.iloc[i, 0]
            axial_path = df_proj.iloc[i, 3]

            if os.path.exists(axial_path):
                accessible += 1
                print(f"✓ {image_id}: {os.path.basename(axial_path)}")
            else:
                print(f"✗ {image_id}: {axial_path}")

        print(f"\nAccessible: {accessible}/{test_count} images")
        return accessible > 0

    except Exception as e:
        print(f"✗ Image access test failed: {e}")
        return False


def test_small_batch():
    """Test processing with small batch."""
    print("\nTesting small batch processing...")

    try:
        processor = ImageDataProcessor(
            './data/lung_intensity_projection.csv',
            './data/lung_data_merged.csv'
        )

        # Analyze data
        valid_ids = processor.analyze_data()
        print(f"✓ Found {len(valid_ids)} valid IDs")

        if len(valid_ids) == 0:
            print("✗ No valid IDs found")
            return False

        # Test with first 5 samples
        test_ids = list(valid_ids)[:5]

        # Process labels
        labels = processor.process_labels(test_ids)
        print(f"✓ Processed {len(labels)} labels")

        # Process images
        images, failed = processor.process_images(test_ids)
        print(f"✓ Processed {len(images)} images, {len(failed)} failed")

        if len(images) > 0:
            # Test save
            processor.save_data(images, labels, './test_output', test_size=0.4)
            print("✓ Test save completed")
            return True
        else:
            print("✗ No images processed successfully")
            return False

    except Exception as e:
        print(f"✗ Small batch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("IRENE Data Processing Test Suite")
    print("=" * 40)

    # Test sequence
    tests = [
        ("CSV Loading", test_csv_loading),
        ("Image Access", test_image_access),
        ("Small Batch", test_small_batch)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        success = test_func()
        results.append((test_name, success))

        if not success:
            print(f"\n✗ {test_name} failed. Please fix before proceeding.")
            break

    # Summary
    print(f"\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")

    if all(result[1] for result in results):
        print(f"\n✓ All tests passed! Ready for full processing.")
        print(f"\nRun full processing with:")
        print(f"python data_processing/prepare_image_data.py")
    else:
        print(f"\n✗ Some tests failed. Please check configuration.")


if __name__ == "__main__":
    main()