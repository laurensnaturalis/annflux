import os
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm


import os
import cv2
import csv
from pycocotools.coco import COCO
from tqdm import tqdm

def extract_crops_and_labels(coco_annotation_file, images_dir, output_dir, csv_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO annotations
    coco = COCO(coco_annotation_file)

    # Open CSV file for writing
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['basename of crop', 'label_true']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Get all image IDs
        img_ids = coco.getImgIds()

        for img_id in tqdm(img_ids):
            # Load image info
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(images_dir, img_info['file_name'])
            img = cv2.imread(img_path)

            # Load annotations for the current image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)

            for ann in annotations:
                # Get bounding box coordinates
                x, y, w, h = ann['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)

                # Crop the image
                crop = img[y:y+h, x:x+w]

                # Generate output filename
                output_filename = f"img_{img_id}_ann_{ann['id']}.jpg"
                output_path = os.path.join(output_dir, output_filename)

                # Save the crop
                cv2.imwrite(output_path, crop)

                # Get category name
                cat_id = ann['category_id']
                cat_info = coco.loadCats(cat_id)[0]
                label = cat_info['name']

                # Write to CSV
                writer.writerow({
                    'basename of crop': output_filename,
                    'label_true': label
                })

# extract_crops_and_labels(coco_annotation_file, images_dir, output_dir, csv_path)



def m():
    extract_crops_and_labels(
        "/mnt/big/Projects/diopsis_coco_public/instances_train.json",
        "/mnt/big/Projects/diopsis_coco_public/images",
        "/home/lhogeweg/annflux/datasources/diopsis-coco/images",
        "/home/lhogeweg/annflux/datasources/diopsis-coco/labels.csv",
    )


if __name__ == "__main__":
    m()
