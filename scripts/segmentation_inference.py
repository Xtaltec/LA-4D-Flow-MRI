import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from src.utils.parsing import parse_predict
from src.utils.seg import prediction

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH", "./")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    args = parse_predict()
    model_name = args.model_name
    structure = model_name.split("_")[-1]
    max_retries = args.max_retries
    path_to_predict = Path(args.path_to_predict)
    path_output = Path(args.path_output) / structure

    # Create mask directory based on model name
    mask_path = Path(BASE_PATH) / "data" / "processed" / "segmentation" / "masks" / structure
    mask_path.mkdir(parents=True, exist_ok=True)

    # Create output directory if it doesn't exist
    path_output.mkdir(parents=True, exist_ok=True)

    tmp_pred_path = Path(str(path_output).replace("predicted", "to_be_predicted"))

    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1}/{max_retries}")

        # Check if there are images to predict
        images_to_predict = list(path_to_predict.glob("*.nii.gz"))
        if not images_to_predict:
            logger.info("No more images to predict")
            break

        # Create temporary directory
        tmp_pred_path.mkdir(parents=True, exist_ok=True)

        # Copy images to the temporary directory, adding _0000 suffix if needed
        for image in images_to_predict:
            if "_0000" not in image.name:
                new_name = image.name.replace(".nii.gz", "_0000.nii.gz")
                new_path = tmp_pred_path / new_name
            else:
                new_path = tmp_pred_path / image.name

            try:
                shutil.copy2(image, new_path)
            except (OSError, shutil.Error) as e:
                logger.error(f"Failed to copy {image}: {e}")
                continue

        # Check if any images were copied successfully
        temp_images = list(tmp_pred_path.glob("*.nii.gz"))
        if not temp_images:
            logger.warning("No images copied to temporary directory")
            continue

        # Run prediction
        try:
            prediction(model_name, str(tmp_pred_path), str(path_output))
            logger.info("Prediction completed")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            continue

        # Remove images that were successfully predicted
        masks = list(path_output.glob("*.nii.gz"))
        if masks:
            predicted_cases = {mask.stem.split(".")[0] for mask in masks}

            for image in temp_images:
                case_name = image.stem.split("_0000")[0]
                if case_name in predicted_cases:
                    try:
                        image.unlink()
                    except OSError as e:
                        logger.error(f"Failed to remove {image}: {e}")

        # Check if there are still images left to predict
        remaining_images = list(tmp_pred_path.glob("*.nii.gz"))
        if not remaining_images:
            logger.info("All images processed successfully")
            try:
                tmp_pred_path.rmdir()
            except OSError:
                pass
            break
        else:
            logger.info(f"{len(remaining_images)} images still need processing")
    else:
        logger.warning(f"Reached maximum number of retries ({max_retries})")

    # Copy all predicted masks to the mask directory
    for mask in path_output.glob("*.nii.gz"):
        try:
            shutil.copy2(mask, mask_path / mask.name)
            logger.info(f"Copied {mask.name} to {mask_path}")
        except (OSError, shutil.Error) as e:
            logger.error(f"Failed to copy {mask}: {e}")


if __name__ == "__main__":
    main()
