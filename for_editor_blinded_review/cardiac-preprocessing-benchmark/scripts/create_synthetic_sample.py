from pathlib import Path

import numpy as np
from skimage.draw import ellipse

from cardiac_image_system.core.io import save_grayscale_image


def main():
    out = Path("data/sample")
    out.mkdir(parents=True, exist_ok=True)

    rows = ["patient_id,phase,image_path,mask_path"]
    rng = np.random.default_rng(2026)
    for i in range(1, 4):
        for phase in ["diastole", "systole"]:
            image = rng.normal(0.08, 0.02, size=(128, 128)).astype("float32")
            mask = np.zeros_like(image, dtype=bool)
            r = 30 if phase == "diastole" else 24
            c = 25 if phase == "diastole" else 20
            rr, cc = ellipse(64, 64, r, c, shape=image.shape)
            mask[rr, cc] = True
            image[mask] += 0.5
            image = np.clip(image, 0, 1)

            image_path = out / f"P{i:03d}_{phase}.png"
            mask_path = out / f"P{i:03d}_{phase}_mask.png"
            save_grayscale_image(image_path, image)
            save_grayscale_image(mask_path, mask.astype("float32"))
            rows.append(f"P{i:03d},{phase},{image_path},{mask_path}")

    (out / "manifest.csv").write_text("\n".join(rows), encoding="utf-8")
    print("Synthetic sample created in data/sample")


if __name__ == "__main__":
    main()
