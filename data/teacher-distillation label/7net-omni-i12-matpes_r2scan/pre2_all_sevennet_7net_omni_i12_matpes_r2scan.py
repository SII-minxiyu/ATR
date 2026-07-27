from sevennet_official_runner import run_with_settings


SETTINGS = {
    "MODEL_LABEL": "7net-omni-i12-matpes_r2scan",
    "DATA_DIR": ".",
    "OUTPUT_DIR": "./7net-omni-i12-matpes_r2scan/pred_xyz_single",
    "FIRST_IDX": 0,
    "LAST_IDX": 22,
    "ONLY_FINAL_STEP": False,
    "SEVENNET_MODEL": "7net-omni-i12",
    "SEVENNET_FILE_TYPE": None,
    "SEVENNET_MODAL_CANDIDATES": ["matpes_r2scan"],
    "ACCELERATOR_PREFERENCE": "auto",
    "SEVENNET_BATCH_SIZE": 32,
    "INPUT_FORMAT": "xyz",
    "INPUT_FILE": "/inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/meta_data/InSe_plus_pureIn_pureSe.xyz",
    "XYZ_INDEX": ":",
    "SKIP_LOG": "./7net-omni-i12-matpes_r2scan/skipped_xyz.txt",
}



if __name__ == "__main__":
    run_with_settings(SETTINGS)
