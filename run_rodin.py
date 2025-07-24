import os
import time
import requests
import trimesh

from constants import RODIN_BASE_URL
from secret_config import RODIN_API_KEY

headers = {
    "Authorization": f"Bearer {RODIN_API_KEY}",
    "Content-Type": "application/json"
}

def check_status(subscription_key: str) -> dict:
    url = f"{RODIN_BASE_URL}/status"
    data = {"subscription_key": subscription_key}
    for _ in range(100):
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error checking task status: {e}")
            time.sleep(5)
    raise ValueError("Failed to check task status after 100 attempts")

def download_results(task_uuid: str) -> dict:
    url = f"{RODIN_BASE_URL}/download"
    data = {"task_uuid": task_uuid}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def run_rodin(seg_rgb_path: str, output_path: str, pick_filename: list[str] = None) -> None:
    assert os.path.exists(seg_rgb_path), "Segmentation RGB image path does not exist"
    os.makedirs(output_path, exist_ok=True)

    endpoint = "https://hyperhuman.deemos.com/api/v2/rodin"
    uuids, file_names, subscription_keys = [], [], []

    files_to_process = pick_filename if pick_filename else os.listdir(seg_rgb_path)
    for file_name in files_to_process:
        suffix = os.path.splitext(file_name)[1][1:]
        if suffix.lower() not in {"jpg", "png"} or "_side" in file_name:
            continue

        file_path = os.path.join(seg_rgb_path, file_name)
        files = []
        base_name = os.path.splitext(file_name)[0]
        for side_suffix in ["", "_side", "_side_2"]:
            side_file_name = f"{base_name}{side_suffix}.{suffix}"
            side_file_path = os.path.join(seg_rgb_path, side_file_name)
            if os.path.exists(side_file_path):
                files.append(('images', open(side_file_path, 'rb')))

        response = requests.post(endpoint, files=files, headers=headers, data={"tire": "Regular", "seed": 9876})
        response_json = response.json()
        if "error" not in response_json:
            response_json["error"] = None
        assert response_json["error"] is None, f"Error running RODIN: {response_json['error']}"

        print(f"Image: {file_name}, Message: {response_json['message']}, Subscription Key: {response_json['jobs']['subscription_key']}")
        uuids.append(response_json["uuid"])
        file_names.append(os.path.splitext(file_name)[0])
        subscription_keys.append(response_json["jobs"]["subscription_key"])

    while True:
        all_done = True
        for input_file, subscription_key in zip(file_names, subscription_keys):
            status = check_status(subscription_key)
            print(f"Polling status for {input_file}")
            job_statuses = [job["status"] for job in status["jobs"]]
            print(f"\tJob statuses: {', '.join(job_statuses)}")
            if not all(status == "Done" for status in job_statuses):
                all_done = False
                break
        if all_done:
            break
        time.sleep(5)

    for input_file, uuid in zip(file_names, uuids):
        results = download_results(uuid)
        for item in results.get("list", []):
            file_name = item["name"]
            suffix = os.path.splitext(file_name)[1][1:]
            dest_file_name = f"{input_file}.glb" if suffix == "glb" else f"{input_file}_{file_name}"
            dest_path = os.path.join(output_path, dest_file_name)
            with open(dest_path, 'wb') as f:
                f.write(requests.get(item["url"]).content)
            print(f"Downloaded {dest_path}")

def glb_to_obj(mesh_folder: str, output_folder: str) -> None:
    for file_name in os.listdir(mesh_folder):
        if not file_name.endswith(".glb"):
            continue
        mesh_path = os.path.join(mesh_folder, file_name)
        mesh = trimesh.load(mesh_path, force='scene')
        if isinstance(mesh, trimesh.Scene):
            combined_mesh = trimesh.util.concatenate([geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)])
        else:
            combined_mesh = mesh
        output_subfolder = os.path.join(output_folder, os.path.splitext(file_name)[0])
        os.makedirs(output_subfolder, exist_ok=True)
        combined_mesh.export(os.path.join(output_subfolder, f"{os.path.splitext(file_name)[0]}.obj"))

if __name__ == "__main__":
    run_rodin("/build_kinematic/123001/segmented_rgb", "/build_kinematic/123001/mesh")
