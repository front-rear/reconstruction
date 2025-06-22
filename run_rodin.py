import os
import time
import requests
import trimesh

from constants import RODIN_BASE_URL
from secret_config import RODIN_API_KEY

from typing import List, Dict, Any

# Define the headers for the requests
headers = {
    "Authorization": f"Bearer {RODIN_API_KEY}",
    "Content-Type": "application/json"
}

# Function to check the status of a task
# Typical response:
# {
#   "error": "OK",
#   "jobs": [
#     {
#       "uuid": "text",
#       "status": "Waiting"
#     }
#   ]
# }
def check_status(subscription_key: str):
    url = f"{RODIN_BASE_URL}/status"
    data = {"subscription_key": subscription_key}
    for i in range(100):
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error while checking the status of the task: {e}")
            time.sleep(5)
    raise ValueError("Failed to check the status of the task after 100 attempts.")

# Function to download the results of a task
# Typical response:
# {
#   "error": "OK",
#   "list": [
#     {
#       "url": "text",
#       "name": "text"
#     }
#   ]
# }
def download_results(task_uuid: str) -> Dict[str, Any]:
    url = f"{RODIN_BASE_URL}/download"
    data = {"task_uuid": task_uuid}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def run_rodin(seg_rgb_path: str, ouput_path: str, pick_filename: List[str] = []):
    assert os.path.exists(seg_rgb_path), "The segmentation RGB image path does not exist."

    if not os.path.exists(ouput_path):
        os.makedirs(ouput_path)

    # Constants
    ENDPOINT = "https://hyperhuman.deemos.com/api/v2/rodin"

    # Send requests
    uuids: List[str] = []
    file_names: List[str] = []
    subscription_keys: List[str] = []
    for file_name in pick_filename if pick_filename else os.listdir(seg_rgb_path):
        suffix: str = os.path.splitext(file_name)[1][1:]
        if (suffix != "jpg" and suffix != "png"):
            continue
        if "_side" in file_name:
            continue
        file_path = os.path.join(seg_rgb_path, file_name)

        # Prepare the multipart form data
        # files = {
        #     'images': (os.path.basename(file_path), image_data, 'image/%s' % suffix),
        #     "prompt": (None, "An empty opened refrigerator with a blue door and a yellow door. sharp edges, simple geometry.")
        # }
        files = []
        side_view_path = file_path.replace(file_name, os.path.splitext(file_name)[0] + "_side." + suffix)
        if os.path.exists(side_view_path):
            print("Found side view image: %s" % side_view_path)
            files.append(('images', open(side_view_path, 'rb')))
        side_view_path = file_path.replace(file_name, os.path.splitext(file_name)[0] + "_side_2." + suffix)
        if os.path.exists(side_view_path):
            print("Found side view image: %s" % side_view_path)
            files.append(('images', open(side_view_path, 'rb')))
        files.append(('images', open(file_path, 'rb')))

        # Prepare the headers
        headers = {'Authorization': f'Bearer {RODIN_API_KEY}'}

        # Make the POST request
        response = requests.post(ENDPOINT, files=files, headers=headers,
                                 data={"tire": "Regular", "seed": 9876})

        # Parse and return the JSON response
        response_json = response.json()
        if "error" not in response_json:
            response_json["error"] = None
        assert response_json['error'] is None, "Error occurred while running RODIN: %s" % response_json['error']

        print("Image: %s, Message: %s, Error: %s, Subscription Key: %s" %
              (file_name, response_json['message'], response_json['error'], response_json['jobs']['subscription_key']))

        uuids.append(response_json['uuid'])
        file_names.append(file_name.removesuffix('.' + suffix))
        subscription_keys.append(response_json['jobs']['subscription_key'])

    # Poll for the jobs to complete
    # Typical response:
    # {
    #     "error": null,
    #     "message": "Submitted.",
    #     "uuid": "example-task-uuid",
    #     "jobs": {
    #         "uuids": ["job-uuid-1", "job-uuid-2"],
    #         "subscription_key": "example-subscription-key"
    #     }
    # }
    # Poll the status endpoint every 5 seconds until the task is done
    while True:
        images_all_done = True
        for input_file, subscription_key in zip(file_names, subscription_keys):
            status_response = check_status(subscription_key)

            assert "error" not in status_response, "Error occurred while checking the status of the task."
            print("Polling status for %s" % input_file)
            for job in status_response['jobs']:
                print(f"\tjob {job['uuid']}: {job['status']}")

            all_done = all(job['status'] == 'Done' for job in status_response['jobs'])
            if not all_done:
                images_all_done = False
                break

        if images_all_done:
            break
        time.sleep(5)

    for input_file, uuid in zip(file_names, uuids):
        # Download the results once the task is done
        download_response = download_results(uuid)
        download_items = download_response['list']

        # Print the download URLs and download them locally.
        for item in download_items:
            file_name: str = item['name']
            suffix = os.path.splitext(item['name'])[1][1:]
            print(f"File Name: {item['name']}, URL: {item['url']}")

            if suffix == "glb":
                file_name = "%s.glb" % input_file
            else:
                file_name = "%s_%s" % (input_file, file_name)

            dest_fname = os.path.join(ouput_path, file_name)
            with open(dest_fname, 'wb') as f:
                response = requests.get(item['url'])
                f.write(response.content)
                print(f"Downloaded {dest_fname}")

def glb_to_obj(mesh_folder: str, output_folder: str):
    """
    Convert a GLB file to OBJ format using trimesh library.
    Handles multiple meshes within the GLB file and provides error handling.
    Also exports textures as PNG images if embedded in materials of the GLB file, saving them directly into the output directory with related names based on the input filename.
    """

    for filename in os.listdir(mesh_folder):
        suffix = os.path.splitext(filename)[1]
        if suffix != ".glb":
            continue
        name_part = filename.removesuffix(".glb")

        # Load the .glb file
        scene = trimesh.load(os.path.join(mesh_folder, filename), force='scene')

        # Handle both single mesh and scene with multiple meshes
        if isinstance(scene, trimesh.Scene):
            # Combine all meshes in the scene
            meshes = []
            for geometry in scene.geometry.values():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            if not meshes:
                raise ValueError("No valid meshes found in GLB file %s" % filename)
            combined_mesh = trimesh.util.concatenate(meshes)
        else:
            combined_mesh = scene

        # Export to OBJ format
        export_folder = os.path.join(output_folder, name_part)
        os.makedirs(export_folder, exist_ok=True)

        # combined_mesh.apply_transform(constants.X_ROTATE_180)
        combined_mesh.export(os.path.join(export_folder, f"{name_part}.obj"), file_type='obj')

if __name__ == '__main__':
    run_rodin("/home/rvsa/gary318/build_kinematic/input_rgbd/123001/segmented_rgb", "/home/rvsa/gary318/build_kinematic/input_rgbd/123001/mesh")
