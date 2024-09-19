import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    """
    Check if the service is ready to receive requests.
    """
    retries = 0

    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_text2img_inference(inference_request):
    """
    Run txt2img inference.
    """
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img',
                                      json=inference_request, timeout=600)
    return response.json()


def run_img2img_inference(inference_request):
    """
    Run img2img inference.
    """
    response = automatic_session.post(url=f'{LOCAL_URL}/img2img',
                                      json=inference_request, timeout=600)
    return response.json()

def run_controlnet_models():
    """
    Get controlnet models.
    """
    response = automatic_session.get(url=f'http://127.0.0.1:3000/controlnet/control_types', timeout=600)
    return response.json()

def run_controlnet_preprocessor(inference_request):
    """
    Run controlnet preprocessor.
    """
    response = automatic_session.post(url=f'http://127.0.0.1:3000/controlnet/detect',
                                      json=inference_request, timeout=600)
    return response.json()

def run_upscaler_preprocessor(inference_request):
    """
    Run controlnet preprocessor.
    """
    response = automatic_session.post(url=f'http://127.0.0.1:3000/extra-single-image',
                                      json=inference_request, timeout=600)
    return response.json()


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    """
    This is the handler function that will be called by the serverless.
    """

    input_data = event.get("input", {})
    mode = input_data.get("mode")
    data = input_data.get("data", {})

    if mode == "txt2img":
        json = run_text2img_inference(data)
    elif mode == "img2img":
        json = run_img2img_inference(data)
    elif mode == "controlnet":
        json = run_controlnet_models()
    elif mode == "controlnet_detect":
        json = run_controlnet_preprocessor(data)
    elif mode == "extra-single-image":
        json = run_upscaler_preprocessor(data)
    else:
        print(f"Received event: {event}")
        raise ValueError(f"Invalid mode: {mode}")

    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return json


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
