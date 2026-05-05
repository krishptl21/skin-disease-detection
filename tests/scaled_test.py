import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1. SETUP
driver = webdriver.Chrome()
driver.maximize_window()
BASE_URL = "http://127.0.0.1:5000"

# Set up paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "test_images")
SCREENSHOT_DIR = os.path.join(SCRIPT_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# 2. TEST DATA (Scenarios based on your PNG files)
scenarios = [
    {"img": "mel.png", "age": "65", "hist": "bleeding", "label": "Scenario_1_Melanoma"},
    {"img": "akeic.png", "age": "55", "hist": "growing", "label": "Scenario_2_AKIEC"},
    {"img": "bkl.png", "age": "42", "hist": "stable", "label": "Scenario_3_Benign"}
]

def run_test_case(data):
    try:
        print(f"Starting automation: {data['label']}")
        driver.get(BASE_URL)
        wait = WebDriverWait(driver, 20) # Handles synchronization

        # Upload Image
        img_path = os.path.join(IMG_DIR, data["img"])
        driver.find_element(By.ID, "fileInput").send_keys(img_path)

        # Form Filling 
        driver.find_element(By.NAME, "age").send_keys(data["age"])
        Select(driver.find_element(By.NAME, "gender")).select_by_value("male")
        Select(driver.find_element(By.NAME, "body_part")).select_by_value("face")
        Select(driver.find_element(By.NAME, "itching")).select_by_value("yes")
        Select(driver.find_element(By.NAME, "pain")).select_by_value("no")
        Select(driver.find_element(By.NAME, "duration")).select_by_value(">1 year")
        Select(driver.find_element(By.NAME, "history")).select_by_value(data["hist"])

        # Click Submit
        driver.find_element(By.XPATH, "//button[@type='submit']").click()

        # Wait for Result Card
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "result-card")))
        
        # Save Screenshot
        driver.save_screenshot(os.path.join(SCREENSHOT_DIR, f"{data['label']}.png"))
        print(f"Successfully logged result: {data['label']}")

    except Exception as e:
        print(f"Error in {data['label']}: {e}")

# 3. EXECUTION
print("Dermascan AI: Starting Multimodal Automation...")
for case in scenarios:
    run_test_case(case)

driver.quit()
print("Automation Finished.")