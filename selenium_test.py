import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1. Project Directory Configuration
PROJECT_PATH = r"C:\Users\krish\AI_Projects\skin_cancer_project"
IMAGE_NAME = "test_lesion.jpg"  # Ensure this file exists in your folder
FULL_IMAGE_PATH = os.path.join(PROJECT_PATH, IMAGE_NAME)

# 2. Browser Setup [cite: 33, 98]
driver = webdriver.Chrome()
driver.maximize_window()
print("Starting Dermascan AI Automation...")

try:
    # 3. Open your Flask App [cite: 42, 180]
    driver.get("http://127.0.0.1:5000")
    
    # Synchronization: Wait for page to load [cite: 35, 114]
    wait = WebDriverWait(driver, 10)
    
    # 4. Upload Image [cite: 18]
    file_input = driver.find_element(By.ID, "fileInput")
    file_input.send_keys(FULL_IMAGE_PATH)
    print("Step 1: Image uploaded.")

    # 5. Fill Clinical Details [cite: 65, 138-140, 255]
    driver.find_element(By.NAME, "age").send_keys("50")
    
    # Selecting from dropdowns [cite: 29, 136, 143]
    Select(driver.find_element(By.NAME, "gender")).select_by_value("male")
    Select(driver.find_element(By.NAME, "body_part")).select_by_value("face")
    Select(driver.find_element(By.NAME, "itching")).select_by_value("yes")
    Select(driver.find_element(By.NAME, "pain")).select_by_value("no")
    Select(driver.find_element(By.NAME, "duration")).select_by_value(">1 year")
    Select(driver.find_element(By.NAME, "history")).select_by_value("growing")
    print("Step 2: Clinical form filled.")

    # 6. Capture Pre-Analysis Screenshot [cite: 96, 181]
    driver.save_screenshot("before_analysis.png")

    # 7. Execute Analysis [cite: 18, 46, 152]
    submit_btn = driver.find_element(By.XPATH, "//button[@type='submit']")
    submit_btn.click()
    print("Step 3: Analyzing... waiting for model inference.")

    # 8. Wait for Results to Appear [cite: 52, 115]
    # We wait for the 'Detected Category' card to be visible in your HTML
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Detected Category')]")))

    # 9. Verify and Final Screenshot [cite: 189, 211]
    driver.save_screenshot("final_results.png")
    print("Step 4: Success! Results captured in final_results.png")

except Exception as e:
    print(f"Error during automation: {e}")

finally:
    time.sleep(3) # Short pause to see the final screen [cite: 170]
    driver.quit() # [cite: 171]
    print("Test Script Finished.")