import time
import os
import argparse
import os.path as osp
from tqdm import tqdm
import json
import subprocess
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class Downloader:
    def __init__(self, email, password, output_dir, log_file):
        self.email = email
        self.password = password
        self.output_dir = output_dir
        self.log_file = log_file

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080") 
        options.binary_location = "/usr/bin/google-chrome" 
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)


    def login_and_get_cookies(self, initial_url='https://aphasia.talkbank.org/'):
        """Login to TalkBank and return cookies."""
        self.driver.get(initial_url)
        
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, "authModals_loginLogoutBtn")))
        time.sleep(2)

        login_button = self.driver.find_element(By.ID, "authModals_loginLogoutBtn")
        login_button.click()

        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, "authModals_userName")))
        time.sleep(2)

        email_field = self.driver.find_element(By.ID, "authModals_userName")
        password_field = self.driver.find_element(By.ID, "authModals_pswd")
        email_field.send_keys(self.email)
        password_field.send_keys(self.password)

        submit_button = self.driver.find_element(By.ID, "authModals_loginBtn")
        submit_button.click()

        time.sleep(5)
        
        cookies = self.driver.get_cookies()
        cookie_string = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
        
        return cookie_string


    def get_download_links(self, page_url):
        """Extract download links from the page."""
        self.driver.get(page_url)
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body > table > tbody > tr > td:nth-child(2) > a")))
        
        download_elements = self.driver.find_elements(By.CSS_SELECTOR, "body > table > tbody > tr > td:nth-child(2) > a")
        download_links = [elem.get_attribute("href") for elem in download_elements]
        
        return download_links


    def download_files(self, cookie_string, download_links, pbar):
        """Download files using curl with cookies."""
        os.makedirs(self.output_dir, exist_ok=True)
        for link in download_links:
            file_name = os.path.join(args.output_dir, os.path.basename(link).split('&')[0])
            
            if osp.exists(file_name):
                print(f"Skipping {file_name} (already exists)")
                pbar.update(1)
                continue

            result = subprocess.run([
                "curl",
                "-J", "-O", link,
                "-H", f"Cookie: {cookie_string}",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
                "-H", f"Referer: {link}",
                "--retry", "3",
                "--retry-delay", "5",
            ], cwd=self.output_dir, capture_output=True)
            
            if result.returncode != 0:
                with open(self.log_file, 'a') as log_f:
                    log_f.write(f"Error downloading {link}: {result.stderr.decode()}\n")
                print(f"Failed to download {link}, logged the error.")

                try:
                    os.remove(file_name)
                except FileNotFoundError:
                    pass

            pbar.update(1)


    def close(self):
        """Close the web driver."""
        self.driver.quit()



if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str)
    parser.add_argument("--password", type=str)
    parser.add_argument("--initial_url", default='https://aphasia.talkbank.org/', type=str)
    parser.add_argument("--output_dir", default='/mnt/d/aphasia/dataset/videos', type=str)
    parser.add_argument("--log_file", default='download_errors.log', type=str) 
    args = parser.parse_args()

    args.log_file = args.log_file.replace('.log', time.strftime("%Y%m%d-%H%M%S") + ".log")

    downloader = Downloader(args.email, args.password, args.output_dir, args.log_file)
    cookie_string = downloader.login_and_get_cookies(args.initial_url)

    files_json_path = osp.join(osp.dirname(osp.abspath(__file__)), 'files.json')

    if osp.exists(files_json_path):
        with open(files_json_path, 'r') as f:
            file_infos = json.load(f)
            corpus_paths = sorted(file_infos.keys())
            fns = sum(file_infos.values(), [])
    else:
        raise FileNotFoundError(f"{files_json_path} not found")

    with tqdm(total=len(fns)) as pbar:
        for corpus_path in corpus_paths:
            page_url = f"https://media.talkbank.org/fileListing?&bp=media&path=/aphasia/English/Control/{corpus_path}/"
            try:
                download_links = downloader.get_download_links(page_url)
                downloader.download_files(cookie_string, download_links, pbar)
            except Exception as e:
                with open(args.log_file, 'a') as log_f:
                    log_f.write(f"Failed to download {corpus_path}: {e}\n")
                print(f"Failed to download {corpus_path}, logged the error.")

    downloader.close()
