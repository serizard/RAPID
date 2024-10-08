import time
import os
import os.path as osp
import numpy as np
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
    def __init__(self, email, password, output_dir):
        self.email = email
        self.password = password
        self.output_dir = output_dir

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--log-level=3")
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)


    def login_and_get_cookies(self, initial_url='https://aphasia.talkbank.org/'):
        """Login to TalkBank and return cookies."""
        self.driver.get(initial_url)
        
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, "authModals_loginLogoutBtn")))

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
            subprocess.run([
                "curl",
                "-J", "-O", link,
                "-H", f"Cookie: {cookie_string}",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
                "-H", f"Referer: {link}"
            ], cwd=self.output_dir)
            pbar.update(1)


    def close(self):
        """Close the web driver."""
        self.driver.quit()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str)
    parser.add_argument("--password", type=str)
    parser.add_argument("--initial_url", default='https://aphasia.talkbank.org/', type=str)
    parser.add_argument("--output_dir", default='/mnt/d/aphasia/dataset/videos', type=str)
    args = parser.parse_args()

    downloader = Downloader(args.email, args.password, args.output_dir)
    cookie_string = downloader.login_and_get_cookies(args.initial_url)

    with open(osp.join(osp.dirname(osp.abspath(__file__)), 'files.json'), 'r') as f:
        file_infos = json.load(f)
        corpus_paths = sorted(file_infos.keys())
        fns = sum(file_infos.values(), [])

    with tqdm(total=len(fns)) as pbar:
        for corpus_path in corpus_paths:
            page_url = f"https://media.talkbank.org/fileListing?&bp=media&path=/aphasia/English/Aphasia/{corpus_path}/"
            try:
                download_links = downloader.get_download_links(page_url)
                downloader.download_files(cookie_string, download_links, pbar)
            except Exception as e:
                print(f"Failed to download {corpus_path}: {e}")
