import yt_dlp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import os, time


class TwitchHandler:
    """Handles Twitch videos grabbing and downloading using Selenium"""

    def __init__(self, channel_link, max_videos, output_dir):
        self.channel_link = channel_link
        self.max_videos = max_videos
        self.output_dir = output_dir


    def get_all_videos(self,):
        """Returns all detected Call of Duty: Black Ops 6 videos on a user's channel"""

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(options=options)
        driver.get(f"{self.channel_link}/videos?filter=all&sort=time")
        desired_game = "Call of Duty: Black Ops 7"
        video_urls = set()
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        print("Fetching videos")

        while True:
            video_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/videos/')]")
            current_video_urls = [element.get_attribute('href') for element in video_elements]

            new_video_urls = set(current_video_urls) - video_urls
            if not new_video_urls:
                print("No new videos found.")
                break
            
            video_urls.update(new_video_urls)

            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(2)

            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@href, '/videos/')]"))
            )

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("Reached the end of the page, no more content.")
                break
            last_height = new_height

        print(f"Found {len(video_urls)} unique videos.")

        filtered_video_urls = []
        for url in video_urls:
            driver.get(url)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/directory/')]"))
                )
                game_element = driver.find_element(By.XPATH, "//a[contains(@href, '/directory/')]")
                game_name = game_element.text.strip().lower()

                if game_name == desired_game.lower():
                    filtered_video_urls.append(url)

            except Exception as e:
                print(f"Error loading video {url}: {e}")

        print(f"Found {len(filtered_video_urls)} videos for {desired_game}.")
        for video in filtered_video_urls:
            print(video)
        
        driver.quit()
        return filtered_video_urls
    

    def download_video(self, video, name):
        """Downloads a single video from Twitch using yt-dlp"""

        save_path = f"{self.output_dir}/Downloads"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ydl_opts = {
            'outtmpl': os.path.join(save_path, f'{name}.%(ext)s'),
            'format': 'best',
        }

        try:
            print(f"Downloading: {video}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video])
            print(f"Download completed for: {video}")
        except Exception as e:
            print(f"Error downloading {video}: {e}")


    def download_channel_videos(self, links):
        """Downloads videos from the grabbed Twitch links"""

        for i in range(self.max_videos):
            print(f"Downloading Video {i+1} from {links[i]}")
            self.download_video(links[i], f'{i}')

            for file in os.listdir(f"{self.output_dir}/Downloads"):
                if not file.endswith('.mp4') or 'temp' in file:
                    os.remove(f"{self.output_dir}/Downloads/{file}")