from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_PATH = Path(__file__).parent
    ASSET_PATH = BASE_PATH / "assets"
    DIAGNOSTIC_IMAGE_PATH = ASSET_PATH / "images/diagnosis.png"
    BRAIN_IMAGE_PATH = ASSET_PATH / "images/thought.png"
    CSS_PATH = BASE_PATH / "app.css"
    FONT_PATH = BASE_PATH / "fonts/NanumGothic-regular.ttf"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INFERENCE_API_KEY = "695f9eb5021752735066a7d14fa166fa5007ff6a2dcaee6b8dae9cb4a4a69b09"

    STORY_PAGES = [
        {
            "image_path": ASSET_PATH / "images/1.png",
            "caption": "Once upon a time, in a little village, there lived a sweet girl named Cinderella. All the animals adored her and loved her dearly."
        },
        {
            "image_path": ASSET_PATH / "images/2.png",
            "caption": "Cinderella lived with her stepmother and two stepsisters named Anastasia and Drizella. They made Cinderella clean, sew, and cook all day long."
        },
        {
            "image_path": ASSET_PATH / "images/3.png",
            "caption": "Cinderella’s stepmother was jealous of her beauty and treated her with coldness and cruelty. Yet, kind-hearted Cinderella tried her best to earn their love."
        },
        {
            "image_path": ASSET_PATH / "images/4.png",
            "caption": "One day, a special invitation arrived for a grand ball at the royal palace. The king hoped that the prince would find a bride, so all the unmarried young ladies in the kingdom were invited."
        },
         {
            "image_path": ASSET_PATH / "images/5.png",
            "caption": "Cinderella was overjoyed at the thought of going to the ball. She found her mother’s old dress in the attic and decided to make it beautiful so she could wear it to the ball."
        },
         {
            "image_path": ASSET_PATH / "images/6.png",
            "caption": "Cinderella’s stepmother didn’t want her to go to the ball. So, she kept giving Cinderella more and more chores—tasks that would take her all evening to finish."
        },
         {
            "image_path": ASSET_PATH / "images/7.png",
            "caption": "While Cinderella worked, her animal friends fixed up her dress. They added pretty ribbons and beads that her stepsisters had thrown away, turning it into a beautiful gown."
         },
         {
            "image_path": ASSET_PATH / "images/8.png",
            "caption": "Cinderella was overjoyed when she saw the dress her animal friends had fixed up for her. Now, she could go to the ball too! She thanked her little friends with all her heart."
        },
         {
            "image_path": ASSET_PATH / "images/9.png",
            "caption": "But when her stepsisters saw the ribbons and beads on Cinderella’s dress, they were furious. They grabbed at the beads and ribbons, pulling them off until the dress was ruined."
        },
         {
            "image_path": ASSET_PATH / "images/10.png",
            "caption": "Heartbroken, Cinderella ran into the garden, tears streaming down her face. But suddenly, her fairy godmother appeared! With a wave of her magic wand, she turned a pumpkin into a magnificent carriage."
        },
         {
            "image_path": ASSET_PATH / "images/11.png",
            "caption": "“Bibbidi-Bobbidi-Boo!”\nIn an instant, Cinderella was transformed, dressed in a beautiful gown with sparkling glass slippers. But her fairy godmother warned her that the magic would fade at midnight."
        },
         {
            "image_path": ASSET_PATH / "images/12.png",
            "caption": "At the ball, the prince saw Cinderella and couldn’t take his eyes off her beauty. As the music began to play, the prince started to dance with the lovely Cinderella."
        },
         {
            "image_path": ASSET_PATH / "images/13.png",
            "caption": "But as the clock struck midnight, Cinderella’s magical evening came to an end. She quickly left the ballroom with only a hurried goodbye, leaving behind a single glass slipper."
        },
         {
            "image_path": ASSET_PATH / "images/14.png",
            "caption": "The prince sent out his servants to find the girl whose foot would fit the glass slipper. Despite the stepmother’s attempts to interfere, it was finally revealed that Cinderella was the true owner of the glass slipper!"
        },
          {
            "image_path": ASSET_PATH / "images/15.png",
            "caption": "Cinderella and the prince soon had their wedding, and everyone celebrated their happiness together."
        },
          {
            "image_path": ASSET_PATH / "images/16.png",
            "caption": "Amid the blessings of everyone around them, the prince and Cinderella lived happily ever after."
        }
    ]
    