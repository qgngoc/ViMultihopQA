
import os
import glob
import json

class WikiParser:
    def __init__(self):
        pass

    def parse(self, file_path: str, output_file: str, with_links: bool = False):
        """
        Parse the wiki dump file and extract the content.
        
        Args:
            file_path (str): Path to the wiki dump file.
            output_dir (str): Directory to save the extracted content.
        """
        output_dir = "../data/wiki_extracted"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.extract_from_file(file_path, output_dir, with_links=with_links)
        self.aggreate_to_jsonl(output_dir, os.path.join(output_dir, output_file))
        
        
        
        
    def extract_from_file(self, file_path: str, output_dir: str, with_links: bool = False):
        command = f"python -m wikiextractor.WikiExtractor {file_path} --json -l --processes 50 -o {output_dir}" if with_links else f"python -m wikiextractor.WikiExtractor {file_path} --json --processes 50 -o {output_dir}"
        os.system(command)
        
    
    def aggreate_to_jsonl(self, folder_path: str, output_file: str):
        fps = glob.glob(f"{folder_path}/**/*")
        
        ds = []
        for fp in fps:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if len(data.get("text", "").strip()) == 0:
                            continue
                        ds.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        
        
        

