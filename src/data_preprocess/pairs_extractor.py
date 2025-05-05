import re
import tqdm
import json

class PairsExtractor:
    """
    PairsExtractor is a class for extracting pairs of items from a list.
    """

    def __init__(self):
        pass
    
    
    def extract_pairs(self, dataset_no_links, dataset_with_links):
        
        dataset = []
        for item_no_links, item_with_links in zip(dataset_no_links, dataset_with_links):
            assert item_no_links['title'] == item_with_links['title']

            item_no_links['text_with_links'] = item_with_links['text']
            dataset.append(item_no_links)

        dataset_map = {item['title']: item for item in dataset}

        context_pairs = []
        i = 0
        for item in tqdm.tqdm(dataset):
            links = self.extract_links(self.get_first_n_para(item['text_with_links']))
            title_1 = item['title']
            context_1 = self.get_first_n_para(item['text'])
            for link in links:
                if link in dataset_map:
                    item_2 = dataset_map[link]
                    title_2 = item_2['title']
                    context_2 = self.get_first_n_para(item_2['text'])
                    if title_1.strip().lower() == title_2.strip().lower():
                        continue

                    context_pairs.append({
                        "id": i,
                        "title_1": title_1,
                        "context_1": context_1,
                        "title_2": title_2,
                        "context_2": context_2
                    })
                    i += 1
        
        with open('../data/context_pairs.jsonl', 'w') as f:
            for item in context_pairs:
                f.write(json.dumps(item, ensure_ascii=False))
                f.write('\n')

        print(len(context_pairs))

        return context_pairs
            
    
    def extract_links(self, text):
        pattern = r'&lt;a href=\"(.*?)\"?&gt;'
        links = re.findall(pattern, text)
        return links
        
    def get_first_n_para(self, text, n_first_para=4):

        return '\n'.join(text.splitlines()[:n_first_para]) if len(text.splitlines()[:n_first_para][-1]) > 10 else '\n'.join(text.splitlines()[:n_first_para - 1])