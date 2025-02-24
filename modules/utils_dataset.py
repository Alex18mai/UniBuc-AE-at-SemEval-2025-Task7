import csv
import ast

cached_posts = None
cached_factchecks = None
parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s



def read_posts(path_csv):
    global cached_posts
    if cached_posts is not None:
        return cached_posts
    
    posts = []
    with open(path_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file) 

        for row in list(reader):
            row_id = int(row['post_id'])
            
            text_multi = []
            text_eng = []

            text_tuple = parse_col(row["text"])
            if text_tuple:
                text_multi.append(text_tuple[0].strip())
                text_eng.append(text_tuple[1].strip())

            ocr_tuple = parse_col(row["ocr"])
            for ocr_line in ocr_tuple:
                if ocr_line:
                    text_multi.append(ocr_line[0])
                    text_eng.append(ocr_line[1])

            post = {
                "id" : row_id,
                "text_multi" : " ".join(text_multi),
                "text_eng" : " ".join(text_eng)
            }
            posts.append(post)

    cached_posts = posts
    return posts


def read_factchecks(path_csv):
    global cached_factchecks
    if cached_factchecks is not None:
        return cached_factchecks
    
    factchecks = []
    with open(path_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file) 

        for row in list(reader):
            row_id = int(row['fact_check_id'])
            
            text_multi = []
            text_eng = []

            claim_tuple = parse_col(row["claim"])
            if claim_tuple:
                text_multi.append(claim_tuple[0].strip())
                text_eng.append(claim_tuple[1].strip())

            title_tuple = parse_col(row["title"])
            if title_tuple:
                text_multi.append(title_tuple[0].strip())
                text_eng.append(title_tuple[1].strip())

            factcheck = {
                "id" : row_id,
                "text_multi" : " ".join(text_multi),
                "text_eng" : " ".join(text_eng)
            }
            factchecks.append(factcheck)

    cached_factchecks = factchecks
    return factchecks