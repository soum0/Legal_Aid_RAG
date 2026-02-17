# import json
# import re

# with open("data/cleaned_pages.json", "r", encoding="utf-8") as f:
#     pages = json.load(f)

# full_text = "\n\n".join(p["text"] for p in pages)

# matches = re.findall(r"\bArticle\s+\d+[A-Za-z]*", full_text, flags=re.IGNORECASE)

# print("Total 'Article X' occurrences found:", len(matches))
# print("Sample:", matches[:20])


# quick validation snippet (run in REPL or small script)
import json
with open("data/structured_articles.json","r",encoding="utf-8") as f:
    arts = json.load(f)
print("Total parsed:", len(arts))
for want in ("1","14","21","32","368"):
    found = [a for a in arts if a["article_raw_number"] == want]
    print(want, "found:", len(found))
    if found:
        print(found[0]["article_title"])
        print("pages:", found[0]["page_start"], "-", found[0]["page_end"])
        print(found[0]["text"][:200].replace("\n"," "), "\n---")
