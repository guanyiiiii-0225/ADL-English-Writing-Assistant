import difflib

def find_diff(text1, text2, filename):
    text1 = text1.split('.')
    text1 = [(x + '.').strip() for x in text1]
    text1 = text1[:-1]

    text2 = text2.split('.')
    text2 = [(x + '.').strip() for x in text2]
    text2 = text2[:-1]


    differ = difflib.HtmlDiff()
    html_diff = differ.make_file(text1, text2)

    with open(filename, "w", encoding="utf-8") as file:
        file.write(html_diff)